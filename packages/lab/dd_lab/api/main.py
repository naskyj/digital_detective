"""Digital Detective local application — FastAPI backend.

One process that does collection, ingest, agent orchestration, querying and
reporting, with a small static single-page UI. Runs on localhost only.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dd_core.casedir import CaseLayout
from dd_core.custody import CustodyLog
from dd_core.manifest import Manifest
from dd_lab.agent.audit import AuditLog
from dd_lab.agent.runner import run_agent, run_pipeline
from dd_lab.agent.tools import InvestigationDeps, correlate_entities, query_warehouse, search_keyword
from dd_lab.api.jobs import JobManager
from dd_lab.api.settings import AppSettings, describe_llm
from dd_lab.case import CaseMeta, CaseStatus, EvidenceSourceMeta, create_case, list_cases, parse_keywords
from dd_lab.verification import verify_cached

log = logging.getLogger("dd_lab.api")

UI_DIR = Path(__file__).resolve().parent.parent / "ui"

app = FastAPI(title="Digital Detective", version="0.1.0")
_settings = AppSettings.load()
_jobs = JobManager(Path(_settings.cases_root) / ".dd_jobs.sqlite")


def settings() -> AppSettings:
    return _settings


def _layout(case_id: str) -> CaseLayout:
    return CaseLayout(_settings.cases_root, case_id)


# --------------------------------------------------------------------- models


class SettingsIn(BaseModel):
    cases_root: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    use_agent: bool | None = None
    classifier_model_dir: str | None = None
    use_zeroshot: bool | None = None
    use_judge: bool | None = None
    judge_llm_provider: str | None = None
    judge_llm_model: str | None = None
    judge_llm_base_url: str | None = None
    judge_llm_api_key: str | None = None


class NewCaseIn(BaseModel):
    case_id: str | None = None  # generated (dd_lab.case.new_case_id) if blank
    name: str
    description: str = ""
    keywords: str = ""


class AddSourceIn(BaseModel):
    source_path: str
    source_label: str
    apps: list[str] | None = None  # None = every profile found
    memory_image_path: str | None = None
    operator: str = ""


class ImageListIn(BaseModel):
    image_path: str


class QueryIn(BaseModel):
    sql: str


# --------------------------------------------------------------------- settings


@app.get("/api/settings")
def get_settings() -> dict:
    s = _settings
    return {
        "cases_root": s.cases_root,
        "use_agent": s.use_agent,
        "classifier_model_dir": s.classifier_model_dir,
        "use_zeroshot": s.use_zeroshot,
        "llm": describe_llm(s.agent_config()),
        "use_judge": s.use_judge,
        "judge_llm": describe_llm(s.judge_agent_config()),
    }


@app.post("/api/settings/test-llm")
def test_llm() -> dict:
    from dd_lab.agent.config import test_connection

    return test_connection(_settings.agent_config())


@app.post("/api/settings/test-judge-llm")
def test_judge_llm() -> dict:
    from dd_lab.agent.config import test_connection

    return test_connection(_settings.judge_agent_config())


@app.put("/api/settings")
def put_settings(body: SettingsIn) -> dict:
    global _jobs
    data = body.model_dump(exclude_none=True)
    for k, v in data.items():
        setattr(_settings, k, v)
    _settings.save()
    _jobs = JobManager(Path(_settings.cases_root) / ".dd_jobs.sqlite")
    return get_settings()


# ------------------------------------------------------------ image acquisition
#
# The lab deliberately does NOT do live evidence collection — that's
# dd_collector's job, run as its own minimal tool directly on (or against a
# copy of) the examined machine. The lab only ever reads evidence that
# already exists: either a case a collector run produced, or a disk image
# extracted below. Keeping that boundary means the lab's heavier dependencies
# (DuckDB, the classifier, the agent) never need to run anywhere near a live
# target, and a live acquisition never depends on anything the lab needs.


def _is_real_image(path: str) -> bool:
    """True if `dissect` can open `path` as an image (a real image file, or
    a folder shaped like an OS root - its own directory loader handles
    both). False for anything else, including a plain folder of already-
    extracted app subfolders - the caller falls back to the folder-import
    path for those instead of a raised ImageUnavailable."""
    from dd_lab.imaging import ImageUnavailable, open_image

    try:
        open_image(path)
        return True
    except ImageUnavailable:
        return False


def _not_recognised_error(path: str) -> str:
    return f"{path} is not a recognisable image, OS-root folder, or evidence folder"


@app.post("/api/image/list")
def image_list(body: ImageListIn) -> dict:
    """List the Electron/Chromium app profiles found at `image_path` — no
    extraction yet. Accepts a real image file, a folder shaped like an OS
    root, OR a plain folder holding one subfolder per already-extracted
    app (three different input shapes, one field) — tries the image/OS-root
    path first via `dissect`, falls back to a flat folder scan when that
    doesn't apply and the path is a directory. Fails cleanly (400, not a
    crash) if neither recognises it, or the `image` extra isn't installed."""
    from pathlib import Path

    from dd_lab.imaging import discover_apps, discover_apps_in_folder, open_image

    p = Path(body.image_path)
    if _is_real_image(body.image_path):
        target = open_image(body.image_path)
        apps = discover_apps(target)
    elif p.is_dir():
        apps = discover_apps_in_folder(p)
    else:
        raise HTTPException(400, _not_recognised_error(body.image_path))
    return {
        "apps": [{"name": a.name, "profile_dir": str(a.profile_dir), "version": a.version} for a in apps],
    }


# --------------------------------------------------------------------- cases


@app.get("/api/cases")
def get_cases() -> dict:
    cases = list_cases(_settings.cases_root)
    for c in cases:
        if c.status == CaseStatus.RUNNING and not _jobs.active_for(c.case_id):
            # Same reconciliation as get_case(): the job that set this is
            # gone (app closed/restarted mid-run) with nothing left alive to
            # update case.json - fix it here so the case list (and its
            # running/completed/stopped counts) reflect reality.
            c.status = CaseStatus.ERROR
            c.last_message = "Analysis was interrupted (the application was closed or restarted mid-run)."
            c.save(_layout(c.case_id))

    counts = {"total": len(cases), "running": 0, "completed": 0, "stopped": 0}
    for c in cases:
        if c.status == CaseStatus.RUNNING:
            counts["running"] += 1
        elif c.status == CaseStatus.COMPLETED:
            counts["completed"] += 1
        elif c.status in (CaseStatus.STOPPED, CaseStatus.ERROR):
            counts["stopped"] += 1
    return {"counts": counts, "cases": [c.model_dump(mode="json") for c in cases]}


@app.post("/api/cases")
def post_case(body: NewCaseIn) -> dict:
    """Create a case's metadata (name/description/keywords) - no evidence
    required yet. `POST /api/cases/{id}/sources` adds evidence sources to
    it, at creation time or any time after."""
    meta = create_case(
        _settings.cases_root, body.case_id, body.name, body.description,
        parse_keywords(body.keywords),
    )
    return meta.model_dump(mode="json")


@app.post("/api/cases/{case_id}/sources")
def add_source(case_id: str, body: AddSourceIn) -> dict:
    """Add one evidence source (an image, an OS-root folder, or a flat
    evidence folder - same auto-detection as `/api/image/list`) to a case -
    its first source, or its Nth. `source_label` prefixes every app this
    source contributes (see `dd_lab.imaging.acquire.source_app_prefix`) so
    two sources' identically-named apps (two people's "discord") land in
    separate evidence folders instead of silently merging into one.

    Reuses the case's EXISTING signing key when one is already there (a 2nd+
    source) rather than generating a new one, which would otherwise change
    the case's signing identity out from under its own chain of custody -
    the one thing that made adding a source through the old
    `/api/image/extract` unsafe to do more than once per case by hand."""
    from pathlib import Path

    from dd_lab.imaging import acquire_from_folder, acquire_from_image

    layout = _layout(case_id)
    meta = CaseMeta.load_or_none(layout)
    if meta is None:
        raise HTTPException(404, "case not found - create it first with POST /api/cases")
    if _jobs.active_for(case_id):
        raise HTTPException(409, "a job is already running for this case")

    is_image = _is_real_image(body.source_path)
    if not is_image and not Path(body.source_path).is_dir():
        raise HTTPException(400, _not_recognised_error(body.source_path))

    from dd_lab.imaging.acquire import source_app_prefix

    new_prefix = source_app_prefix(body.source_label)
    clashing = next(
        (s for s in meta.sources if s.label != body.source_label
         and source_app_prefix(s.label) == new_prefix),
        None,
    )
    if clashing:
        raise HTTPException(
            400,
            f'"{body.source_label}" sanitises to the same evidence-folder prefix as this '
            f'case\'s existing source "{clashing.label}" - their apps would land in the same '
            f"folders and merge. Use a more distinct label.",
        )

    existing_key = layout.dir / "keys" / "collector_key.pem"
    signing_key_pem = existing_key.read_bytes() if existing_key.is_file() else None

    def work(progress) -> dict:
        acquire = acquire_from_image if is_image else acquire_from_folder
        result = acquire(
            body.source_path, _settings.cases_root,
            apps=body.apps, memory_image_path=body.memory_image_path,
            case_id=case_id, operator=body.operator, signing_key_pem=signing_key_pem,
            source_label=body.source_label, progress=progress,
        )
        m = CaseMeta.load_or_none(layout) or meta
        m.sources.append(EvidenceSourceMeta(
            label=body.source_label, kind="image" if is_image else "folder",
            path=body.source_path, operator=body.operator, apps=result.apps,
        ))
        if m.status == CaseStatus.NEW:
            m.status = CaseStatus.COMPLETED  # evidence exists now; "Run analysis" is the next step
        m.save(layout)
        return {
            "case_id": result.case_id, "files": result.file_count,
            "bytes": result.total_bytes, "fingerprint": result.public_key_fingerprint,
            "image_sha256": result.image_sha256, "apps": result.apps, "warnings": result.warnings,
        }

    job_id = _jobs.submit("add_source", work, case_id=case_id)
    return {"job_id": job_id}


@app.get("/api/cases/{case_id}")
def get_case(case_id: str) -> dict:
    layout = _layout(case_id)
    meta = CaseMeta.load_or_none(layout)
    if not layout.manifest.is_file():
        if meta is None:
            raise HTTPException(404, "case not found")
        # A case can now exist with metadata and no evidence yet - sources
        # are added over time (POST .../sources), not necessarily all at
        # creation - so this is a normal state to render, not an error.
        return {
            "meta": meta.model_dump(mode="json"),
            "integrity": None,
            "evidence": {"apps": [], "files": 0, "bytes": 0, "acquired": None, "host": None},
            "warehouse": None,
            "reports": [],
            "active_job": _jobs.active_for(case_id) or _last_job(case_id),
        }
    meta = meta or CaseMeta(case_id=case_id, name=case_id)
    manifest = Manifest.read(layout.dir)
    v, verified_ts = verify_cached(_settings.cases_root, case_id)

    summary = None
    if layout.warehouse.is_file():
        from dd_lab.etl.warehouse import Warehouse

        wh = Warehouse(layout.warehouse)
        try:
            wh.build_views()
            summary = {
                "artifacts": wh.count(),
                "by_category": wh.query("SELECT * FROM v_by_category"),
                "by_source": wh.query("SELECT * FROM v_source_breakdown"),
                "high_priority": wh.query("SELECT count(*) AS n FROM v_high_priority")[0]["n"],
            }
        finally:
            wh.close()

    active_job = _jobs.active_for(case_id) or _last_job(case_id)
    job_is_active = active_job and active_job["status"] in ("queued", "running")
    if meta.status == CaseStatus.RUNNING and not job_is_active:
        # The job that set status=running is gone (the app was closed/restarted
        # mid-run, or the process died) - nothing was left alive to update
        # case.json when that happened, so it would otherwise say "running"
        # forever. Reconcile here, on read, so the UI reflects reality.
        meta.status = CaseStatus.ERROR
        meta.last_message = "Analysis was interrupted (the application was closed or restarted mid-run)."
        meta.save(layout)

    reports = sorted(layout.reports.glob("*.html")) if layout.reports.is_dir() else []
    return {
        "meta": meta.model_dump(mode="json"),
        "integrity": {"ok": v.ok, "signature_valid": v.signature_valid, "files": v.checked,
                      "issues": [i.model_dump() for i in v.issues[:20]],
                      "verified_ts": verified_ts.isoformat()},
        "evidence": {
            "apps": [t.name for t in manifest.target_apps],
            "files": manifest.entry_count, "bytes": manifest.total_bytes,
            "acquired": manifest.completed_ts.isoformat(),
            "host": manifest.host.hostname,
        },
        "warehouse": summary,
        "reports": [r.name for r in reports],
        "active_job": active_job,
    }


def _last_job(case_id: str) -> dict | None:
    recent = _jobs.list(case_id, limit=1)
    return _jobs.get(recent[0]["id"]) if recent else None


@app.post("/api/cases/{case_id}/verify")
def reverify_case(case_id: str) -> dict:
    layout = _layout(case_id)
    if not layout.manifest.is_file():
        raise HTTPException(404, "case not found")
    v, ts = verify_cached(_settings.cases_root, case_id, force=True)
    return {"ok": v.ok, "signature_valid": v.signature_valid, "files": v.checked,
            "issues": [i.model_dump() for i in v.issues[:50]], "verified_ts": ts.isoformat()}


@app.post("/api/cases/{case_id}/run")
def run_case(case_id: str) -> dict:
    layout = _layout(case_id)
    if not layout.manifest.is_file():
        raise HTTPException(404, "case not found")
    if _jobs.active_for(case_id):
        raise HTTPException(409, "a job is already running for this case")

    use_agent = _settings.use_agent
    cfg = _settings.agent_config()
    judge_cfg = _settings.judge_agent_config()
    enable_judge = _settings.use_judge

    def work(progress) -> dict:
        # `progress` streams into the job's live log in real time as each
        # stage/tool-call happens - not just a message at the very start and
        # very end, for both the deterministic pipeline and the LLM agent.
        clf = _settings.build_classifier()
        if use_agent:
            outcome = run_agent(
                _settings.cases_root, case_id, cfg, classifier=clf,
                progress=progress, judge_cfg=judge_cfg, enable_judge=enable_judge,
            )
        else:
            outcome = run_pipeline(
                _settings.cases_root, case_id, classifier=clf,
                progress=progress, judge_cfg=judge_cfg, enable_judge=enable_judge,
            )
        return {"ok": outcome.ok, "summary": outcome.summary, "report": outcome.report_path}

    job_id = _jobs.submit("run" if not use_agent else "agent", work, case_id=case_id)
    return {"job_id": job_id}


@app.get("/api/cases/{case_id}/artifact-tree")
def case_artifact_tree(case_id: str) -> dict:
    """Per-app, per-source-type artifact counts — backs the case page's left
    navigation tree (app -> Local Storage / Cache / Memory / ... , each with
    a count), the same drill-down shape a tool like Magnet AXIOM uses."""
    deps = InvestigationDeps(cases_root=Path(_settings.cases_root), case_id=case_id)
    if not deps.layout.warehouse.is_file():
        raise HTTPException(409, "case not processed yet")
    from dd_lab.etl.warehouse import Warehouse

    wh = Warehouse(deps.layout.warehouse)
    try:
        rows = wh.query(
            "SELECT app, source_type, count(*) AS n FROM artifacts "
            "GROUP BY app, source_type ORDER BY app, n DESC"
        )
    finally:
        wh.close()
    apps: dict[str, list[dict]] = {}
    totals: dict[str, int] = {}
    for r in rows:
        apps.setdefault(r["app"], []).append({"source_type": r["source_type"], "n": r["n"]})
        totals[r["app"]] = totals.get(r["app"], 0) + r["n"]
    ordered = sorted(apps, key=lambda a: totals[a], reverse=True)
    meta = CaseMeta.load_or_none(deps.layout)
    return {
        "apps": [
            {"app": a, "total": totals[a], "sources": apps[a],
             "source_label": _source_label_for_app(a, meta.sources if meta else [])}
            for a in ordered
        ],
    }


def _source_label_for_app(app: str, sources: list[EvidenceSourceMeta]) -> str | None:
    """Which registered evidence source (if any) produced this `app` entry
    - matched by the exact prefix `acquire_from_image`/`acquire_from_folder`
    gave its apps (see `source_app_prefix`), not by trying to reverse-parse
    a label back out of the (already-sanitised) app string. None for an app
    from a case with no labelled sources (every case before this feature,
    and any case whose apps were added with `source_label` left blank) -
    the UI renders those exactly as before, no source-grouping level."""
    from dd_lab.imaging.acquire import source_app_prefix

    for s in sources:
        if app.startswith(source_app_prefix(s.label)):
            return s.label
    return None


def _derive_cache_url(source_locator: str) -> str:
    from dd_lab.parsers.cache import raw_key_from_locator, url_from_cache_key

    return url_from_cache_key(raw_key_from_locator(source_locator))


@app.get("/api/cases/{case_id}/artifacts")
def case_artifacts(
    case_id: str,
    category: str | None = None,
    app: str | None = None,
    source_type: str | None = None,
    q: str | None = None,
    limit: int = 200,
    offset: int = 0,
) -> dict:
    deps = InvestigationDeps(cases_root=Path(_settings.cases_root), case_id=case_id)
    if not deps.layout.warehouse.is_file():
        raise HTTPException(409, "case not processed yet")
    where, params = [], []
    for col, val in (("category", category), ("app", app), ("source_type", source_type)):
        if val:
            where.append(f"{col} = ?")
            params.append(val)
    if q:
        # source_locator is included so a process detail view can filter
        # memory_string rows down to one PID (locators are "pid<PID>:string#n"
        # for process-scoped carving - see dd_lab/memory/analyze.py) without
        # a dedicated endpoint; it's also just generally useful to search.
        where.append("(key ILIKE ? OR value ILIKE ? OR source_locator ILIKE ?)")
        params += [f"%{q}%", f"%{q}%", f"%{q}%"]
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    from dd_lab.etl.warehouse import Warehouse

    wh = Warehouse(deps.layout.warehouse)
    try:
        total = wh.query(f"SELECT count(*) AS n FROM artifacts {clause}", list(params))[0]["n"]
        rows = wh.query(
            f"SELECT app, source_type, category, confidence, key, substr(value,1,300) AS value, "
            f"record_ts, source_file, source_locator, artifact_id, regex_vote, keyword_vote, "
            f"ner_vote, model_vote, evidence_spans, deleted FROM artifacts {clause} "
            f"ORDER BY (category != 'GENERAL') DESC, confidence DESC NULLS LAST, key, artifact_id "
            f"LIMIT {int(limit)} OFFSET {int(offset)}",
            list(params),
        )
        facets = {
            "apps": [r["app"] for r in wh.query(
                "SELECT DISTINCT app FROM artifacts ORDER BY app")],
            "source_types": [r["source_type"] for r in wh.query(
                "SELECT DISTINCT source_type FROM artifacts ORDER BY source_type")],
        }
        _enrich_artifact_rows(wh, rows)
    finally:
        wh.close()
    return {"rows": rows, "total": total, "shown": len(rows), "facets": facets}


def _enrich_artifact_rows(wh, rows: list[dict]) -> None:
    """Add the derived columns the table/CSV export need beyond what the
    warehouse stores directly, then drop `source_locator` (an internal
    encoding no UI/export column should need to know) - the single place
    both `/artifacts` and `/artifacts/export.csv` get this from, so they
    never drift apart on what a cache/local-storage/... row's extra
    columns actually mean.

    cache rows get `url`/`cache_offset`/`filename` (see cache.py);
    local_storage/session_storage/indexeddb rows get `latest` - True for
    the row holding the CURRENT value of its (app, key), by LevelDB
    sequence number (the only recency signal these three have - none of
    them set record_ts). Historical rows are marked, not dropped: a prior
    value is real evidence, not noise (see this session's `deleted` flag
    work) - `latest` just makes "the current one" findable. Computed
    across EVERY row for the relevant (app, source_type) pairs, not just
    the ones on this page/in this filtered view - the current value of a
    key might not be on the page being rendered right now.
    """
    from dd_lab.parsers.base import seq_from_locator
    from dd_lab.parsers.cache import cache_offset_from_locator, url_filename_hint

    seq_types = {"local_storage", "session_storage", "indexeddb"}
    present = {(r["app"], r["source_type"]) for r in rows if r["source_type"] in seq_types}
    best_seq: dict[tuple[str, str, str], int] = {}
    if present:
        where = " OR ".join(["(app = ? AND source_type = ?)"] * len(present))
        params: list[str] = [v for pair in present for v in pair]
        for r2 in wh.query(
            f"SELECT app, source_type, key, source_locator FROM artifacts WHERE {where}", params
        ):
            seq = seq_from_locator(r2["source_locator"])
            if seq is None:
                continue
            k = (r2["app"], r2["source_type"], r2["key"])
            if seq > best_seq.get(k, -1):
                best_seq[k] = seq

    for r in rows:
        st = r["source_type"]
        if st == "cache":
            url = _derive_cache_url(r["source_locator"])
            r["url"] = url
            r["cache_offset"] = cache_offset_from_locator(r["source_locator"])
            r["filename"] = url_filename_hint(url)
            r["latest"] = None
        else:
            r["url"] = r["cache_offset"] = r["filename"] = None
            if st in seq_types:
                seq = seq_from_locator(r["source_locator"])
                k = (r["app"], st, r["key"])
                r["latest"] = seq is None or seq >= best_seq.get(k, seq)
            else:
                r["latest"] = None
        del r["source_locator"]


@app.get("/api/cases/{case_id}/correlations")
def case_correlations(case_id: str, min_distinct_sources: int = 2) -> dict:
    """Interactive counterpart to the static report's §5 Cross-Source
    Correlations section — same underlying build_correlation_report (via the
    agent tool wrapper, so this endpoint and the agent see identical results,
    not two independent implementations), but live and queryable instead of
    generated once at report time. Phase 5 of the agent-improvement plan."""
    deps = InvestigationDeps(cases_root=Path(_settings.cases_root), case_id=case_id)
    if not deps.layout.warehouse.is_file():
        raise HTTPException(409, "case not processed yet")
    return correlate_entities(deps, min_distinct_sources)


@app.get("/api/cases/{case_id}/timeline")
def case_timeline(
    case_id: str,
    category: str | None = None,
    app: str | None = None,
    limit: int = 200,
    offset: int = 0,
) -> dict:
    """Interactive counterpart to the static report's §7 Evidence Timeline,
    which is generated once, hard-capped at 120 rows, and always excludes
    GENERAL — this is paginated like every other browse view instead, with
    category/app left as optional filters the caller chooses (not forced),
    matching /artifacts' own convention. Phase 5 of the agent-improvement
    plan; the memory-derived CreateTime/ExitTime events this makes visible
    at all were added by the immediately preceding phase (record_ts on
    Volatility's own timestamps, dd_lab/memory/volatility.py)."""
    deps = InvestigationDeps(cases_root=Path(_settings.cases_root), case_id=case_id)
    if not deps.layout.warehouse.is_file():
        raise HTTPException(409, "case not processed yet")
    where, params = ["record_ts IS NOT NULL"], []
    for col, val in (("category", category), ("app", app)):
        if val:
            where.append(f"{col} = ?")
            params.append(val)
    clause = "WHERE " + " AND ".join(where)
    from dd_lab.etl.warehouse import Warehouse

    wh = Warehouse(deps.layout.warehouse)
    try:
        total = wh.query(f"SELECT count(*) AS n FROM artifacts {clause}", list(params))[0]["n"]
        rows = wh.query(
            f"SELECT record_ts, app, source_type, category, key, substr(value,1,300) AS value, "
            f"artifact_id, source_file, source_locator FROM artifacts {clause} "
            f"ORDER BY record_ts LIMIT {int(limit)} OFFSET {int(offset)}",
            list(params),
        )
        # unfiltered by the CURRENT category/app choice (same convention as
        # /artifacts' own facets) so switching the category dropdown doesn't
        # shrink its own option list down to just what's already selected -
        # still scoped to record_ts IS NOT NULL, a category with no
        # timestamped rows at all isn't a useful choice on this view.
        facets = {
            "categories": [r["category"] for r in wh.query(
                "SELECT DISTINCT category FROM artifacts WHERE record_ts IS NOT NULL ORDER BY category")],
            "apps": [r["app"] for r in wh.query(
                "SELECT DISTINCT app FROM artifacts WHERE record_ts IS NOT NULL ORDER BY app")],
        }
    finally:
        wh.close()
    return {"rows": rows, "total": total, "shown": len(rows), "facets": facets}


@app.get("/api/cases/{case_id}/classification")
def case_classification(case_id: str) -> dict:
    """Per-category classification robustness: not just how many artifacts
    landed in a category, but which of the ensemble's voters (regex/keyword/
    NER/model) actually agreed on it — a category where only one voter ever
    fires is a much weaker signal than one several voters agree on."""
    deps = InvestigationDeps(cases_root=Path(_settings.cases_root), case_id=case_id)
    if not deps.layout.warehouse.is_file():
        raise HTTPException(409, "case not processed yet")
    from dd_lab.etl.warehouse import Warehouse

    wh = Warehouse(deps.layout.warehouse)
    try:
        rows = wh.query(
            "SELECT category, count(*) AS n, round(avg(confidence),3) AS mean_conf, "
            "sum(CASE WHEN regex_vote = category THEN 1 ELSE 0 END) AS regex_agree, "
            "sum(CASE WHEN keyword_vote = category THEN 1 ELSE 0 END) AS keyword_agree, "
            "sum(CASE WHEN ner_vote = category THEN 1 ELSE 0 END) AS ner_agree, "
            "sum(CASE WHEN model_vote = category THEN 1 ELSE 0 END) AS model_agree "
            "FROM artifacts GROUP BY category ORDER BY n DESC"
        )
        examples = {}
        for r in rows:
            examples[r["category"]] = wh.query(
                "SELECT app, source_type, key, substr(value,1,200) AS value, "
                "regex_vote, keyword_vote, ner_vote, model_vote "
                "FROM artifacts WHERE category = ? ORDER BY confidence DESC NULLS LAST LIMIT 5",
                [r["category"]],
            )
    finally:
        wh.close()
    return {"by_category": rows, "examples": examples}


@app.get("/api/cases/{case_id}/cache-files")
def case_cache_files(
    case_id: str,
    app_name: str | None = Query(None, alias="app"),
    q: str | None = None,
    limit: int = 200,
    offset: int = 0,
) -> dict:
    """A ChromeCacheView-style listing: one row per cache entry (HTTP disk
    cache and the Cache API alike), naming the exact physical file behind it.
    Reads straight off the evidence directory — unlike /artifacts this needs
    no completed analysis run, so the cache can be browsed immediately after
    a case is created."""
    layout = _layout(case_id)
    if not layout.manifest.is_file():
        raise HTTPException(404, "case not found")
    from dd_lab.parsers import discover_apps
    from dd_lab.parsers.base import ParseContext
    from dd_lab.parsers.cache import list_cache_files

    all_apps = discover_apps(layout.evidence)
    target_apps = [app_name] if app_name else all_apps
    rows: list[dict] = []
    for a in target_apps:
        ctx = ParseContext(case_id=case_id, app=a, evidence_root=layout.evidence)
        if not ctx.app_root.is_dir():
            continue
        rows.extend(e.model_dump(mode="json") for e in list_cache_files(ctx))
    if q:
        ql = q.lower()
        rows = [r for r in rows if ql in (r["url"] or "").lower() or ql in r["key"].lower()]
    rows.sort(key=lambda r: r["response_time"] or r["request_time"] or "", reverse=True)
    total = len(rows)
    page = rows[offset : offset + limit]
    return {"rows": page, "total": total, "shown": len(page), "apps": all_apps}


@app.get("/api/cases/{case_id}/cache-files/open")
def case_cache_file_open(
    case_id: str,
    app_name: str = Query(..., alias="app"),
    dir_: str = Query(..., alias="dir"),
    key: str = Query(...),
) -> Response:
    """The decoded body of one cache entry, for the UI's "Open" action.
    `dir` is validated against this app's actual cache directories before
    anything is opened — it never resolves an arbitrary path."""
    layout = _layout(case_id)
    if not layout.manifest.is_file():
        raise HTTPException(404, "case not found")
    from dd_lab.parsers.base import ParseContext
    from dd_lab.parsers.cache import cache_dirs, read_cache_entry

    ctx = ParseContext(case_id=case_id, app=app_name, evidence_root=layout.evidence)
    if dir_ not in cache_dirs(ctx):
        raise HTTPException(404, "cache directory not found for this app")
    body, content_type, filename_hint = read_cache_entry(ctx, dir_, key)
    if body is None:
        raise HTTPException(404, "this cache entry has no readable body")
    # inline, not attachment: previews in the SPA's iframe viewer; a browser
    # tab opening this URL directly is still free to save it under this name
    return Response(
        content=body, media_type=content_type,
        headers={"Content-Disposition": f'inline; filename="{filename_hint}"'},
    )


@app.get("/api/cases/{case_id}/artifacts/export.csv")
def case_artifacts_export(
    case_id: str,
    category: str | None = None,
    app: str | None = None,
    source_type: str | None = None,
    q: str | None = None,
) -> Response:
    """A full CSV export of the currently filtered artifact view - unlike
    /artifacts (capped for on-screen paging), this exports every matching
    row up to a hard safety cap, with the untruncated value."""
    deps = InvestigationDeps(cases_root=Path(_settings.cases_root), case_id=case_id)
    if not deps.layout.warehouse.is_file():
        raise HTTPException(409, "case not processed yet")
    where, params = [], []
    for col, val in (("category", category), ("app", app), ("source_type", source_type)):
        if val:
            where.append(f"{col} = ?")
            params.append(val)
    if q:
        # source_locator is included so a process detail view can filter
        # memory_string rows down to one PID (locators are "pid<PID>:string#n"
        # for process-scoped carving - see dd_lab/memory/analyze.py) without
        # a dedicated endpoint; it's also just generally useful to search.
        where.append("(key ILIKE ? OR value ILIKE ? OR source_locator ILIKE ?)")
        params += [f"%{q}%", f"%{q}%", f"%{q}%"]
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    from dd_lab.etl.warehouse import Warehouse

    fields = ["app", "source_type", "category", "confidence", "key", "value",
              "record_ts", "source_file", "artifact_id", "deleted",
              "url", "cache_offset", "filename", "latest"]
    _derived = ("url", "cache_offset", "filename", "latest")
    query_cols = [f for f in fields if f not in _derived] + ["source_locator"]
    wh = Warehouse(deps.layout.warehouse)
    try:
        rows = wh.query(
            f"SELECT {', '.join(query_cols)} FROM artifacts {clause} "
            f"ORDER BY (category != 'GENERAL') DESC, confidence DESC NULLS LAST, key, artifact_id "
            f"LIMIT 200000",
            list(params),
        )
        _enrich_artifact_rows(wh, rows)
    finally:
        wh.close()
    import csv
    import io

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return Response(
        content=buf.getvalue(), media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{case_id}_artifacts.csv"'},
    )


# Registered AFTER /artifacts/export.csv, not before: a wildcard
# /artifacts/{artifact_id} route registered first would swallow requests to
# /artifacts/export.csv too (Starlette matches path segments in registration
# order, and "export.csv" matches {artifact_id} just fine) - the same class
# of route-ordering mistake as a misplaced decorator, just at the path level
# instead of the function level.
@app.get("/api/cases/{case_id}/artifacts/{artifact_id}")
def case_artifact_detail(case_id: str, artifact_id: str) -> dict:
    """One artifact's full row, value UNTRUNCATED (unlike /artifacts, which
    caps it for on-screen paging) - backs the table's row-detail view."""
    deps = InvestigationDeps(cases_root=Path(_settings.cases_root), case_id=case_id)
    if not deps.layout.warehouse.is_file():
        raise HTTPException(409, "case not processed yet")
    from dd_lab.etl.warehouse import Warehouse

    wh = Warehouse(deps.layout.warehouse)
    try:
        rows = wh.query(
            "SELECT app, source_type, category, confidence, key, value, record_ts, source_file, "
            "source_locator, artifact_id, deleted FROM artifacts WHERE artifact_id = ?",
            [artifact_id],
        )
        if not rows:
            raise HTTPException(404, "artifact not found")
        _enrich_artifact_rows(wh, rows)
    finally:
        wh.close()
    return rows[0]


@app.get("/api/cases/{case_id}/localstorage-files")
def case_localstorage_files(case_id: str, app_name: str = Query(..., alias="app")) -> dict:
    """Every physical .ldb/.log file in this app's Local Storage store, each
    with a record count read directly from that one file - the same
    ChromeCacheView-style traceability the Cache files panel gives, adapted
    to how LevelDB actually stores things (a key isn't pinned to one file;
    compaction moves it between .ldb files over time, so this lists files,
    not keys)."""
    layout = _layout(case_id)
    if not layout.manifest.is_file():
        raise HTTPException(404, "case not found")
    from dd_lab.parsers.base import LOCAL_STORAGE, ParseContext
    from dd_lab.parsers.leveldb_browse import list_leveldb_files

    ctx = ParseContext(case_id=case_id, app=app_name, evidence_root=layout.evidence)
    files = list_leveldb_files(ctx.path(LOCAL_STORAGE))
    return {"files": [f.model_dump(mode="json") for f in files]}


@app.get("/api/cases/{case_id}/localstorage-files/open")
def case_localstorage_file_open(
    case_id: str,
    app_name: str = Query(..., alias="app"),
    file: str = Query(...),
    limit: int = 200,
    offset: int = 0,
) -> dict:
    """One physical Local Storage file's raw records, read directly from
    that file - not the merged, cross-file logical view the normal parser
    produces. `file` is checked against this app's real store files first."""
    layout = _layout(case_id)
    if not layout.manifest.is_file():
        raise HTTPException(404, "case not found")
    from dd_lab.parsers.base import LOCAL_STORAGE, ParseContext
    from dd_lab.parsers.leveldb_browse import list_leveldb_files, read_leveldb_file

    ctx = ParseContext(case_id=case_id, app=app_name, evidence_root=layout.evidence)
    store_dir = ctx.path(LOCAL_STORAGE)
    if file not in {f.filename for f in list_leveldb_files(store_dir)}:
        raise HTTPException(404, "file not found in this app's Local Storage store")
    records, total, truncated = read_leveldb_file(store_dir, file, limit=limit, offset=offset)
    return {
        "records": [r.model_dump(mode="json") for r in records],
        "total": total, "shown": len(records), "truncated": truncated,
    }


@app.get("/api/cases/{case_id}/db-files")
def case_db_files(case_id: str, app_name: str = Query(..., alias="app")) -> dict:
    """Every SQLite-backed evidence file for this app (currently: Cookies),
    each with its table list and row counts - raw database browsing,
    independent of the named parser that reads Cookies for a specific
    purpose."""
    layout = _layout(case_id)
    if not layout.manifest.is_file():
        raise HTTPException(404, "case not found")
    from dd_lab.parsers.base import COOKIES, COOKIES_LEGACY, ParseContext
    from dd_lab.parsers.sqlite_browse import list_sqlite_tables

    ctx = ParseContext(case_id=case_id, app=app_name, evidence_root=layout.evidence)
    files = []
    for rel in (COOKIES, COOKIES_LEGACY):
        entry = list_sqlite_tables(ctx.path(rel))
        if entry:
            files.append({**entry.model_dump(mode="json"), "path": rel})
    return {"files": files}


@app.get("/api/cases/{case_id}/db-files/open")
def case_db_file_open(
    case_id: str,
    app_name: str = Query(..., alias="app"),
    path: str = Query(...),
    table: str = Query(...),
    limit: int = 200,
    offset: int = 0,
) -> dict:
    """One table's raw rows from a SQLite-backed evidence file. `path` is
    checked against the known database-file locations (never an arbitrary
    path); `table` is checked against the database's own table list inside
    sqlite_browse.read_sqlite_table before being used in SQL."""
    layout = _layout(case_id)
    if not layout.manifest.is_file():
        raise HTTPException(404, "case not found")
    from dd_lab.parsers.base import COOKIES, COOKIES_LEGACY, ParseContext
    from dd_lab.parsers.sqlite_browse import read_sqlite_table

    if path not in (COOKIES, COOKIES_LEGACY):
        raise HTTPException(404, "not a recognised database file")
    ctx = ParseContext(case_id=case_id, app=app_name, evidence_root=layout.evidence)
    db_path = ctx.path(path)
    if not db_path.is_file():
        raise HTTPException(404, "database file not found for this app")
    columns, rows, total = read_sqlite_table(db_path, table, limit=limit, offset=offset)
    return {"columns": columns, "rows": rows, "total": total, "shown": len(rows)}


@app.post("/api/cases/{case_id}/query")
def case_query(case_id: str, body: QueryIn) -> dict:
    deps = InvestigationDeps(cases_root=Path(_settings.cases_root), case_id=case_id)
    if not deps.layout.warehouse.is_file():
        raise HTTPException(409, "case not processed yet")
    return query_warehouse(deps, body.sql)


@app.get("/api/cases/{case_id}/keyword/{keyword}")
def case_keyword(case_id: str, keyword: str) -> dict:
    deps = InvestigationDeps(cases_root=Path(_settings.cases_root), case_id=case_id)
    if not deps.layout.warehouse.is_file():
        raise HTTPException(409, "case not processed yet")
    return search_keyword(deps, keyword)


@app.get("/api/cases/{case_id}/report", response_class=HTMLResponse)
def case_report(case_id: str, name: str | None = None) -> HTMLResponse:
    layout = _layout(case_id)
    reports = sorted(layout.reports.glob("*.html")) if layout.reports.is_dir() else []
    if not reports:
        raise HTTPException(404, "no report generated yet")
    target = next((r for r in reports if r.name == name), reports[-1])
    return HTMLResponse(target.read_text("utf-8"))


@app.get("/api/cases/{case_id}/audit")
def case_audit(case_id: str) -> dict:
    layout = _layout(case_id)
    return {
        "custody": [r.model_dump(mode="json") for r in CustodyLog(layout.dir).entries()],
        "audit": [r.model_dump(mode="json") for r in AuditLog(layout.audit_log).records()],
    }


# --------------------------------------------------------------------- jobs


@app.get("/api/jobs")
def list_active_jobs() -> dict:
    """Every job still queued/running, across all cases — backs the
    persistent activity indicator so a long extraction or analysis stays
    visible (and known to be genuinely still running) no matter which page
    is open. Background jobs run on a server-side worker thread; navigating
    the UI elsewhere never stops one."""
    active = [j for j in _jobs.list(limit=200) if j["status"] in ("queued", "running")]
    return {"jobs": active}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job


# --------------------------------------------------------------------- static UI

if UI_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    idx = UI_DIR / "index.html"
    if idx.is_file():
        return HTMLResponse(idx.read_text("utf-8"))
    return HTMLResponse("<h1>Digital Detective</h1><p>UI assets not found.</p>")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "cases_root": _settings.cases_root}


def run(host: str = "127.0.0.1", port: int = 8765, *, open_window: bool = True) -> None:
    import uvicorn

    if open_window:
        try:
            from dd_lab.api.desktop import launch

            launch(app, host, port)
            return
        except Exception:  # noqa: BLE001
            log.warning("pywebview unavailable; serving in browser at http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run()
