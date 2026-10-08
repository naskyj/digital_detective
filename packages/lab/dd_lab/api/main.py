"""Digital Detective local application — FastAPI backend.

One process that does collection, ingest, agent orchestration, querying and
reporting, with a small static single-page UI. Runs on localhost only.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dd_core.casedir import CaseLayout
from dd_core.custody import CustodyLog
from dd_core.enums import CustodyEvent
from dd_core.manifest import Manifest
from dd_lab.agent.audit import AuditLog
from dd_lab.agent.runner import run_agent, run_pipeline
from dd_lab.agent.tools import (
    InvestigationDeps,
    correlate_entities,
    query_warehouse,
    search_keyword,
    write_finding,
)
from dd_lab.api.jobs import JobManager
from dd_lab.api.settings import AppSettings, describe_llm
from dd_lab.case import CaseMeta, CaseStatus, EvidenceSourceMeta, create_case, list_cases, parse_keywords
from dd_lab.classify.model import model_status
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
    relevance_model_dir: str | None = None
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


class ChatIn(BaseModel):
    message: str


class AddFindingIn(BaseModel):
    claim: str
    citations: list[dict]
    entity_type: str | None = None
    entity_value: str | None = None


class ReviewIn(BaseModel):
    artifact_id: str
    decision: str  # "approved" | "rejected" - validated in Warehouse.record_review
    reviewed_by: str = ""
    note: str = ""


# --------------------------------------------------------------------- settings


@app.get("/api/settings")
def get_settings() -> dict:
    s = _settings
    return {
        "cases_root": s.cases_root,
        "use_agent": s.use_agent,
        "classifier_model_dir": s.classifier_model_dir,
        "relevance_model_dir": s.relevance_model_dir,
        # Whether that directory is actually USABLE, not merely set. A
        # configured-but-unloadable model is otherwise invisible: every
        # artifact still gets classified, by rules alone, with no error
        # anywhere. See classify/model.py::model_status.
        "relevance_model": model_status(s.relevance_model_dir),
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
    old_root = _settings.cases_root
    for k, v in data.items():
        setattr(_settings, k, v)
    _settings.save()
    # ONLY rebuild the job manager when the cases root actually moved (the
    # jobs DB lives under it, so it's a genuinely different database then).
    #
    # Rebuilding it unconditionally was a real, destructive bug, hit for
    # real on a live case: JobManager.__init__ marks every 'queued'/
    # 'running' job as errored, because on a genuine app restart those
    # worker threads are gone. But a settings save does NOT restart the
    # app - the threads are still running. So saving settings mid-run
    # falsely flagged an in-flight job as "interrupted", which ALSO
    # cleared the `_jobs.active_for(case_id)` guard that stops a second
    # job starting on the same case. On the real 675k-artifact case that
    # let a full "Run analysis" start on top of a running re-classify,
    # and its reset_app() step deleted an app's artifacts mid-flight.
    if _settings.cases_root != old_root:
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


@app.post("/api/cases/{case_id}/reclassify")
def reclassify_case(
    case_id: str,
    reuse_model_scores: bool = False,
    only_unscored: bool = False,
    exclude_source_types: str | None = None,
    source_types: str | None = None,
) -> dict:
    """Re-run classification against artifacts already in the warehouse -
    cheap and in-place (a single set-based UPDATE per batch via
    classify_warehouse), unlike "Run analysis" above which re-parses from
    raw evidence. classify_warehouse itself has existed, fully tested,
    since the classifier_backend_spec rewrite, but had no caller anywhere
    in the app until now - the only way to get artifacts re-classified was
    a full re-run. Real use case this exists for: relevance_model_dir (or
    classifier_model_dir) just got configured/changed, or the ensemble's
    own logic changed, and existing artifacts need to reflect that without
    burning a full re-ingest.

    `reuse_model_scores=true` additionally skips re-running the relevance
    model, re-deriving relevance from each row's already-stored score
    instead (see classify_warehouse). On the real 675k-artifact case that
    is the difference between ~30 minutes and ~27 hours - use it whenever
    only the combination logic changed, not the model itself.

    `only_unscored=true` runs the model over ONLY the rows that don't have
    a score yet - for finishing an interrupted model run without redoing
    its completed work. Typical sequence after an interrupted run:
    `?only_unscored=true` (fills the gap), then `?reuse_model_scores=true`
    (re-derives relevance uniformly across the whole case, ~30 min).

    `exclude_source_types` is a comma-separated list of source types to
    skip - for scoping an expensive model pass away from a source type too
    large to be worth CPU inference (e.g. `memory_string`, ~547k of 675k
    artifacts on the real case). Skipped rows stay fully classified, just
    with rules-only relevance."""
    layout = _layout(case_id)
    if not layout.manifest.is_file():
        raise HTTPException(404, "case not found")
    if not layout.warehouse.is_file():
        raise HTTPException(409, "case not processed yet - run analysis first")
    if _jobs.active_for(case_id):
        raise HTTPException(409, "a job is already running for this case")

    excluded = tuple(s.strip() for s in exclude_source_types.split(",") if s.strip()) \
        if exclude_source_types else None
    included = tuple(s.strip() for s in source_types.split(",") if s.strip()) \
        if source_types else None

    def work(progress) -> dict:
        from dd_lab.classify import classify_warehouse
        from dd_lab.etl.warehouse import Warehouse

        progress(
            "re-classifying "
            + ("only artifacts with no stored model score" if only_unscored else "artifacts")
            + " already in the warehouse"
            + (f", skipping source types: {', '.join(excluded)}" if excluded else "")
            + (f", ONLY source types: {', '.join(included)}" if included else "")
            + (" (reusing stored relevance-model scores)" if reuse_model_scores else "")
            + "..."
        )
        clf = _settings.build_classifier()
        wh = Warehouse(layout.warehouse)
        try:
            result = classify_warehouse(
                wh, clf, reuse_model_scores=reuse_model_scores, only_unscored=only_unscored,
                exclude_source_types=excluded, source_types=included,
            )
        finally:
            wh.close()
        progress(f"re-classified {result.classified} artifacts")
        return {"ok": True, "classified": result.classified, "by_category": result.by_category}

    job_id = _jobs.submit("reclassify", work, case_id=case_id)
    return {"job_id": job_id}


@app.post("/api/cases/{case_id}/findings/supersede-stale")
def case_supersede_stale_findings(case_id: str, dry_run: bool = True) -> dict:
    """Mark findings whose entity the current extractor no longer recognises.

    Defaults to dry_run: this changes stored case records, so the caller sees
    exactly what would be marked before anything is. Nothing is ever deleted -
    a superseded finding keeps its text, citations and timestamps and simply
    stops being presented as live (see FindingsStore.supersede_stale)."""
    layout = _layout(case_id)
    if not layout.manifest.is_file():
        raise HTTPException(404, "case not found")
    from dd_lab.agent.findings import FindingsStore

    store = FindingsStore(layout.findings_log)
    stale = store.supersede_stale(dry_run=dry_run)
    return {
        "ok": True,
        "dry_run": dry_run,
        "superseded": len(stale),
        # In a dry run nothing has been marked, so counting current statuses
        # would report the pre-run number and read as "would supersede 202,
        # leaving 207". Subtract instead, so the figure means the same thing
        # in both modes: what will be live afterwards.
        "remaining": len([f for f in store.list() if str(f.status) != "superseded"])
        - (len(stale) if dry_run else 0),
        "examples": [
            {"entity_type": f.entity_type, "entity_value": f.entity_value,
             "created_ts": str(f.created_ts), "created_by": f.created_by}
            for f in stale[:20]
        ],
    }


@app.post("/api/cases/{case_id}/regenerate-report")
def regenerate_report(case_id: str) -> dict:
    """Rebuild the report from whatever's currently in the warehouse and
    logs/findings.json - no re-parse, re-classify, or re-investigate,
    unlike "Run analysis"/"Re-run analysis" above. Same real gap this
    session already found and fixed once for classification
    (classify_warehouse existed, fully tested, with zero production call
    sites until /reclassify was added): build_report/generate_report exist
    and work, but the ONLY way to get a new report was the full /run job -
    a finding saved via POST /findings (the "Save as finding" chat action)
    had nowhere to actually reach the report without it. build_report
    reads findings.json fresh every call and seal_report's the new file
    with its own timestamp+hash - no in-place edit, no conflict with a
    previous report's own seal."""
    layout = _layout(case_id)
    if not layout.manifest.is_file():
        raise HTTPException(404, "case not found")
    if not layout.warehouse.is_file():
        raise HTTPException(409, "case not processed yet - run analysis first")
    if _jobs.active_for(case_id):
        raise HTTPException(409, "a job is already running for this case")

    def work(progress) -> dict:
        from dd_lab.report import build_report

        progress("regenerating report...")
        # The model-written summary is attempted only when a model is
        # actually reachable; with none the report is complete without it
        # (see report/builder.py). Gated on the RESOLVED config, not on
        # settings.llm_api_key - the key legitimately comes from the
        # environment instead, in which case that field is empty and the
        # summary was silently skipped even with a working model.
        _cfg = _settings.agent_config()
        _usable = bool(_cfg.api_key) or _cfg.provider == "ollama"  # ollama is local, needs no key
        path = build_report(
            _settings.cases_root, case_id, actor="investigator",
            llm_cfg=_cfg if _usable else None,
        )
        progress(f"report written to {path}")
        return {"ok": True, "report_path": str(path)}

    job_id = _jobs.submit("regenerate-report", work, case_id=case_id)
    return {"job_id": job_id}


@app.get("/api/cases/{case_id}/relevance")
def case_relevance(case_id: str) -> dict:
    """Relevance breakdown - mirrors case_classification's shape (by-
    category vote-agreement) but for the now-primary relevant/
    relevance_score output: how many artifacts landed on each side, and
    the trained relevance model's own mean contribution where configured."""
    deps = InvestigationDeps(cases_root=Path(_settings.cases_root), case_id=case_id)
    if not deps.layout.warehouse.is_file():
        raise HTTPException(409, "case not processed yet")
    from dd_lab.etl.warehouse import Warehouse

    wh = Warehouse(deps.layout.warehouse)
    try:
        rows = wh.query("SELECT * FROM v_by_relevance")
        examples = {}
        for r in rows:
            examples[str(r["relevant"])] = wh.query(
                "SELECT app, source_type, category, key, substr(value,1,200) AS value, "
                "relevance_score, relevance_model_score, regex_vote, keyword_vote, ner_vote "
                "FROM artifacts WHERE relevant = ? ORDER BY relevance_score DESC NULLS LAST, "
                "relevance_model_score DESC NULLS LAST LIMIT 5",
                [r["relevant"]],
            )
    finally:
        wh.close()
    return {"by_relevance": rows, "examples": examples}


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
    relevant: bool | None = None,
    sort: str | None = None,
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
    if relevant is not None:
        where.append("relevant = ?")
        params.append(relevant)
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
            f"SELECT app, source_type, category, confidence, relevant, relevance_score, key, "
            f"substr(value,1,300) AS value, "
            f"record_ts, source_file, source_locator, artifact_id, regex_vote, keyword_vote, "
            f"ner_vote, model_vote, entropy_vote, relevance_model_score, evidence_spans, deleted "
            f"FROM artifacts {clause} "
            f"ORDER BY {_artifact_order_by(sort)} "
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


# Shared between /artifacts and /artifacts/export.csv, same reason
# _enrich_artifact_rows is shared - one place, so they can't drift apart on
# what a `sort` value means. A whitelist dict, not raw string interpolation
# of the `sort` query param - ORDER BY can't be parametrized with `?` the
# way WHERE values can, so an unvalidated column name here would be a real
# SQL-injection surface.
#
# Both relevance orderings tie-break on relevance_model_score. The composed
# score saturates (see the v_high_priority view in etl/warehouse.py): 2,559
# artifacts in the annotated corpus share the score 1.0, of which only 48% are
# relevant, so without a tie-break the examiner reads a 2,559-row block in
# arbitrary order. The model probability orders that block almost perfectly
# and costs nothing - it changes no relevance decision, only the order within
# a tie.
_ARTIFACT_SORTS = {
    "": "relevant DESC, relevance_score DESC NULLS LAST, "
        "relevance_model_score DESC NULLS LAST, confidence DESC NULLS LAST, "
        "key, artifact_id",
    "relevance": "relevance_score DESC NULLS LAST, "
                 "relevance_model_score DESC NULLS LAST, key, artifact_id",
    "confidence": "confidence DESC NULLS LAST, key, artifact_id",
    "time": "record_ts DESC NULLS LAST, key, artifact_id",
}


def _artifact_order_by(sort: str | None) -> str:
    return _ARTIFACT_SORTS.get(sort or "", _ARTIFACT_SORTS[""])


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


#: One entry: the last correlation report computed. Deliberately not a
#: growing cache - this is a single-investigator desktop app working one
#: case at a time, and the thing being cached is tens of MB of citations.
_CORRELATION_CACHE: dict = {}


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

    # Cached on the warehouse's own identity. build_correlation_report runs
    # one un-indexed ILIKE full scan per candidate entity, which measured 58
    # SECONDS on the real 675k-artifact case - paid again on every single
    # page load, because nothing was memoised. The result is a pure function
    # of the warehouse contents, so keying on (size, mtime) is exact: any
    # re-classify or re-ingest rewrites the file and invalidates this by
    # construction, and a stale answer is not reachable.
    st = deps.layout.warehouse.stat()
    key = (case_id, min_distinct_sources, st.st_size, st.st_mtime_ns)
    if _CORRELATION_CACHE.get("key") == key:
        return _CORRELATION_CACHE["value"]
    value = correlate_entities(deps, min_distinct_sources)
    _CORRELATION_CACHE.clear()
    _CORRELATION_CACHE.update(key=key, value=value)
    return value


@app.get("/api/cases/{case_id}/narrative")
def case_narrative(case_id: str, relevant_only: bool = True) -> dict:
    """What happened, reconstructed as episodes of continuous activity.

    Deterministic - same evidence, same account, on any machine, with no
    model and no network. The flat `/timeline` endpoint below is still the
    row-level view; this is the one that answers the question a reader
    actually has."""
    layout = _layout(case_id)
    if not layout.manifest.is_file():
        raise HTTPException(404, "case not found")
    if not layout.warehouse.is_file():
        raise HTTPException(409, "case not processed yet - run analysis first")
    from dd_lab.etl.warehouse import Warehouse
    from dd_lab.investigate import build_case_narrative

    wh = Warehouse(layout.warehouse)
    try:
        return {"ok": True, **build_case_narrative(wh, relevant_only=relevant_only)}
    finally:
        wh.close()


@app.get("/api/cases/{case_id}/timeline")
def case_timeline(
    case_id: str,
    category: str | None = None,
    app: str | None = None,
    relevant: bool | None = None,
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
    if relevant is not None:
        where.append("relevant = ?")
        params.append(relevant)
    clause = "WHERE " + " AND ".join(where)
    from dd_lab.etl.warehouse import Warehouse

    wh = Warehouse(deps.layout.warehouse)
    try:
        total = wh.query(f"SELECT count(*) AS n FROM artifacts {clause}", list(params))[0]["n"]
        rows = wh.query(
            f"SELECT record_ts, app, source_type, category, relevant, relevance_score, key, "
            f"substr(value,1,300) AS value, "
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
            "sum(CASE WHEN model_vote = category THEN 1 ELSE 0 END) AS model_agree, "
            "sum(CASE WHEN entropy_vote = category THEN 1 ELSE 0 END) AS entropy_agree "
            "FROM artifacts GROUP BY category ORDER BY n DESC"
        )
        examples = {}
        for r in rows:
            examples[r["category"]] = wh.query(
                "SELECT app, source_type, key, substr(value,1,200) AS value, "
                "regex_vote, keyword_vote, ner_vote, model_vote, entropy_vote "
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
    relevant: bool | None = None,
    sort: str | None = None,
    q: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> StreamingResponse:
    """A complete CSV export of the currently filtered artifact view, with
    untruncated values and the full classification result for every row.

    Streamed, and uncapped by default: pass `limit`/`offset` only to split a
    very large export into parts deliberately. An export that silently stops
    part-way is worse than one that fails, because the file looks complete."""
    deps = InvestigationDeps(cases_root=Path(_settings.cases_root), case_id=case_id)
    if not deps.layout.warehouse.is_file():
        raise HTTPException(409, "case not processed yet")
    where, params = [], []
    for col, val in (("category", category), ("app", app), ("source_type", source_type)):
        if val:
            where.append(f"{col} = ?")
            params.append(val)
    if relevant is not None:
        where.append("relevant = ?")
        params.append(relevant)
    if q:
        # source_locator is included so a process detail view can filter
        # memory_string rows down to one PID (locators are "pid<PID>:string#n"
        # for process-scoped carving - see dd_lab/memory/analyze.py) without
        # a dedicated endpoint; it's also just generally useful to search.
        where.append("(key ILIKE ? OR value ILIKE ? OR source_locator ILIKE ?)")
        params += [f"%{q}%", f"%{q}%", f"%{q}%"]
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    from dd_lab.etl.warehouse import Warehouse

    # The full classification result, not just the category. `relevant` and
    # `relevance_score` are the pipeline's PRIMARY output - reporting,
    # correlation and timeline construction all key off them - and were
    # absent from this export entirely, so a CSV could not be used to check
    # or re-score the classification it was exporting. The per-voter columns
    # make a decision auditable offline: they are what distinguishes an
    # artifact all five voters agreed on from one carried by a single regex.
    fields = ["artifact_id", "app", "source_type", "category", "confidence",
              "relevant", "relevance_score", "relevance_model_score",
              "regex_vote", "keyword_vote", "ner_vote", "model_vote", "entropy_vote",
              "evidence_spans", "classifier_version",
              "key", "value", "record_ts", "deleted",
              "source_file", "source_locator_raw",
              "url", "cache_offset", "filename", "latest"]
    _derived = ("url", "cache_offset", "filename", "latest", "source_locator_raw")
    query_cols = [f for f in fields if f not in _derived] + ["source_locator"]
    import csv
    import io

    # Streamed in batches rather than assembled in memory and capped.
    #
    # This previously ended `LIMIT 200000` and built the whole file as one
    # string. On a 675,234-artifact case that silently produced a file
    # containing exactly 200,000 rows with nothing in it saying so - a export
    # that looks complete and is not, which for evidence is worse than an
    # error. Streaming removes both problems: there is no cap, and the
    # response never holds more than one batch.
    #
    # `limit`/`offset` remain available for deliberate splitting into parts;
    # they now default to "everything" instead of silently truncating.
    batch = 20_000
    order_by = _artifact_order_by(sort)

    def _rows():
        wh = Warehouse(deps.layout.warehouse)
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        yield buf.getvalue()
        buf.seek(0), buf.truncate(0)
        try:
            taken = 0
            while True:
                want = batch if limit is None else min(batch, limit - taken)
                if want <= 0:
                    break
                page = wh.query(
                    f"SELECT {', '.join(query_cols)} FROM artifacts {clause} "  # noqa: S608
                    f"ORDER BY {order_by} LIMIT {int(want)} OFFSET {int(offset + taken)}",
                    list(params),
                )
                if not page:
                    break
                # The raw locator is kept as its own column before
                # _enrich_artifact_rows consumes and drops it: it is the
                # in-file offset a reader needs to reach the original bytes,
                # and for memory rows it carries the PID.
                for r in page:
                    r["source_locator_raw"] = r.get("source_locator")
                _enrich_artifact_rows(wh, page)
                writer.writerows(page)
                yield buf.getvalue()
                buf.seek(0), buf.truncate(0)
                taken += len(page)
                if len(page) < want:
                    break
        finally:
            wh.close()

    suffix = f"_{offset}" if offset else ""
    return StreamingResponse(
        _rows(), media_type="text/csv",
        headers={
            "Content-Disposition":
                f'attachment; filename="{case_id}_artifacts{suffix}.csv"',
        },
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
            "SELECT app, source_type, category, confidence, relevant, relevance_score, key, value, "
            "record_ts, source_file, source_locator, artifact_id, deleted, "
            "regex_vote, keyword_vote, ner_vote, model_vote, entropy_vote, relevance_model_score "
            "FROM artifacts WHERE artifact_id = ?",
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


@app.get("/api/cases/{case_id}/chat")
def case_chat_history(case_id: str) -> dict:
    """The persisted conversation so far, flattened for the chat panel to
    render on load. An unanalysed case (or one with no chat yet) simply has
    no turns - not an error, chat just hasn't started."""
    from dd_lab.agent.chat import history_for_ui, load_history

    layout = _layout(case_id)
    if not layout.manifest.is_file():
        raise HTTPException(404, "case not found")
    deps = InvestigationDeps(cases_root=Path(_settings.cases_root), case_id=case_id)
    return {"turns": history_for_ui(load_history(deps))}


@app.post("/api/cases/{case_id}/chat")
def case_chat_send(case_id: str, body: ChatIn) -> dict:
    """Phase 6 of the agent-improvement plan - a read-mostly, multi-turn
    chat over the same investigation tool substrate as the main agent
    (query_warehouse, search_keyword, correlate_entities, list_processes,
    process_memory, list_findings, ...), excluding anything mutating or
    expensive (run_ingest/run_memory_analysis/write_finding/generate_report -
    see dd_lab/agent/chat.py's own docstring). Synchronous request/response,
    not a background job - a chat reply is a normal few-second wait, not a
    multi-minute analysis run."""
    from dd_lab.agent.chat import run_chat_turn

    layout = _layout(case_id)
    if not layout.manifest.is_file():
        raise HTTPException(404, "case not found")
    deps = InvestigationDeps(cases_root=Path(_settings.cases_root), case_id=case_id)
    result = run_chat_turn(deps, _settings.agent_config(), body.message)
    if not result.ok:
        return {"ok": False, "error": result.error}
    return {"ok": True, "reply": result.reply}


@app.post("/api/cases/{case_id}/findings")
def case_add_finding(case_id: str, body: AddFindingIn) -> dict:
    """Investigator-triggered save - the "Save as finding" action on a chat
    reply (or anywhere else in the UI that wants to record a claim). Calls
    write_finding directly, the same citation-verification path the
    investigation agent's own write_finding tool call uses - it is NOT
    reachable from the chat LLM itself (chat.py's deliberately read-only
    toolset excludes write_finding entirely; see that module's docstring),
    this is the investigator's own explicit action instead. created_by is
    hardcoded to "investigator" here - the caller can never claim to be the
    agent, keeping every finding's real provenance visible in the report
    (§4 already renders created_by per finding, no report-builder change
    needed). Every citation is still independently re-verified against the
    warehouse; a wrong/hallucinated artifact_id fails safely
    (REJECTED_UNVERIFIED), never silently accepted."""
    layout = _layout(case_id)
    if not layout.manifest.is_file():
        raise HTTPException(404, "case not found")
    deps = InvestigationDeps(cases_root=Path(_settings.cases_root), case_id=case_id)
    return write_finding(
        deps, body.claim, body.citations, body.entity_type, body.entity_value,
        created_by="investigator",
    )


@app.get("/api/cases/{case_id}/review-queue")
def case_review_queue(case_id: str, limit: int = Query(50, ge=1, le=500),
                      offset: int = Query(0, ge=0),
                      include_reviewed: bool = False,
                      lane: str = "likely_misses") -> dict:
    """The artifacts the classifier did NOT flag, best candidates first.

    Measured on the annotated corpus, the classifier misses roughly one
    relevant artifact in 24 (recall 0.958). Those misses cannot be found with
    a lower threshold: the composed score is near-bimodal, and 45 of the 54
    sit at exactly 0.0 among 551,809 others — "everything below 0.5" is
    553,727 artifacts, which is the problem the tool exists to solve, restated.

    The model's own probability does rank them. Reviewing the top 50 of this
    queue recovers 21 of the 54 misses, with 42% of what the examiner reads
    being a genuine miss; the top 1,000 recovers 31. A fixed budget off the
    top of this ordering is therefore worth spending, and a threshold sweep
    is not. The budget is the examiner's to choose.

    `lane="suppressed_sensitive"` is the second, smaller queue: artifacts in a
    credential/payment/location category that the model veto suppressed. Those
    categories used to be exempt from the veto entirely, which put 903 false
    positives in the report to keep 18 real findings. They are now vetoed like
    anything else and land here instead, so the report stops asserting them
    while a human can still reach them. Ordered by category sensitivity, not
    model score — by construction every row in this lane scores below the veto
    threshold, so that score carries no information within it.
    """
    from dd_lab.etl.warehouse import Warehouse

    layout = _layout(case_id)
    if not layout.warehouse.is_file():
        raise HTTPException(404, "case not found")
    wh = Warehouse(layout.warehouse)
    try:
        try:
            rows = wh.review_queue(limit=limit, offset=offset,
                                   include_reviewed=include_reviewed, lane=lane)
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        _enrich_artifact_rows(wh, rows)
        return {"rows": rows, "limit": limit, "offset": offset, "lane": lane,
                "counts": wh.review_counts()}
    finally:
        wh.close()


@app.post("/api/cases/{case_id}/review")
def case_record_review(case_id: str, body: ReviewIn) -> dict:
    """Record an examiner's accept/reject on one artifact from the queue.

    An approval flips `relevant`, so the artifact flows into the report,
    correlation and timeline exactly like any other relevant artifact — which
    is the point: the examiner is correcting a miss, not annotating one.

    Three properties make that safe to do to evidence. The classifier's own
    decision is preserved in `relevant_auto` and never overwritten, so the
    override is always visible as an override and the classifier stays
    independently scoreable (R3). A later re-classification will not revoke
    the decision (see apply_classifications). And each decision appends to the
    hash-chained custody log, so who decided what, and when, is attributable
    and non-repudiable (R2).
    """
    from dd_lab.etl.warehouse import Warehouse

    layout = _layout(case_id)
    if not layout.warehouse.is_file():
        raise HTTPException(404, "case not found")
    wh = Warehouse(layout.warehouse)
    try:
        row = wh.record_review(body.artifact_id, body.decision, body.reviewed_by)
    except KeyError:
        raise HTTPException(404, f"artifact {body.artifact_id} not in this case") from None
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    finally:
        wh.close()

    CustodyLog(layout.dir).append(
        CustodyEvent.REVIEW_DECISION,
        actor=body.reviewed_by or "investigator",
        detail=(f"{body.decision} artifact {body.artifact_id} "
                f"({row.get('app')}/{row.get('source_type')}: {row.get('key')})"),
        data={
            "artifact_id": body.artifact_id,
            "decision": body.decision,
            # The automated decision this overrides, and the scores behind it,
            # so the log entry stands on its own without the warehouse.
            "relevant_auto": row.get("relevant_auto"),
            "relevance_score": row.get("relevance_score"),
            "relevance_model_score": row.get("relevance_model_score"),
            "source_file": row.get("source_file"),
            "source_locator": row.get("source_locator"),
            "note": body.note,
        },
    )
    return {"ok": True, "artifact": row}


@app.get("/api/cases/{case_id}/report", response_class=HTMLResponse)
def case_report(case_id: str, name: str | None = None, download: bool = False) -> HTMLResponse:
    """The generated report. `download=true` serves the same bytes as an
    attachment rather than inline.

    The file is self-contained - CSS inline, no external references - so the
    downloaded copy renders identically anywhere, and its SHA-256 still
    matches the seal recorded in the case's custody chain. That is the
    property that makes it exportable evidence rather than a screenshot: a
    recipient can verify the copy they received is the copy that was
    generated. Print-to-PDF is the browser's own, so no rendering dependency
    is introduced; the report carries a print stylesheet for it."""
    layout = _layout(case_id)
    reports = sorted(layout.reports.glob("*.html")) if layout.reports.is_dir() else []
    if not reports:
        raise HTTPException(404, "no report generated yet")
    target = next((r for r in reports if r.name == name), reports[-1])
    headers = (
        {"Content-Disposition": f'attachment; filename="{target.name}"'} if download else None
    )
    return HTMLResponse(target.read_text("utf-8"), headers=headers)


@app.get("/api/cases/{case_id}/reports")
def case_report_list(case_id: str) -> dict:
    """Every report generated for this case, newest first, each with the
    custody-seal verification result for that specific file - so a reader
    choosing which to export can see which ones still match what was
    sealed."""
    from dd_lab.report.builder import verify_report

    layout = _layout(case_id)
    if not layout.manifest.is_file():
        raise HTTPException(404, "case not found")
    out = []
    for r in sorted(layout.reports.glob("*.html"), reverse=True) if layout.reports.is_dir() else []:
        ok, detail = verify_report(layout, r)
        out.append({
            "name": r.name,
            "size_bytes": r.stat().st_size,
            "generated_ts": datetime.fromtimestamp(r.stat().st_mtime, UTC).isoformat(),
            "seal_ok": ok,
            "seal_detail": detail,
        })
    return {"ok": True, "reports": out}


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
