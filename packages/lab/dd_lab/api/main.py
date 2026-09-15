"""Digital Detective local application — FastAPI backend.

One process that does collection, ingest, agent orchestration, querying and
reporting, with a small static single-page UI. Runs on localhost only.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dd_core.casedir import CaseLayout
from dd_core.custody import CustodyLog
from dd_core.manifest import Manifest
from dd_lab.agent.audit import AuditLog
from dd_lab.agent.runner import run_agent, run_pipeline
from dd_lab.agent.tools import InvestigationDeps, query_warehouse, search_keyword
from dd_lab.api.jobs import JobManager
from dd_lab.api.settings import AppSettings, describe_llm
from dd_lab.case import CaseMeta, CaseStatus, create_case, list_cases, parse_keywords
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
    case_id: str
    name: str
    description: str = ""
    keywords: str = ""


class ImageListIn(BaseModel):
    image_path: str


class ImageExtractIn(BaseModel):
    image_path: str
    apps: list[str] | None = None  # None = every profile found
    memory_image_path: str | None = None
    operator: str = ""
    case_id: str | None = None


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


@app.post("/api/image/list")
def image_list(body: ImageListIn) -> dict:
    """Open a disk image and list the Electron/Chromium app profiles found on
    it — no extraction yet. Fails cleanly (400, not a crash) if the `image`
    extra isn't installed or the file isn't a recognisable image."""
    from dd_lab.imaging import ImageUnavailable, discover_apps, open_image

    try:
        target = open_image(body.image_path)
        apps = discover_apps(target)
    except ImageUnavailable as e:
        raise HTTPException(400, str(e)) from e
    return {
        "apps": [{"name": a.name, "profile_dir": str(a.profile_dir), "version": a.version} for a in apps],
    }


@app.post("/api/image/extract")
def image_extract(body: ImageExtractIn) -> dict:
    from dd_lab.imaging import acquire_from_image

    def work(progress) -> dict:
        # ImageUnavailable (and anything else) is caught by JobManager and
        # surfaced on the job record — no crash, just a failed job the UI shows.
        result = acquire_from_image(
            body.image_path, _settings.cases_root,
            apps=body.apps, memory_image_path=body.memory_image_path,
            case_id=body.case_id, operator=body.operator, progress=progress,
        )
        return {
            "case_id": result.case_id, "files": result.file_count,
            "bytes": result.total_bytes, "fingerprint": result.public_key_fingerprint,
            "image_sha256": result.image_sha256, "apps": result.apps, "warnings": result.warnings,
        }

    job_id = _jobs.submit("image_extract", work, case_id=body.case_id)
    return {"job_id": job_id}


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
    try:
        meta = create_case(
            _settings.cases_root, body.case_id, body.name, body.description,
            parse_keywords(body.keywords),
        )
    except FileNotFoundError as e:
        raise HTTPException(400, str(e)) from e
    return meta.model_dump(mode="json")


@app.get("/api/cases/{case_id}")
def get_case(case_id: str) -> dict:
    layout = _layout(case_id)
    if not layout.manifest.is_file():
        raise HTTPException(404, "case not found")
    meta = CaseMeta.load_or_none(layout) or CaseMeta(case_id=case_id, name=case_id)
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


@app.get("/api/cases/{case_id}/artifacts")
def case_artifacts(
    case_id: str,
    category: str | None = None,
    app: str | None = None,
    source_type: str | None = None,
    q: str | None = None,
    limit: int = 200,
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
        where.append("(key ILIKE ? OR value ILIKE ?)")
        params += [f"%{q}%", f"%{q}%"]
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    from dd_lab.etl.warehouse import Warehouse

    wh = Warehouse(deps.layout.warehouse)
    try:
        total = wh.query(f"SELECT count(*) AS n FROM artifacts {clause}", list(params))[0]["n"]
        rows = wh.query(
            f"SELECT app, source_type, category, confidence, key, substr(value,1,300) AS value, "
            f"record_ts, source_file, artifact_id, regex_vote, keyword_vote, ner_vote, model_vote, "
            f"evidence_spans FROM artifacts {clause} "
            f"ORDER BY (category != 'GENERAL') DESC, confidence DESC NULLS LAST, key "
            f"LIMIT {int(limit)}",
            list(params),
        )
        facets = {
            "apps": [r["app"] for r in wh.query(
                "SELECT DISTINCT app FROM artifacts ORDER BY app")],
            "source_types": [r["source_type"] for r in wh.query(
                "SELECT DISTINCT source_type FROM artifacts ORDER BY source_type")],
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
