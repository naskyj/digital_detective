"""Deterministic tools the agent (or a plain script) drives.

Every function takes `InvestigationDeps` and returns a small JSON-able result.
They are pure orchestration over the Phase 0-3 machinery; none of them ask an
LLM and none of them mutate evidence. The agent decides *when* to call them; the
supervisor records *that* they were called.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from dd_core.casedir import CaseLayout
from dd_core.manifest import Manifest, verify_case
from dd_lab.agent.audit import AuditLog
from dd_lab.agent.state import PipelineState
from dd_lab.case import CaseMeta
from dd_lab.classify import Classifier
from dd_lab.etl.warehouse import Warehouse
from dd_lab.ingest import IntegrityError, ingest_case
from dd_lab.parsers import discover_apps
from dd_lab.parsers.base import MEMORY
from dd_lab.report import build_report

_SELECT_ONLY = re.compile(r"^\s*(with|select)\b", re.I)
_FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|copy|pragma|call)\b", re.I)


@dataclass
class InvestigationDeps:
    cases_root: Path
    case_id: str
    layout: CaseLayout = field(init=False)
    meta: CaseMeta = field(init=False)
    audit: AuditLog = field(init=False)
    state: PipelineState = field(init=False)
    classifier: Classifier | None = None

    def __post_init__(self) -> None:
        self.cases_root = Path(self.cases_root)
        self.layout = CaseLayout(self.cases_root, self.case_id)
        self.meta = CaseMeta.load_or_none(self.layout) or CaseMeta(
            case_id=self.case_id, name=self.case_id
        )
        self.audit = AuditLog(self.layout.audit_log)
        self.state = PipelineState.load(
            self.layout.dir / "logs" / "pipeline_state.json", self.case_id
        )


# --------------------------------------------------------------------- tools


def case_status(deps: InvestigationDeps) -> dict:
    wh_exists = deps.layout.warehouse.is_file()
    count = 0
    if wh_exists:
        wh = Warehouse(deps.layout.warehouse)
        try:
            count = wh.count()
        finally:
            wh.close()
    return {
        "case_id": deps.case_id,
        "name": deps.meta.name,
        "keywords": deps.meta.keywords,
        "progress": deps.state.summary(),
        "warehouse_artifacts": count,
        "report": deps.state.report_path,
    }


def list_evidence(deps: InvestigationDeps) -> dict:
    manifest = Manifest.read(deps.layout.dir)
    apps = [a.name for a in manifest.target_apps] or discover_apps(deps.layout.evidence)
    per_app = {}
    for app in apps:
        app_dir = deps.layout.evidence / app
        sources = sorted(p.name for p in app_dir.iterdir()) if app_dir.is_dir() else []
        per_app[app] = {
            "sources": sources,
            "has_memory": (app_dir / MEMORY).is_dir(),
        }
    return {
        "apps": apps,
        "per_app": per_app,
        "manifest_files": manifest.entry_count,
        "acquisition_host": manifest.host.hostname,
    }


def verify_integrity(deps: InvestigationDeps) -> dict:
    result = verify_case(deps.layout.dir)
    deps.state.integrity_verified = result.ok
    deps.state.save()
    return {
        "ok": result.ok,
        "signature_valid": result.signature_valid,
        "files_checked": result.checked,
        "issues": [i.model_dump() for i in result.issues[:20]],
    }


def run_ingest(deps: InvestigationDeps, apps: list[str] | None = None) -> dict:
    try:
        result = ingest_case(
            deps.cases_root, deps.case_id,
            actor="agent", classifier=deps.classifier,
        )
    except IntegrityError as e:
        return {"ok": False, "error": f"integrity check failed: {e}"}
    deps.state.integrity_verified = True
    deps.state.ingested_apps = result.apps
    deps.state.classified = True
    deps.state.save()
    return {
        "ok": True,
        "artifact_count": result.artifact_count,
        "per_app": result.per_app,
        "by_category": result.by_category,
    }


def run_memory_analysis(deps: InvestigationDeps) -> dict:
    from dd_lab.memory import analyze_memory

    result = analyze_memory(deps.cases_root, deps.case_id, classifier=deps.classifier)
    if not result.ok:
        return {"ok": False, "skipped": True, "reason": result.skipped_reason}
    deps.state.memory_analyzed = True
    deps.state.save()
    return {
        "ok": True,
        "dumps": result.dumps,
        "artifacts_added": result.artifact_count,
        "volatility_used": result.volatility_used,
    }


def query_warehouse(deps: InvestigationDeps, sql: str, limit: int = 200) -> dict:
    if not _SELECT_ONLY.match(sql) or _FORBIDDEN.search(sql):
        return {"ok": False, "error": "only read-only SELECT/WITH queries are allowed"}
    if " limit " not in sql.lower():
        sql = sql.rstrip("; \n") + f" LIMIT {int(limit)}"
    wh = Warehouse(deps.layout.warehouse)
    try:
        rows = wh.query(sql)
    except Exception as e:  # noqa: BLE001 - surface the DB error to the agent
        return {"ok": False, "error": str(e)}
    finally:
        wh.close()
    return {"ok": True, "row_count": len(rows), "rows": rows[:limit]}


def category_summary(deps: InvestigationDeps) -> dict:
    wh = Warehouse(deps.layout.warehouse)
    try:
        wh.build_views()
        return {
            "by_category": wh.query("SELECT * FROM v_by_category"),
            "by_source": wh.query("SELECT * FROM v_source_breakdown"),
            "high_priority_count": wh.query("SELECT count(*) AS n FROM v_high_priority")[0]["n"],
        }
    finally:
        wh.close()


def search_keyword(deps: InvestigationDeps, keyword: str, limit: int = 50) -> dict:
    wh = Warehouse(deps.layout.warehouse)
    try:
        rows = wh.query(
            "SELECT app, source_type, category, key, substr(value,1,200) AS value, record_ts "
            "FROM artifacts WHERE value ILIKE ? OR key ILIKE ? "
            "ORDER BY (category != 'GENERAL') DESC, confidence DESC LIMIT ?",
            [f"%{keyword}%", f"%{keyword}%", int(limit)],
        )
    finally:
        wh.close()
    return {"keyword": keyword, "hits": len(rows), "rows": rows}


def generate_report(deps: InvestigationDeps) -> dict:
    path = build_report(deps.cases_root, deps.case_id, actor="agent")
    deps.state.report_path = str(path)
    deps.state.save()
    return {"ok": True, "report_path": str(path)}
