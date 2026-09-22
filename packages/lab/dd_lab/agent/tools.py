"""Deterministic tools the agent (or a plain script) drives.

Every function takes `InvestigationDeps` and returns a small JSON-able result.
They are pure orchestration over the Phase 0-3 machinery; none of them ask an
LLM and none of them mutate evidence. The agent decides *when* to call them; the
supervisor records *that* they were called.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from dd_core.casedir import CaseLayout
from dd_core.manifest import Manifest, verify_case
from dd_lab.agent.audit import AuditLog
from dd_lab.agent.findings import Citation, Finding, FindingsStore, FindingStatus
from dd_lab.agent.state import PipelineState
from dd_lab.case import CaseMeta
from dd_lab.classify import Classifier
from dd_lab.etl.warehouse import Warehouse
from dd_lab.ingest import IntegrityError, ingest_case
from dd_lab.investigate import build_correlation_report, verify_claim
from dd_lab.parsers import discover_apps
from dd_lab.parsers.base import MEMORY

# `dd_lab.report` imports `dd_lab.agent.audit` (for the audit-trail appendix),
# so importing it back here at module level would close an import cycle the
# moment anything imports `dd_lab.report` before `dd_lab.agent` — deferred to
# the one place it's actually used instead.

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
    # Optional live-progress sink (e.g. a running job's log) — separate from
    # `audit`, which is the permanent, reviewable record written regardless.
    # `notify()` is a no-op if this is unset, so every tool works the same
    # whether or not anyone is watching in real time.
    progress: Callable[[str], None] | None = None

    def notify(self, msg: str) -> None:
        if self.progress is not None:
            self.progress(msg)

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
            # memory analysis is its own explicit step (run_memory_analysis,
            # called separately by both run_pipeline and the LLM agent) -
            # ingest_case()'s own analyze_memory=True default would otherwise
            # run it here too, silently duplicating the memory-image scan and
            # every Volatility plugin call for nothing (the second pass's
            # rows all get rejected by the warehouse's ON CONFLICT DO NOTHING).
            analyze_memory=False,
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
        "by_category": result.by_category,
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
            "ORDER BY relevant DESC, confidence DESC LIMIT ?",
            [f"%{keyword}%", f"%{keyword}%", int(limit)],
        )
    finally:
        wh.close()
    return {"keyword": keyword, "hits": len(rows), "rows": rows}


def correlate_entities(deps: InvestigationDeps, min_distinct_sources: int = 2) -> dict:
    """Phase 2 — find identifiers (ids, invite links, emails, asset ids) that
    show up in more than one app or storage mechanism, e.g. the same Roblox
    asset id in a Cache blob, a memory-string carve, and an IndexedDB message.
    Every hit carries its full citation (artifact_id/app/source_file/locator)
    and a timestamp-ordered timeline — this is the cross-source link a single-
    row classifier cannot produce; use it to build hypotheses, then verify
    every claim with `verify_claim` before writing it into a finding."""
    wh = Warehouse(deps.layout.warehouse)
    try:
        entities = build_correlation_report(wh, min_distinct_sources=min_distinct_sources)
    finally:
        wh.close()
    return {
        "ok": True,
        "entity_count": len(entities),
        "entities": [
            {
                "entity_type": e.entity_type,
                "value": e.value,
                "apps": e.apps,
                "source_types": e.source_types,
                "timeline": [
                    {
                        "artifact_id": c.artifact_id, "app": c.app, "source_type": c.source_type,
                        "source_file": c.source_file, "source_locator": c.source_locator,
                        "key": c.key, "value": c.value, "record_ts": c.record_ts.isoformat()
                        if c.record_ts else None,
                        # Corroborating citations are kept regardless of relevance (see
                        # correlate.py::Citation's own comment) - surfaced here so the
                        # agent/chat can tell a genuinely relevant hit apart from an
                        # incidental one instead of treating every citation as equal.
                        "relevant": c.relevant, "relevance_score": c.relevance_score,
                    }
                    for c in e.timeline()
                ],
            }
            for e in entities
        ],
    }


def case_timeline(deps: InvestigationDeps, *, relevant_only: bool = True, limit: int = 200) -> dict:
    """The case-wide event timeline, oldest first - every timestamped
    artifact, not scoped to one correlated entity the way
    correlate_entities' own per-entity timeline is. Use this to reconstruct
    the overall sequence of events across the whole case; use
    correlate_entities when you already have a specific identifier and want
    just its own occurrences in order. Defaults to relevant_only=True (the
    primary signal this pipeline produces) - pass relevant_only=False only
    if you deliberately need the unfiltered view, e.g. to sanity-check
    whether something borderline was excluded."""
    wh = Warehouse(deps.layout.warehouse)
    try:
        clause = "WHERE record_ts IS NOT NULL" + (" AND relevant" if relevant_only else "")
        rows = wh.query(
            f"SELECT record_ts, app, source_type, category, relevant, relevance_score, "
            f"key, substr(value,1,300) AS value, artifact_id FROM artifacts {clause} "
            f"ORDER BY record_ts LIMIT ?",
            [int(limit)],
        )
    finally:
        wh.close()
    return {"relevant_only": relevant_only, "event_count": len(rows), "events": rows}


def explain_classification(deps: InvestigationDeps, artifact_id: str) -> dict:
    """Why one specific artifact was (or wasn't) marked relevant, and what
    category it landed in - the real per-voter breakdown (regex/keyword/
    NER/model/entropy votes, the trained relevance model's own raw score if
    one is configured, and the combined relevance_score/confidence), not a
    guess. Use this whenever asked to explain/justify a classification
    decision on a specific artifact, so the answer is grounded in what the
    ensemble actually did (see dd_lab/classify/ensemble.py's module
    docstring for how these combine) rather than invented."""
    wh = Warehouse(deps.layout.warehouse)
    try:
        rows = wh.query(
            "SELECT artifact_id, app, source_type, key, value, category, confidence, "
            "relevant, relevance_score, regex_vote, keyword_vote, ner_vote, model_vote, "
            "entropy_vote, relevance_model_score, deleted "
            "FROM artifacts WHERE artifact_id = ?",
            [artifact_id],
        )
    finally:
        wh.close()
    if not rows:
        return {"ok": False, "reason": f"no artifact with id {artifact_id!r}"}
    return {"ok": True, **rows[0]}


def _pivot_processes_by_pid(rows: list[dict]) -> list[dict]:
    """`rows` are flat memory_process artifacts - one row per Volatility
    column, one shared source_locator per (plugin, row-index) group (e.g.
    every column from the same pslist row carries source_locator
    "windows.pslist.PsList#3"; cmdline's PID/Process/Args share their own
    "windows.cmdline.CmdLine#3"). Pivot into one dict per process, keyed by
    its own PID column's value - the exact logic the Processes UI view
    (processesPanel's pivotByPid, dd_lab/ui/index.html) already does
    client-side, ported here so agent tools don't hand-roll SQL for it."""
    locator_pid: dict[str, str] = {}
    for r in rows:
        if r["key"].endswith(".PID"):
            locator_pid[r["source_locator"]] = r["value"]
    by_pid: dict[str, dict] = {}
    for r in rows:
        pid = locator_pid.get(r["source_locator"])
        if pid is None:
            continue  # a stray column with no PID sibling in its own group
        col = r["key"].rsplit(".", 1)[-1]
        by_pid.setdefault(pid, {"pid": pid})[col] = r["value"]
    return sorted(by_pid.values(), key=lambda p: int(p["pid"]))


def list_processes(deps: InvestigationDeps, app: str) -> dict:
    """Every process Volatility found belonging to `app` (root matches plus
    every spawned descendant, see match_target_processes), pivoted into one
    row per process with its PID/PPID/image name/command line/create time.
    Requires memory analysis to have run and `vol` to have been available on
    the analysis machine — an empty result usually means one of those, not
    that the app has no processes."""
    wh = Warehouse(deps.layout.warehouse)
    try:
        rows = wh.query(
            "SELECT key, value, source_locator FROM artifacts "
            "WHERE app = ? AND source_type = 'memory_process' LIMIT 5000",
            [app],
        )
    finally:
        wh.close()
    procs = _pivot_processes_by_pid(rows)
    return {"ok": True, "app": app, "process_count": len(procs), "processes": procs}


def process_memory(deps: InvestigationDeps, pid: int, limit: int = 50) -> dict:
    """Memory strings carved from one specific process's own memory, not the
    whole image — use this after `list_processes` to zoom into one PID
    instead of `search_keyword` over everything. Matches the "pid<PID>:"
    source_locator prefix the per-process carving path writes (see
    dd_lab/memory/analyze.py); returns nothing for a PID that was found
    structurally (in `list_processes`) but whose own memory couldn't be
    extracted and carved separately, or for a whole-image fallback carve
    (no target-app process matched, so nothing is PID-scoped at all)."""
    wh = Warehouse(deps.layout.warehouse)
    try:
        rows = wh.query(
            "SELECT app, source_type, category, key, substr(value,1,300) AS value, "
            "record_ts, source_file, source_locator, artifact_id FROM artifacts "
            "WHERE source_type = 'memory_string' AND source_locator LIKE ? "
            "ORDER BY relevant DESC, confidence DESC LIMIT ?",
            [f"pid{int(pid)}:%", int(limit)],
        )
    finally:
        wh.close()
    return {"ok": True, "pid": pid, "hits": len(rows), "rows": rows}


def verify_finding(deps: InvestigationDeps, artifact_id: str, claim_substring: str | None = None) -> dict:
    """Self-verification: re-read `artifact_id` straight from the warehouse and
    confirm `claim_substring` (if given) is actually present in its stored
    value. Call this on every artifact_id a finding cites BEFORE writing that
    finding into the report — a claim that fails this must not be reported."""
    wh = Warehouse(deps.layout.warehouse)
    try:
        v = verify_claim(wh, artifact_id, claim_substring)
    finally:
        wh.close()
    return {
        "ok": v.ok, "reason": v.reason, "app": v.app,
        "source_file": v.source_file, "source_locator": v.source_locator,
        "value": v.value, "value_sha256": v.value_sha256,
    }


def write_finding(
    deps: InvestigationDeps,
    claim: str,
    citations: list[dict],
    entity_type: str | None = None,
    entity_value: str | None = None,
    created_by: str = "agent",
) -> dict:
    """Register a conclusion with its supporting evidence. `citations` is a
    list of {"artifact_id": ..., "quote": <exact substring you are relying on,
    optional>}. Every citation is independently re-verified against the
    warehouse right now and its value/hash is snapshotted, so later drift in
    the data is detectable rather than silently followed. A finding with any
    citation that fails verification — or with no citations at all — is
    rejected immediately: it is never usable, never judged, never reported.
    This is the ONLY way a claim becomes part of the case; do not put
    conclusions only in your final text output.

    `created_by` defaults to "agent" (every existing caller - the
    investigation agent's own tool call, the deterministic pipeline's
    correlate_and_write_findings). The investigator-triggered "save as
    finding" endpoint (api/main.py's case_add_finding) passes
    "investigator" instead - report §4 already renders created_by per
    finding, so this is visible provenance for free, distinguishing a
    human-saved claim from one the agent wrote autonomously during its own
    investigation."""
    wh = Warehouse(deps.layout.warehouse)
    try:
        cite_results = [
            _verify_citation(wh, c.get("artifact_id", ""), c.get("quote")) for c in citations
        ]
    finally:
        wh.close()

    all_ok = bool(cite_results) and all(c.ok for c in cite_results)
    finding = Finding(
        finding_id=uuid.uuid4().hex[:16],
        case_id=deps.case_id,
        claim=claim,
        entity_type=entity_type,
        entity_value=entity_value,
        citations=cite_results,
        status=FindingStatus.VERIFIED if all_ok else FindingStatus.REJECTED_UNVERIFIED,
        created_by=created_by,
    )
    FindingsStore(deps.layout.findings_log).add(finding)
    return {
        "ok": all_ok,
        "finding_id": finding.finding_id,
        "status": str(finding.status),
        "unverified_citations": [c.artifact_id for c in cite_results if not c.ok],
    }


def _verify_citation(wh: Warehouse, artifact_id: str, quote: str | None) -> Citation:
    v = verify_claim(wh, artifact_id, quote)
    return Citation(
        artifact_id=artifact_id, ok=v.ok, reason=v.reason, app=v.app,
        source_file=v.source_file, source_locator=v.source_locator,
        quote=quote, value_sha256=v.value_sha256,
    )


def correlate_and_write_findings(deps: InvestigationDeps, min_distinct_sources: int = 2) -> dict:
    """Convenience used by the deterministic (no-LLM) pipeline: run
    correlate_entities and turn every cross-source hit directly into a finding,
    citing every artifact the entity was found in. An LLM agent should
    normally call correlate_entities + write_finding itself so it can inspect
    before committing to a claim; this skips straight to writing because the
    deterministic pipeline has no LLM to do that judgement."""
    wh = Warehouse(deps.layout.warehouse)
    try:
        entities = build_correlation_report(wh, min_distinct_sources=min_distinct_sources)
    finally:
        wh.close()

    written = []
    for e in entities:
        citations = [{"artifact_id": c.artifact_id, "quote": e.value} for c in e.artifacts]
        claim = (
            f"The {e.entity_type.replace('_', ' ')} '{e.value}' appears in {len(e.artifacts)} "
            f"artifact(s) across app(s) {', '.join(e.apps)} and source type(s) "
            f"{', '.join(e.source_types)}."
        )
        written.append(write_finding(deps, claim, citations, entity_type=e.entity_type, entity_value=e.value))
    return {"ok": True, "finding_count": len(written), "findings": written}


def list_findings(deps: InvestigationDeps, status: str | None = None) -> dict:
    """Current findings for this case and their review status."""
    store = FindingsStore(deps.layout.findings_log)
    findings = store.list(FindingStatus(status) if status else None)
    return {"ok": True, "count": len(findings), "findings": [f.model_dump(mode="json") for f in findings]}


def generate_report(deps: InvestigationDeps) -> dict:
    from dd_lab.report import build_report

    path = build_report(deps.cases_root, deps.case_id, actor="agent")
    deps.state.report_path = str(path)
    deps.state.save()
    return {"ok": True, "report_path": str(path)}
