"""Pydantic AI agent that orchestrates a forensic investigation.

The agent only *orchestrates and interprets* — it chooses which deterministic
tool to run next, reacts to failures, and reasons over query results. It never
transforms evidence itself. Every tool call is mirrored to the case audit log by
the wrappers here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext

from dd_lab.agent import tools as T
from dd_lab.agent.config import AgentConfig, build_model
from dd_lab.agent.tools import InvestigationDeps

SYSTEM_PROMPT = """\
You are the investigation agent for Digital Detective, a local digital-forensics
tool for Electron-application evidence. You run one investigation case at a time.

Evidence integrity verification, parsing, classification, and memory analysis
already happened, unconditionally, before you were invoked — they are
mechanical, idempotent steps, never a judgment call, so they are no longer
your job to decide whether to run. `run_ingest`/`run_memory_analysis`/
`verify_integrity` remain available as tools only for a narrow, deliberate
follow-up (e.g. re-parsing one specific app after fixing something about it) —
do not call them as part of a normal investigation, and never skip
investigating because you assume nothing changed; the warehouse in front of
you right now is always the current, authoritative state to investigate.

Your job: investigate what's already in the warehouse and produce a finished
forensic report, citing everything.

Rules:
- You never invent findings. Every claim must trace to a tool result.
- Use `case_status` and `category_summary` to get oriented, then
  `query_warehouse` and `search_keyword` to investigate — in particular,
  check every investigation keyword the case provides.
- `query_warehouse` is read-only SQL over the `artifacts` table and the `v_*`
  views. Columns: app, source_type, source_file, key, value, value_type,
  record_ts, category, confidence.
- Call `correlate_entities` to find identifiers (ids, invite links, emails,
  asset ids) that appear across more than one app or storage mechanism —
  that cross-source link is the thing a per-row classifier cannot see. Its
  entity-extraction coverage is NOT exhaustive (app-specific id/token shapes
  get added over time) — a keyword/manual query can still surface something
  correlation missed, so don't treat an empty correlation result as "nothing
  to find" on its own. Use its timeline (already time-ordered) to reconstruct
  the sequence of events for an entity, and cross-platform links (e.g. a
  Roblox id that also shows up in Discord evidence) to connect identities
  across apps.
- Conclusions become part of the case ONLY via `write_finding` — never rely on
  your final text output to convey a finding. Call `write_finding` with the
  claim and every artifact_id (plus, where relevant, the exact quoted
  substring) that supports it; it re-verifies each citation itself and rejects
  the finding immediately if any citation doesn't check out, so use
  `verify_finding` beforehand on anything you're unsure of to avoid a rejected
  finding. State only what the cited artifacts show — never a legal
  conclusion (e.g. never assert a crime occurred); describe the evidentiary
  pattern and let a human investigator draw that conclusion.
- Call `generate_report` last, after writing your findings. Then call the
  final result.

Be concise in your reasoning. Prefer a few well-chosen queries over many.
"""


@dataclass
class InvestigationResult:
    output: str
    steps: int
    report_path: str | None


def build_agent(cfg: AgentConfig, model=None) -> Agent[InvestigationDeps, str]:
    agent = Agent(
        model or build_model(cfg),
        deps_type=InvestigationDeps,
        system_prompt=SYSTEM_PROMPT,
        retries=2,
        model_settings={"temperature": cfg.temperature},
    )

    def _audit_call(ctx: RunContext[InvestigationDeps], name: str, **params):
        ctx.deps.audit.append("tool_call", name, json.dumps(params)[:500], **params)
        brief = ", ".join(f"{k}={v!r}" for k, v in params.items() if v not in (None, "", []))[:160]
        ctx.deps.notify(f"Agent → {name}({brief})" if brief else f"Agent → {name}")

    def _audit_result(ctx: RunContext[InvestigationDeps], name: str, result: dict) -> dict:
        summary = {k: result[k] for k in
                   ("ok", "skipped", "artifact_count", "row_count", "hits", "entity_count", "reason",
                    "finding_id", "status", "count")
                   if k in result}
        ctx.deps.audit.append("tool_result", name, json.dumps(summary)[:500])
        brief = ", ".join(f"{k}={v!r}" for k, v in summary.items())[:160]
        ctx.deps.notify(f"  {name} → {brief}" if brief else f"  {name} done")
        return result

    @agent.tool
    def case_status(ctx: RunContext[InvestigationDeps]) -> dict:
        """Current case: name, keywords, what's already been done, artifact count."""
        _audit_call(ctx, "case_status")
        return _audit_result(ctx, "case_status", T.case_status(ctx.deps))

    @agent.tool
    def list_evidence(ctx: RunContext[InvestigationDeps]) -> dict:
        """List the applications and evidence sources present, and whether memory dumps exist."""
        _audit_call(ctx, "list_evidence")
        return _audit_result(ctx, "list_evidence", T.list_evidence(ctx.deps))

    @agent.tool
    def verify_integrity(ctx: RunContext[InvestigationDeps]) -> dict:
        """Re-hash every evidence file against the signed manifest and check
        the signature. Already run once, unconditionally, before you were
        invoked (a failure aborts the run before you ever start) - only call
        this again for a deliberate, narrow follow-up."""
        _audit_call(ctx, "verify_integrity")
        return _audit_result(ctx, "verify_integrity", T.verify_integrity(ctx.deps))

    @agent.tool
    def run_ingest(ctx: RunContext[InvestigationDeps], apps: list[str] | None = None) -> dict:
        """Parse evidence, classify every artifact, and load into the case
        warehouse. Already run once for every app, unconditionally, before
        you were invoked - only call this again for a deliberate, narrow
        follow-up (e.g. re-parsing one specific app via `apps` after fixing
        something about it), never as a normal step of investigating."""
        _audit_call(ctx, "run_ingest", apps=apps)
        return _audit_result(ctx, "run_ingest", T.run_ingest(ctx.deps, apps))

    @agent.tool
    def run_memory_analysis(ctx: RunContext[InvestigationDeps]) -> dict:
        """Re-analyse memory dumps if present - Volatility structural analysis
        (process list, network connections, loaded DLLs) plus process-scoped
        string carving when `vol` is available, whole-image carving as the
        fallback otherwise. Already run once, unconditionally, before you were
        invoked - only call this again for a deliberate, narrow follow-up."""
        _audit_call(ctx, "run_memory_analysis")
        return _audit_result(ctx, "run_memory_analysis", T.run_memory_analysis(ctx.deps))

    @agent.tool
    def query_warehouse(ctx: RunContext[InvestigationDeps], sql: str) -> dict:
        """Run a read-only SELECT/WITH query over the artifacts table and v_* views."""
        _audit_call(ctx, "query_warehouse", sql=sql)
        return _audit_result(ctx, "query_warehouse", T.query_warehouse(ctx.deps, sql))

    @agent.tool
    def category_summary(ctx: RunContext[InvestigationDeps]) -> dict:
        """Artifact counts per category and per source, plus the high-priority count."""
        _audit_call(ctx, "category_summary")
        return _audit_result(ctx, "category_summary", T.category_summary(ctx.deps))

    @agent.tool
    def search_keyword(ctx: RunContext[InvestigationDeps], keyword: str) -> dict:
        """Full-text search artifact keys and values for a keyword."""
        _audit_call(ctx, "search_keyword", keyword=keyword)
        return _audit_result(ctx, "search_keyword", T.search_keyword(ctx.deps, keyword))

    @agent.tool
    def correlate_entities(ctx: RunContext[InvestigationDeps], min_distinct_sources: int = 2) -> dict:
        """Find ids/links/emails/asset-ids that appear across more than one app
        or storage mechanism, each with a full citation and a timeline."""
        _audit_call(ctx, "correlate_entities", min_distinct_sources=min_distinct_sources)
        return _audit_result(
            ctx, "correlate_entities", T.correlate_entities(ctx.deps, min_distinct_sources)
        )

    @agent.tool
    def verify_finding(
        ctx: RunContext[InvestigationDeps], artifact_id: str, claim_substring: str | None = None
    ) -> dict:
        """Re-check a citation against the warehouse before writing a finding.
        Call this on every artifact_id you are about to cite."""
        _audit_call(ctx, "verify_finding", artifact_id=artifact_id, claim_substring=claim_substring)
        return _audit_result(
            ctx, "verify_finding", T.verify_finding(ctx.deps, artifact_id, claim_substring)
        )

    @agent.tool
    def write_finding(
        ctx: RunContext[InvestigationDeps],
        claim: str,
        citations: list[dict],
        entity_type: str | None = None,
        entity_value: str | None = None,
    ) -> dict:
        """Register a conclusion + its supporting citations
        ([{"artifact_id":..., "quote": optional}]). This is the only way a
        claim becomes part of the case; it is rejected immediately if any
        citation fails re-verification."""
        _audit_call(ctx, "write_finding", claim=claim, citations=citations)
        return _audit_result(
            ctx, "write_finding", T.write_finding(ctx.deps, claim, citations, entity_type, entity_value)
        )

    @agent.tool
    def list_findings(ctx: RunContext[InvestigationDeps], status: str | None = None) -> dict:
        """Findings recorded so far for this case, with their review status."""
        _audit_call(ctx, "list_findings", status=status)
        return _audit_result(ctx, "list_findings", T.list_findings(ctx.deps, status))

    @agent.tool
    def generate_report(ctx: RunContext[InvestigationDeps]) -> dict:
        """Generate the HTML forensic report for the case."""
        _audit_call(ctx, "generate_report")
        return _audit_result(ctx, "generate_report", T.generate_report(ctx.deps))

    return agent
