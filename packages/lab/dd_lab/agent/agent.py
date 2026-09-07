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
You are the orchestration agent for Digital Detective, a local digital-forensics
tool for Electron-application evidence. You run one investigation case at a time.

Your job: drive the case from raw acquired evidence to a finished, integrity-
verified forensic report, deciding the order of steps and handling failures.

Rules:
- ALWAYS verify evidence integrity before any analysis. If integrity fails, stop
  and report it — do not analyse tampered evidence.
- You never invent findings. Every claim must trace to a tool result.
- Use `run_ingest` once to parse + classify all evidence into the warehouse.
- Then use `category_summary`, `query_warehouse` and `search_keyword` to
  investigate — in particular, check every investigation keyword the case
  provides.
- `query_warehouse` is read-only SQL over the `artifacts` table and the `v_*`
  views. Columns: app, source_type, source_file, key, value, value_type,
  record_ts, category, confidence.
- Call `generate_report` last. Then call the final result.
- If a step was already done (see `case_status`), don't repeat it.

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

    def _audit_result(ctx: RunContext[InvestigationDeps], name: str, result: dict) -> dict:
        summary = {k: result[k] for k in ("ok", "skipped", "artifact_count", "row_count", "hits")
                   if k in result}
        ctx.deps.audit.append("tool_result", name, json.dumps(summary)[:500])
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
        """Re-hash every evidence file against the signed manifest and check the signature."""
        _audit_call(ctx, "verify_integrity")
        return _audit_result(ctx, "verify_integrity", T.verify_integrity(ctx.deps))

    @agent.tool
    def run_ingest(ctx: RunContext[InvestigationDeps], apps: list[str] | None = None) -> dict:
        """Parse all evidence, classify every artifact, and load into the case warehouse."""
        _audit_call(ctx, "run_ingest", apps=apps)
        return _audit_result(ctx, "run_ingest", T.run_ingest(ctx.deps, apps))

    @agent.tool
    def run_memory_analysis(ctx: RunContext[InvestigationDeps]) -> dict:
        """Analyse memory dumps if present (Volatility integration pending)."""
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
    def generate_report(ctx: RunContext[InvestigationDeps]) -> dict:
        """Generate the HTML forensic report for the case."""
        _audit_call(ctx, "generate_report")
        return _audit_result(ctx, "generate_report", T.generate_report(ctx.deps))

    return agent
