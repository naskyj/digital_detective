"""Multi-turn, read-mostly chat over the same investigation tool substrate.

Phase 6 of the agent-improvement plan (magical-dazzling-dragon.md) - the last
phase, deliberately scoped only once Phases 1-5's tool substrate was final
(no existing chat UI pattern in this codebase to mirror, so this is new
ground rather than a port of something already proven here).

Deliberately narrower than the investigation agent (agent.py):
- No verify_integrity/run_ingest/run_memory_analysis — those already run
  unconditionally before either agent starts (see runner.py::run_agent);
  a chat turn re-running Volatility over a multi-GB memory image because a
  user asked a casual question would be a very surprising side effect.
- No write_finding/generate_report — a chat reply reading the warehouse
  correctly is not the same thing as a citation-checked finding. Findings
  still only ever get written by the investigation agent/pipeline's own
  explicit write_finding call; chat can point a user at what it found and
  suggest they ask for a full investigation, never write on their behalf.

Everything registered here is a read: case_status, list_evidence,
category_summary, query_warehouse, search_keyword, correlate_entities,
case_timeline, explain_classification, list_processes, process_memory,
list_findings, verify_finding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from dd_lab.agent import tools as T
from dd_lab.agent.config import AgentConfig, build_model
from dd_lab.agent.tools import InvestigationDeps

CHAT_SYSTEM_PROMPT = """\
You are a conversational investigation assistant for Digital Detective, a
local digital-forensics tool. You're chatting with a human investigator
about ONE case whose evidence has already been parsed, classified, and
memory-analysed — you are answering questions about what's already in the
warehouse, not running new analysis.

Rules:
- You never invent facts. Every claim you make must come from a tool result
  you actually got back this conversation — quote or closely paraphrase it,
  and name the artifact_id/app/source_file it came from so the investigator
  can look it up themselves.
- Use `case_status`/`list_evidence`/`category_summary` to get oriented if you
  don't already know the shape of the case from earlier in this
  conversation. Use `query_warehouse`/`search_keyword` to look things up, and
  `correlate_entities` when asked about links between apps or identifiers
  that show up in more than one place. Use `list_processes`/`process_memory`
  for questions about a specific process's own memory.
- Use `case_timeline` when asked to reconstruct the sequence of events
  across the WHOLE case (not one specific identifier — that's
  `correlate_entities`' own per-entity timeline instead).
- Use `explain_classification` whenever asked why a specific artifact was
  (or wasn't) marked relevant, or what category it landed in — answer from
  its actual per-voter breakdown (regex/keyword/NER/model/entropy votes,
  the relevance score and, if a relevance model is configured, its own raw
  score), not a guess about what probably happened.
- You cannot write findings or generate a report from chat yourself — that
  requires the investigator's own explicit action (a "Save as finding"
  step outside this conversation, which independently re-verifies whatever
  you cite before it becomes part of the case), or the full investigation
  run. If asked to write a finding or produce the report, say so plainly
  and point at the specific artifact_id(s)/quotes the investigator would
  need to save it themselves, rather than attempting a workaround.
- `list_findings`/`verify_finding` are read-only checks on findings already
  written by that investigation run — you may report and verify those.
- Be direct and concise. Prefer answering with a short synthesis backed by
  citations over dumping a raw table, but show the actual values/quotes you
  relied on, not just your paraphrase of them.
"""


def _history_path(deps: InvestigationDeps) -> Path:
    return deps.layout.dir / "logs" / "chat_history.json"


def load_history(deps: InvestigationDeps) -> list[ModelMessage]:
    path = _history_path(deps)
    if not path.is_file():
        return []
    try:
        return ModelMessagesTypeAdapter.validate_json(path.read_bytes())
    except Exception:  # noqa: BLE001 - a corrupt/old-format history must never block chat
        return []


def save_history(deps: InvestigationDeps, messages: list[ModelMessage]) -> None:
    path = _history_path(deps)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(ModelMessagesTypeAdapter.dump_json(messages))


def history_for_ui(messages: list[ModelMessage]) -> list[dict]:
    """Flatten persisted messages to the simple {role, text} shape the chat
    panel renders. Tool calls/returns are internal reasoning steps (already
    in the case's own audit log, same as every other agent tool call - see
    `_audit_call`/`_audit_result` below), not part of the visible transcript."""
    turns: list[dict] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            text = "\n".join(
                p.content for p in msg.parts if isinstance(p, UserPromptPart) and isinstance(p.content, str)
            )
            if text:
                turns.append({"role": "user", "text": text})
        elif isinstance(msg, ModelResponse):
            text = "\n".join(p.content for p in msg.parts if isinstance(p, TextPart) and p.content)
            if text:
                turns.append({"role": "assistant", "text": text})
    return turns


def build_chat_agent(cfg: AgentConfig, model=None) -> Agent[InvestigationDeps, str]:
    agent = Agent(
        model or build_model(cfg),
        deps_type=InvestigationDeps,
        system_prompt=CHAT_SYSTEM_PROMPT,
        retries=2,
        model_settings={"temperature": cfg.temperature},
    )

    # Same tool_call/tool_result audit-trail convention as agent.py's
    # build_agent, under a "chat_"-prefixed kind so a case's audit log can
    # tell an investigation run's tool calls apart from a chat session's.
    def _audit_call(ctx: RunContext[InvestigationDeps], name: str, **params):
        ctx.deps.audit.append("chat_tool_call", name, json.dumps(params)[:500], **params)

    def _audit_result(ctx: RunContext[InvestigationDeps], name: str, result: dict) -> dict:
        summary = {k: result[k] for k in
                   ("ok", "hits", "entity_count", "row_count", "count", "process_count")
                   if k in result}
        ctx.deps.audit.append("chat_tool_result", name, json.dumps(summary)[:500])
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
    def category_summary(ctx: RunContext[InvestigationDeps]) -> dict:
        """Artifact counts per category and per source, plus the high-priority count."""
        _audit_call(ctx, "category_summary")
        return _audit_result(ctx, "category_summary", T.category_summary(ctx.deps))

    @agent.tool
    def query_warehouse(ctx: RunContext[InvestigationDeps], sql: str) -> dict:
        """Run a read-only SELECT/WITH query over the artifacts table and v_* views."""
        _audit_call(ctx, "query_warehouse", sql=sql)
        return _audit_result(ctx, "query_warehouse", T.query_warehouse(ctx.deps, sql))

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
    def case_timeline(
        ctx: RunContext[InvestigationDeps], relevant_only: bool = True, limit: int = 200
    ) -> dict:
        """The case-wide event timeline, oldest first (not scoped to one
        entity the way correlate_entities' own timeline is)."""
        _audit_call(ctx, "case_timeline", relevant_only=relevant_only, limit=limit)
        return _audit_result(
            ctx, "case_timeline", T.case_timeline(ctx.deps, relevant_only=relevant_only, limit=limit)
        )

    @agent.tool
    def explain_classification(ctx: RunContext[InvestigationDeps], artifact_id: str) -> dict:
        """Why one artifact was (or wasn't) marked relevant and what category
        it landed in - the real per-voter breakdown, not a guess."""
        _audit_call(ctx, "explain_classification", artifact_id=artifact_id)
        return _audit_result(
            ctx, "explain_classification", T.explain_classification(ctx.deps, artifact_id)
        )

    @agent.tool
    def list_processes(ctx: RunContext[InvestigationDeps], app: str) -> dict:
        """Every process Volatility found belonging to `app`, pivoted into one
        row per process (PID/PPID/image name/command line/create time)."""
        _audit_call(ctx, "list_processes", app=app)
        return _audit_result(ctx, "list_processes", T.list_processes(ctx.deps, app))

    @agent.tool
    def process_memory(ctx: RunContext[InvestigationDeps], pid: int, limit: int = 50) -> dict:
        """Memory strings carved from one specific process's own memory, not the whole image."""
        _audit_call(ctx, "process_memory", pid=pid, limit=limit)
        return _audit_result(ctx, "process_memory", T.process_memory(ctx.deps, pid, limit))

    @agent.tool
    def list_findings(ctx: RunContext[InvestigationDeps], status: str | None = None) -> dict:
        """Findings already written for this case by an investigation run, with their review status."""
        _audit_call(ctx, "list_findings", status=status)
        return _audit_result(ctx, "list_findings", T.list_findings(ctx.deps, status))

    @agent.tool
    def verify_finding(
        ctx: RunContext[InvestigationDeps], artifact_id: str, claim_substring: str | None = None
    ) -> dict:
        """Re-check whether `claim_substring` (if given) is actually present in artifact_id's stored value."""
        _audit_call(ctx, "verify_finding", artifact_id=artifact_id, claim_substring=claim_substring)
        return _audit_result(
            ctx, "verify_finding", T.verify_finding(ctx.deps, artifact_id, claim_substring)
        )

    return agent


@dataclass
class ChatTurnResult:
    ok: bool
    reply: str | None
    error: str | None = None


def run_chat_turn(
    deps: InvestigationDeps, cfg: AgentConfig, message: str, model=None
) -> ChatTurnResult:
    """One request/response chat turn: load persisted history, run the chat
    agent with it, persist the full updated history, return the reply.
    Synchronous by design (a chat reply, even from a real LLM, is a normal
    few-second wait for this kind of UI - no streaming plumbing yet, matching
    the plan's "request/response to start")."""
    if not deps.layout.warehouse.is_file():
        return ChatTurnResult(ok=False, reply=None, error="This case hasn't been analysed yet — run analysis first.")
    history = load_history(deps)
    agent = build_chat_agent(cfg, model)
    try:
        result = agent.run_sync(message, deps=deps, message_history=history)
    except Exception as e:  # noqa: BLE001 - surface the failure to the UI rather than 500ing
        return ChatTurnResult(ok=False, reply=None, error=str(e))
    save_history(deps, result.all_messages())
    return ChatTurnResult(ok=True, reply=result.output)
