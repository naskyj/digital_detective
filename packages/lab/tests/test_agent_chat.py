"""dd_lab.agent.chat — Phase 6 of the agent-improvement plan. A scripted
FunctionModel (no real LLM), same convention as test_agent_llm.py, over a
lightweight hand-loaded warehouse (test_findings_tool.py's pattern) rather
than a full pipeline run - chat doesn't need real evidence, just a case with
a warehouse."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from dd_core.casedir import CaseLayout
from dd_core.enums import ArtifactCategory as C
from dd_core.enums import SourceType as S
from dd_core.schema import ClassificationResult, NormalizedArtifact
from dd_lab.agent.chat import build_chat_agent, history_for_ui, load_history, run_chat_turn
from dd_lab.agent.config import AgentConfig
from dd_lab.agent.tools import InvestigationDeps
from dd_lab.etl.warehouse import Warehouse


def _deps(tmp_path, case_id="case1") -> InvestigationDeps:
    layout = CaseLayout(tmp_path, case_id)
    wh = Warehouse(layout.warehouse)
    wh.load_artifacts([
        NormalizedArtifact(
            artifact_id="a1", case_id=case_id, app="Discord", source_type=S.INDEXEDDB,
            source_file="discord/IndexedDB/x", source_locator="loc1", key="content",
            value="the password is hunter2", extracted_ts=datetime(2026, 1, 1, tzinfo=UTC),
            classification=ClassificationResult(category=C.CREDENTIAL, confidence=0.9),
        ),
    ])
    wh.close()
    return InvestigationDeps(cases_root=tmp_path, case_id=case_id)


def test_chat_on_an_unanalysed_case_fails_cleanly_without_calling_the_model(tmp_path):
    layout = CaseLayout(tmp_path, "case_unanalysed")
    layout.dir.mkdir(parents=True)
    deps = InvestigationDeps(cases_root=tmp_path, case_id="case_unanalysed")

    def fn(messages, info: AgentInfo) -> ModelResponse:
        raise AssertionError("the model must never be invoked for an unanalysed case")

    result = run_chat_turn(deps, AgentConfig(), "what's in this case?", model=FunctionModel(fn))
    assert result.ok is False
    assert "run analysis" in result.error.lower() or "hasn't been analysed" in result.error.lower()


def test_chat_answers_using_a_tool_and_persists_history(tmp_path):
    deps = _deps(tmp_path)

    def fn(messages: list[ModelRequest | ModelResponse], info: AgentInfo) -> ModelResponse:
        if not any(isinstance(m, ModelResponse) for m in messages):
            return ModelResponse(parts=[ToolCallPart(tool_name="search_keyword", args={"keyword": "password"})])
        return ModelResponse(parts=[TextPart("Found a credential: 'the password is hunter2' (a1).")])

    result = run_chat_turn(deps, AgentConfig(), "any credentials in this case?", model=FunctionModel(fn))

    assert result.ok is True
    assert "hunter2" in result.reply

    # persisted to disk, readable back via load_history + flattened for the UI
    history = load_history(deps)
    turns = history_for_ui(history)
    assert turns[0] == {"role": "user", "text": "any credentials in this case?"}
    assert turns[-1]["role"] == "assistant"
    assert "hunter2" in turns[-1]["text"]
    # the tool call itself doesn't leak into the user-visible transcript
    assert not any("search_keyword" in t["text"] for t in turns)


def test_chat_tool_calls_are_audited_under_a_chat_prefixed_kind(tmp_path):
    deps = _deps(tmp_path)

    def fn(messages, info: AgentInfo) -> ModelResponse:
        if not any(isinstance(m, ModelResponse) for m in messages):
            return ModelResponse(parts=[ToolCallPart(tool_name="category_summary", args={})])
        return ModelResponse(parts=[TextPart("done")])

    run_chat_turn(deps, AgentConfig(), "summarise", model=FunctionModel(fn))

    kinds = {r.kind for r in deps.audit.records()}
    assert "chat_tool_call" in kinds
    assert "chat_tool_result" in kinds
    # and NOT mixed in with an investigation run's own "tool_call"/"tool_result"
    # kind (agent.py's build_agent) - kept distinguishable in one shared log.
    assert "tool_call" not in kinds


def test_chat_second_turn_sees_the_first_turns_history(tmp_path):
    deps = _deps(tmp_path)
    seen_message_counts = []

    def fn(messages: list[ModelRequest | ModelResponse], info: AgentInfo) -> ModelResponse:
        seen_message_counts.append(len(messages))
        return ModelResponse(parts=[TextPart(f"reply {len(seen_message_counts)}")])

    r1 = run_chat_turn(deps, AgentConfig(), "first question", model=FunctionModel(fn))
    r2 = run_chat_turn(deps, AgentConfig(), "second question", model=FunctionModel(fn))

    assert r1.ok and r2.ok
    assert r1.reply == "reply 1"
    assert r2.reply == "reply 2"
    # the second call's model function saw strictly more messages than the
    # first - proof message_history from turn 1 was actually passed in, not
    # a fresh conversation each time.
    assert seen_message_counts[1] > seen_message_counts[0]

    turns = history_for_ui(load_history(deps))
    assert [t["text"] for t in turns] == ["first question", "reply 1", "second question", "reply 2"]


def test_chat_agent_has_no_mutating_or_expensive_tools(tmp_path):
    """Phase 6's whole safety design: write_finding/generate_report/run_ingest/
    run_memory_analysis/verify_integrity must never be reachable from chat."""
    agent = build_chat_agent(AgentConfig(), model=FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart("x")])))
    tool_names = {t.name for t in agent.toolsets[0].tools.values()}
    forbidden = {"write_finding", "generate_report", "run_ingest", "run_memory_analysis", "verify_integrity"}
    assert not (tool_names & forbidden), tool_names & forbidden
    assert "query_warehouse" in tool_names and "correlate_entities" in tool_names


def test_history_for_ui_ignores_a_tool_call_only_response():
    messages = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        ModelResponse(parts=[ToolCallPart(tool_name="case_status", args={})]),
        ModelResponse(parts=[TextPart("hi there")]),
    ]
    assert history_for_ui(messages) == [
        {"role": "user", "text": "hello"},
        {"role": "assistant", "text": "hi there"},
    ]
