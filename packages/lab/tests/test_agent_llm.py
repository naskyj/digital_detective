"""Exercise the LLM agent loop with a scripted FunctionModel (no real LLM).

Proves the tool wiring, the audit mirroring, and that a tool-calling model can
drive the pipeline end to end.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from dd_lab.agent.audit import AuditLog
from dd_lab.agent.config import AgentConfig
from dd_lab.agent.runner import run_agent
from dd_lab.case import CaseMeta

from tests.sample_case import build_sample_case

_CANDIDATES = [
    Path.home() / "Library/Application Support/Code",
    Path.home() / "Library/Application Support/Postman",
    Path.home() / ".config/Code",
]
_FAST = frozenset({"local_storage", "session_storage", "preferences", "cookies", "local_state"})


def _profile():
    for p in _CANDIDATES:
        if (p / "Local Storage" / "leveldb").is_dir() or (p / "Preferences").is_file():
            return p
    return None


pytestmark = pytest.mark.skipif(_profile() is None, reason="no Electron profile available")


def _script():
    """A deterministic 'model' that walks the expected tool sequence."""
    steps = iter([
        "verify_integrity",
        "run_ingest",
        "category_summary",
        ("search_keyword", {"keyword": "token"}),
        "generate_report",
    ])

    def fn(messages: list[ModelRequest | ModelResponse], info: AgentInfo) -> ModelResponse:
        try:
            step = next(steps)
        except StopIteration:
            return ModelResponse(parts=[TextPart("Investigation complete; report generated.")])
        name, args = (step, {}) if isinstance(step, str) else step
        return ModelResponse(parts=[ToolCallPart(tool_name=name, args=args)])

    return FunctionModel(fn)


def test_agent_drives_pipeline(tmp_path):
    profile = _profile()
    root = tmp_path / "cases"
    build_sample_case(root, "case_llm_1", "app", profile, include=_FAST)
    from dd_core.casedir import CaseLayout

    CaseMeta(case_id="case_llm_1", name="LLM agent test", keywords=["token"]).save(
        CaseLayout(root, "case_llm_1")
    )

    progress_log: list[str] = []
    outcome = run_agent(
        root, "case_llm_1", AgentConfig(), model=_script(),
        progress=progress_log.append, enable_judge=False,
    )
    assert outcome.ok
    assert outcome.report_path and Path(outcome.report_path).is_file()

    records = AuditLog(CaseLayout(root, "case_llm_1").audit_log).records()
    called = [r.name for r in records if r.kind == "tool_call"]
    assert called == ["verify_integrity", "run_ingest", "category_summary",
                      "search_keyword", "generate_report"]
    assert any(r.kind == "run_end" for r in records)

    # every tool call the agent made streamed into the live progress log too,
    # not just the permanent audit trail - this is what the UI's job log shows
    # in real time while the agent works.
    assert any("Agent starting" in m for m in progress_log)
    for name in called:
        assert any(name in m for m in progress_log)
    assert "Analysis complete" in progress_log[-1]


def test_agent_ingests_even_when_the_model_tries_to_skip_straight_to_findings(tmp_path):
    """Regression for a real incident (2026-09-18): the LLM agent read
    case_status(), judged re-ingesting unnecessary because the case already
    had prior findings/report, and investigated stale pre-run data instead
    of ever calling run_ingest. Simulates exactly that model behaviour - a
    scripted model that never calls verify_integrity/run_ingest/
    run_memory_analysis at all, going straight to list_findings then
    generate_report - and asserts the warehouse is populated with real,
    freshly-ingested artifacts regardless, because those steps now run
    unconditionally in Python before the model gets a turn."""
    profile = _profile()
    root = tmp_path / "cases"
    build_sample_case(root, "case_llm_2", "app", profile, include=_FAST)
    from dd_core.casedir import CaseLayout

    layout = CaseLayout(root, "case_llm_2")
    CaseMeta(case_id="case_llm_2", name="LLM agent skip-ingest test").save(layout)

    # pre-seed PipelineState as if a prior run already completed everything -
    # exactly what made the real agent judge re-ingesting "already done".
    from dd_lab.agent.state import PipelineState

    state = PipelineState.load(layout.dir / "logs" / "pipeline_state.json", "case_llm_2")
    state.integrity_verified = True
    state.ingested_apps = ["app"]
    state.classified = True
    state.memory_analyzed = True
    state.finished = True
    state.save()

    steps = iter(["list_findings", "generate_report"])

    def fn(messages: list[ModelRequest | ModelResponse], info: AgentInfo) -> ModelResponse:
        try:
            name = next(steps)
        except StopIteration:
            return ModelResponse(parts=[TextPart("Investigation complete; report generated.")])
        return ModelResponse(parts=[ToolCallPart(tool_name=name, args={})])

    model = FunctionModel(fn)
    outcome = run_agent(root, "case_llm_2", AgentConfig(), model=model, enable_judge=False)
    assert outcome.ok

    records = AuditLog(layout.audit_log).records()
    called = [r.name for r in records if r.kind == "tool_call"]
    assert called == ["list_findings", "generate_report"]  # the model's own calls, never ingest

    decisions = {r.name for r in records if r.kind == "decision"}
    assert {"verify_integrity", "run_ingest", "run_memory_analysis"} <= decisions

    from dd_lab.etl.warehouse import Warehouse

    wh = Warehouse(layout.warehouse)
    try:
        assert wh.count() > 0  # real artifacts, from the Python-driven ingest, not stale/absent
    finally:
        wh.close()
