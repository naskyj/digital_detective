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
