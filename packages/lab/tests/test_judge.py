"""Adversarial judge — scripted FunctionModel, no real LLM needed.

Proves: only VERIFIED findings are judged (a REJECTED_UNVERIFIED one never
reaches the judge), the verdict is written back onto the finding, and the
audit log records that an independent review happened.
"""

from __future__ import annotations

from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from dd_core.casedir import CaseLayout
from dd_lab.agent.audit import AuditLog
from dd_lab.agent.findings import Citation, Finding, FindingsStore, FindingStatus
from dd_lab.agent.judge import judge_findings


def _scripted_verdict(verdict: str, reasoning: str) -> FunctionModel:
    def fn(messages: list[ModelRequest | ModelResponse], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[
            ToolCallPart(tool_name="final_result", args={"verdict": verdict, "reasoning": reasoning}),
        ])

    return FunctionModel(fn)


def _finding(case_id: str, status: FindingStatus, finding_id: str) -> Finding:
    return Finding(
        finding_id=finding_id,
        case_id=case_id,
        claim="The email 'kid@example.com' appears in both Discord and Roblox evidence.",
        citations=[Citation(artifact_id="a1", ok=True, reason="verified", app="Discord",
                            source_file="discord/x", source_locator="loc1", quote="kid@example.com")],
        status=status,
    )


def test_judge_accepts_a_well_supported_finding(tmp_path):
    root, case_id = tmp_path, "case1"
    layout = CaseLayout(root, case_id)
    store = FindingsStore(layout.findings_log)
    store.add(_finding(case_id, FindingStatus.VERIFIED, "f1"))

    results = judge_findings(root, case_id, model=_scripted_verdict("accept", "citation supports the claim"))

    assert len(results) == 1
    assert results[0].verdict == "accept"

    updated = FindingsStore(layout.findings_log).get("f1")
    assert updated.status == FindingStatus.JUDGE_ACCEPTED
    assert updated.judge_verdict.verdict == "accept"
    assert "supports" in updated.judge_verdict.reasoning

    audit = AuditLog(layout.audit_log).records()
    assert any(r.kind == "judge_verdict" and r.name == "f1" for r in audit)


def test_judge_flags_an_overreaching_finding(tmp_path):
    root, case_id = tmp_path, "case2"
    layout = CaseLayout(root, case_id)
    FindingsStore(layout.findings_log).add(_finding(case_id, FindingStatus.VERIFIED, "f2"))

    judge_findings(root, case_id, model=_scripted_verdict("flag", "wording overreaches the evidence"))

    updated = FindingsStore(layout.findings_log).get("f2")
    assert updated.status == FindingStatus.JUDGE_FLAGGED


def test_unverified_finding_never_reaches_the_judge(tmp_path):
    root, case_id = tmp_path, "case3"
    layout = CaseLayout(root, case_id)
    FindingsStore(layout.findings_log).add(_finding(case_id, FindingStatus.REJECTED_UNVERIFIED, "f3"))

    called = {"n": 0}

    def fn(messages, info):
        called["n"] += 1
        return ModelResponse(parts=[
            ToolCallPart(tool_name="final_result", args={"verdict": "accept", "reasoning": "n/a"}),
        ])

    results = judge_findings(root, case_id, model=FunctionModel(fn))

    assert results == []
    assert called["n"] == 0
    assert FindingsStore(layout.findings_log).get("f3").status == FindingStatus.REJECTED_UNVERIFIED
