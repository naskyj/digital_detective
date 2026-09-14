"""Regression eval — a synthetic case with known ground truth, run through the
whole Phase 2 chain: correlate -> write findings -> judge -> report.

This is what actually catches "the agent got quietly worse" — unlike the unit
tests for each piece, this fixes an expected outcome for a whole scenario and
fails if any link in the chain regresses. Extend this file with more planted
scenarios as real cases reveal new ones; keep the ground truth in the test,
never inferred from whatever the code currently does.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from dd_core.casedir import CaseLayout
from dd_core.enums import ArtifactCategory as C
from dd_core.enums import SourceType as S
from dd_core.manifest import AcquisitionHost, Manifest
from dd_core.schema import ClassificationResult, NormalizedArtifact
from dd_core.signing import generate_keypair
from dd_lab.agent.findings import FindingsStore, FindingStatus
from dd_lab.agent.judge import judge_findings
from dd_lab.agent.tools import InvestigationDeps, correlate_and_write_findings
from dd_lab.etl.warehouse import Warehouse
from dd_lab.report import build_report

_NOW = datetime(2026, 9, 1, tzinfo=UTC)
_ASSET_ID = "9876543210"


def _seed_signed_manifest(layout: CaseLayout, case_id: str) -> None:
    """build_report needs a real signed manifest for its integrity section;
    this scenario is about the analysis chain, not acquisition, so an empty
    (zero-entry) one is enough — it verifies trivially clean."""
    layout.create_evidence_skeleton()
    priv, pub = generate_keypair()
    manifest = Manifest(
        case_id=case_id, collector_version="test-0.0", operator="test",
        host=AcquisitionHost(hostname="test", os="test", os_version="0", username="test", timezone="UTC"),
        started_ts=_NOW, completed_ts=_NOW, target_apps=[], entries=[],
        public_key_pem=pub.decode(),
    )
    manifest.write(layout.dir, priv)


def _artifact(app, source_type, source_file, locator, key, value, category, minutes=0, case_id="case"):
    return NormalizedArtifact(
        artifact_id=f"{app}:{source_file}:{locator}", case_id=case_id, app=app,
        source_type=source_type, source_file=source_file, source_locator=locator, key=key,
        value=value, record_ts=_NOW + timedelta(minutes=minutes), extracted_ts=_NOW,
        classification=ClassificationResult(category=category, confidence=0.9),
    )


@pytest.fixture
def ground_truth_case(tmp_path):
    """Ground truth: one Roblox asset id planted across 3 apps/source types
    (must correlate); one unrelated single-occurrence artifact (must NOT)."""
    case_id = "ground_truth_case"
    layout = CaseLayout(tmp_path, case_id)
    _seed_signed_manifest(layout, case_id)
    wh = Warehouse(layout.warehouse)
    wh.load_artifacts([
        _artifact("Discord", S.CACHE, "discord/Cache/f1", "http-body",
                  "cache.body.embed.thumbnail", f"rbxassetid://{_ASSET_ID}", C.CHAT_CONTENT, 0,
                  case_id=case_id),
        _artifact("_system", S.MEMORY_STRING, "_system/memory/physmem.raw", "offset:100",
                  "memory.string", f"seen rbxassetid://{_ASSET_ID} in dm", C.CHAT_CONTENT, 5,
                  case_id=case_id),
        _artifact("Discord", S.INDEXEDDB, "discord/IndexedDB/messages", "store/998",
                  "content", f"check rbxassetid://{_ASSET_ID}", C.CHAT_CONTENT, 10, case_id=case_id),
        _artifact("Discord", S.CACHE, "discord/Cache/f2", "http-body",
                  "cache.body.unrelated", "totally unrelated one-off content", C.GENERAL, 0,
                  case_id=case_id),
    ])
    wh.close()
    return tmp_path, case_id


def test_end_to_end_correlate_write_judge_report(ground_truth_case):
    root, case_id = ground_truth_case
    deps = InvestigationDeps(cases_root=root, case_id=case_id)

    # 1. correlation -> findings: exactly the planted entity, nothing else
    result = correlate_and_write_findings(deps)
    assert result["finding_count"] == 1, result["findings"]

    findings = FindingsStore(deps.layout.findings_log).list()
    assert len(findings) == 1
    finding = findings[0]
    assert finding.entity_type == "roblox_asset_id"
    assert finding.entity_value == _ASSET_ID
    assert finding.status == FindingStatus.VERIFIED
    assert len(finding.citations) == 3
    assert {c.app for c in finding.citations} == {"Discord", "_system"}
    assert all(c.ok for c in finding.citations)

    # 2. judge (scripted "accept" - a well-supported, narrowly-worded claim)
    def fn(messages, info):
        return ModelResponse(parts=[ToolCallPart(
            tool_name="final_result",
            args={"verdict": "accept", "reasoning": "citations directly support the claim"},
        )])

    judged = judge_findings(root, case_id, model=FunctionModel(fn))
    assert len(judged) == 1
    assert judged[0].verdict == "accept"
    updated = FindingsStore(deps.layout.findings_log).get(finding.finding_id)
    assert updated.status == FindingStatus.JUDGE_ACCEPTED

    # 3. the report reflects all of the above, with citations, not just a raw count
    report_path = build_report(root, case_id, actor="test")
    html = report_path.read_text("utf-8")
    assert "roblox asset id" in html.lower()
    assert _ASSET_ID in html
    assert "judge_accepted" in html
    assert "discord/Cache/f1" in html and "discord/IndexedDB/messages" in html
    assert "_system/memory/physmem.raw" in html
    # the unrelated one-off must not be reported as a correlation or a finding
    assert "totally unrelated one-off content" not in html


def test_no_ground_truth_entity_means_no_finding(tmp_path):
    """A case with nothing cross-source planted must produce zero findings -
    the false-positive-rate half of the regression check."""
    case_id = "empty_case"
    layout = CaseLayout(tmp_path, case_id)
    wh = Warehouse(layout.warehouse)
    wh.load_artifacts([
        _artifact("Discord", S.CACHE, "discord/Cache/f1", "http-body",
                  "cache.body.x", "nothing correlated here", C.GENERAL, 0, case_id=case_id),
    ])
    wh.close()

    deps = InvestigationDeps(cases_root=tmp_path, case_id=case_id)
    result = correlate_and_write_findings(deps)
    assert result["finding_count"] == 0
