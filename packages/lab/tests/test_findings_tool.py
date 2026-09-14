"""agent/tools.py::write_finding + list_findings — no real evidence needed,
just a warehouse with one known artifact."""

from __future__ import annotations

from datetime import UTC, datetime

from dd_core.casedir import CaseLayout
from dd_core.enums import ArtifactCategory as C
from dd_core.enums import SourceType as S
from dd_core.schema import ClassificationResult, NormalizedArtifact
from dd_lab.agent.findings import FindingStatus
from dd_lab.agent.tools import InvestigationDeps, list_findings, write_finding
from dd_lab.etl.warehouse import Warehouse


def _deps(tmp_path, case_id="case1") -> InvestigationDeps:
    layout = CaseLayout(tmp_path, case_id)
    wh = Warehouse(layout.warehouse)
    wh.load_artifacts([
        NormalizedArtifact(
            artifact_id="a1", case_id=case_id, app="Discord", source_type=S.INDEXEDDB,
            source_file="discord/IndexedDB/x", source_locator="loc1", key="content",
            value="hello world", extracted_ts=datetime(2026, 1, 1, tzinfo=UTC),
            classification=ClassificationResult(category=C.CHAT_CONTENT, confidence=0.9),
        ),
    ])
    wh.close()
    return InvestigationDeps(cases_root=tmp_path, case_id=case_id)


def test_write_finding_with_good_citation_is_verified(tmp_path):
    deps = _deps(tmp_path)
    result = write_finding(
        deps, "Discord evidence shows 'hello world'.",
        [{"artifact_id": "a1", "quote": "hello world"}],
    )

    assert result["ok"] is True
    assert result["status"] == FindingStatus.VERIFIED
    assert result["unverified_citations"] == []


def test_write_finding_with_bad_artifact_id_is_rejected(tmp_path):
    deps = _deps(tmp_path)
    result = write_finding(deps, "made up claim", [{"artifact_id": "does-not-exist"}])

    assert result["ok"] is False
    assert result["status"] == FindingStatus.REJECTED_UNVERIFIED
    assert result["unverified_citations"] == ["does-not-exist"]


def test_write_finding_with_misquote_is_rejected(tmp_path):
    deps = _deps(tmp_path)
    result = write_finding(
        deps, "claims something not actually there",
        [{"artifact_id": "a1", "quote": "this text is not in the artifact"}],
    )

    assert result["ok"] is False
    assert result["status"] == FindingStatus.REJECTED_UNVERIFIED


def test_write_finding_with_no_citations_is_rejected(tmp_path):
    deps = _deps(tmp_path)
    result = write_finding(deps, "unsupported claim", [])
    assert result["ok"] is False


def test_list_findings_reports_everything_written(tmp_path):
    deps = _deps(tmp_path)
    write_finding(deps, "good", [{"artifact_id": "a1", "quote": "hello world"}])
    write_finding(deps, "bad", [{"artifact_id": "nope"}])

    everything = list_findings(deps)
    assert everything["count"] == 2

    verified_only = list_findings(deps, status="verified")
    assert verified_only["count"] == 1
    assert verified_only["findings"][0]["claim"] == "good"
