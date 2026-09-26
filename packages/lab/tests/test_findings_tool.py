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


def test_write_finding_created_by_defaults_to_agent_but_can_be_overridden(tmp_path):
    """Every existing caller (the investigation agent's own tool call, the
    deterministic pipeline's correlate_and_write_findings) relies on the
    "agent" default - confirmed unchanged. The investigator-triggered
    "save as finding" API endpoint (main.py::case_add_finding) is the one
    real caller that passes created_by="investigator" - this is the direct,
    tools.py-level proof that value actually lands on the stored Finding
    (visible in the report's §4 provenance line), not just accepted and
    dropped."""
    deps = _deps(tmp_path)
    default_result = write_finding(deps, "default caller", [{"artifact_id": "a1"}])
    investigator_result = write_finding(
        deps, "investigator caller", [{"artifact_id": "a1"}], created_by="investigator",
    )

    findings = {f["finding_id"]: f for f in list_findings(deps)["findings"]}
    assert findings[default_result["finding_id"]]["created_by"] == "agent"
    assert findings[investigator_result["finding_id"]]["created_by"] == "investigator"


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


# ------------------------------------------------- superseded findings
#
# findings.json is append-only. An email pattern once matched retina-image
# filenames, and agent runs on 2026-09-17/19 wrote 199 findings asserting
# things like "the email 'wechat@3x.png' appears in N artifacts". The pattern
# was fixed on 09-23 - but every report generated afterwards still carried
# all 199, because nothing re-checks a finding's PREMISE. write_finding only
# verifies that each cited artifact exists and contains the quoted text, and
# that remained true throughout.


def test_supersede_stale_marks_an_entity_the_extractor_no_longer_recognises(tmp_path):
    from dd_lab.agent.findings import Finding, FindingsStore, FindingStatus

    store = FindingsStore(tmp_path / "findings.json")
    bogus = store.add(Finding(
        finding_id="f1", case_id="c", claim="The email 'wechat@3x.png' appears in 4 artifacts.",
        entity_type="email", entity_value="wechat@3x.png",
        citations=[], status=FindingStatus.VERIFIED, created_by="agent"))
    real = store.add(Finding(
        finding_id="f2", case_id="c", claim="The email 'appro@openssl.org' appears in 3 artifacts.",
        entity_type="email", entity_value="appro@openssl.org",
        citations=[], status=FindingStatus.VERIFIED, created_by="agent"))

    stale = store.supersede_stale()

    assert [f.finding_id for f in stale] == ["f1"]
    assert store.get(bogus.finding_id).status is FindingStatus.SUPERSEDED
    assert store.get(real.finding_id).status is FindingStatus.VERIFIED


def test_supersede_stale_respects_context_gated_entity_types(tmp_path):
    """A Slack team id shape is too generic to trust outside Slack's own
    evidence, so its extractor is app-gated. Checking it with no app context
    answers the wrong question - an early version would have superseded the
    real case's top-ranked correlation, T0C1J4SDLG2."""
    from dd_lab.agent.findings import Finding, FindingsStore, FindingStatus

    store = FindingsStore(tmp_path / "findings.json")
    store.add(Finding(
        finding_id="f1", case_id="c", claim="Slack workspace T0C1J4SDLG2.",
        entity_type="slack_team_id", entity_value="T0C1J4SDLG2",
        citations=[], status=FindingStatus.VERIFIED, created_by="agent"))

    assert store.supersede_stale() == []
    assert store.get("f1").status is FindingStatus.VERIFIED


def test_types_outside_the_extractors_vocabulary_are_left_alone(tmp_path):
    """An LLM-authored finding about a session_token must not be superseded
    merely because no pattern emits that type - absence means nothing there."""
    from dd_lab.agent.findings import Finding, FindingsStore, FindingStatus

    store = FindingsStore(tmp_path / "findings.json")
    store.add(Finding(
        finding_id="f1", case_id="c", claim="A CSRF token was present.",
        entity_type="session_token", entity_value="x-csrf-token (www.roblox.com)",
        citations=[], status=FindingStatus.VERIFIED, created_by="agent"))

    assert store.supersede_stale() == []
    assert store.get("f1").status is FindingStatus.VERIFIED


def test_dry_run_changes_nothing(tmp_path):
    from dd_lab.agent.findings import Finding, FindingsStore, FindingStatus

    store = FindingsStore(tmp_path / "findings.json")
    store.add(Finding(
        finding_id="f1", case_id="c", claim="x", entity_type="email",
        entity_value="wechat@3x.png", citations=[],
        status=FindingStatus.VERIFIED, created_by="agent"))

    assert len(store.supersede_stale(dry_run=True)) == 1
    assert store.get("f1").status is FindingStatus.VERIFIED

    assert len(store.supersede_stale()) == 1
    assert store.get("f1").status is FindingStatus.SUPERSEDED
    # idempotent - an already-superseded finding is not re-reported
    assert store.supersede_stale() == []


def test_nothing_is_ever_deleted(tmp_path):
    """A conclusion that was drawn and later invalidated is part of the case
    history; a report that quietly loses it is worse than one showing it
    marked."""
    from dd_lab.agent.findings import Finding, FindingsStore

    store = FindingsStore(tmp_path / "findings.json")
    store.add(Finding(
        finding_id="f1", case_id="c", claim="the original claim text",
        entity_type="email", entity_value="wechat@3x.png", citations=[],
        created_by="agent"))
    store.supersede_stale()

    kept = store.get("f1")
    assert kept is not None
    assert kept.claim == "the original claim text"
    assert len(store.list()) == 1
