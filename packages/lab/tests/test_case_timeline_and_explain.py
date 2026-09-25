"""agent/tools.py::case_timeline + explain_classification — new tools this
session (see plan magical-dazzling-dragon.md): a case-wide timeline (not
scoped to one correlated entity the way correlate_entities' own timeline
is), and grounding "why was this classified this way" answers in the real
per-voter breakdown instead of a guess. No real evidence needed, just a
warehouse with a couple of known artifacts - same convention as
test_findings_tool.py."""

from __future__ import annotations

from datetime import UTC, datetime

from dd_core.casedir import CaseLayout
from dd_core.enums import ArtifactCategory as C
from dd_core.enums import SourceType as S
from dd_core.schema import ClassificationResult, NormalizedArtifact
from dd_lab.agent.tools import InvestigationDeps, case_timeline, explain_classification
from dd_lab.etl.warehouse import Warehouse


def _deps(tmp_path, case_id="case1") -> InvestigationDeps:
    layout = CaseLayout(tmp_path, case_id)
    wh = Warehouse(layout.warehouse)
    wh.load_artifacts([
        NormalizedArtifact(
            artifact_id="relevant-1", case_id=case_id, app="Discord", source_type=S.INDEXEDDB,
            source_file="discord/IndexedDB/x", source_locator="loc1", key="content",
            value="hello world", record_ts=datetime(2026, 1, 1, tzinfo=UTC),
            extracted_ts=datetime(2026, 1, 1, tzinfo=UTC),
            classification=ClassificationResult(
                category=C.CHAT_CONTENT, confidence=0.9, relevant=True, relevance_score=0.9,
                regex_vote=C.CHAT_CONTENT, classifier_version="test",
            ),
        ),
        NormalizedArtifact(
            artifact_id="not-relevant-1", case_id=case_id, app="Discord", source_type=S.CACHE,
            source_file="discord/Cache/f1", source_locator="loc2", key="request_time",
            value="1699999999", record_ts=datetime(2026, 1, 2, tzinfo=UTC),
            extracted_ts=datetime(2026, 1, 1, tzinfo=UTC),
            classification=ClassificationResult(
                category=C.GENERAL, confidence=0.0, relevant=False, relevance_score=0.0,
                classifier_version="test",
            ),
        ),
        NormalizedArtifact(
            artifact_id="no-timestamp", case_id=case_id, app="Discord", source_type=S.INDEXEDDB,
            source_file="discord/IndexedDB/x", source_locator="loc3", key="content",
            value="no ts here", extracted_ts=datetime(2026, 1, 1, tzinfo=UTC),
            classification=ClassificationResult(
                category=C.CHAT_CONTENT, confidence=0.9, relevant=True, relevance_score=0.9,
                classifier_version="test",
            ),
        ),
    ])
    wh.close()
    return InvestigationDeps(cases_root=tmp_path, case_id=case_id)


def test_case_timeline_defaults_to_relevant_only(tmp_path):
    deps = _deps(tmp_path)
    result = case_timeline(deps)
    assert result["relevant_only"] is True
    ids = {e["artifact_id"] for e in result["events"]}
    assert ids == {"relevant-1"}  # not-relevant-1 excluded, no-timestamp excluded (no record_ts)


def test_case_timeline_relevant_only_false_includes_everything_timestamped(tmp_path):
    deps = _deps(tmp_path)
    result = case_timeline(deps, relevant_only=False)
    ids = {e["artifact_id"] for e in result["events"]}
    assert ids == {"relevant-1", "not-relevant-1"}  # still excludes the untimestamped row
    assert result["event_count"] == 2


def test_case_timeline_is_oldest_first(tmp_path):
    deps = _deps(tmp_path)
    result = case_timeline(deps, relevant_only=False)
    ts = [e["record_ts"] for e in result["events"]]
    assert ts == sorted(ts)


def test_explain_classification_returns_the_real_per_voter_breakdown(tmp_path):
    deps = _deps(tmp_path)
    result = explain_classification(deps, "relevant-1")
    assert result["ok"] is True
    assert result["category"] == "CHAT_CONTENT"
    assert result["relevant"] is True
    assert result["relevance_score"] == 0.9
    assert result["regex_vote"] == "CHAT_CONTENT"
    # Fields the ensemble always sets, present even when unused by this row
    for col in ("keyword_vote", "ner_vote", "model_vote", "entropy_vote", "relevance_model_score"):
        assert col in result


def test_explain_classification_unknown_artifact_id(tmp_path):
    deps = _deps(tmp_path)
    result = explain_classification(deps, "does-not-exist")
    assert result["ok"] is False
    assert "does-not-exist" in result["reason"]


# --------------------------------------------------- narrative on the tool
#
# case_timeline used to return a flat `events` list and nothing else, with
# `event_count = len(rows)` - i.e. capped at `limit`, so the model could not
# learn how many events the case actually had - and no offset, so it could
# only ever see the OLDEST N. Answering "what happened" meant re-deriving a
# sequence from raw rows on every conversation, differently each time.


def test_case_timeline_returns_the_computed_narrative(tmp_path):
    out = case_timeline(_deps(tmp_path))

    assert "narrative" in out and out["narrative"]
    assert out["episode_count"] >= 1
    assert out["episodes"][0]["apps"] == ["Discord"]
    assert "summary" in out["episodes"][0]
    # coverage is the honest bound on what any account of the case can claim
    assert set(out["coverage"]) == {"total", "events", "expiry", "state", "event_pct"}


def test_case_timeline_event_count_is_the_true_total_not_the_page_size(tmp_path):
    deps = _deps(tmp_path)
    full = case_timeline(deps, relevant_only=False)
    paged = case_timeline(deps, relevant_only=False, limit=1)

    assert paged["events_shown"] == 1
    assert paged["event_count"] == full["event_count"] > 1


def test_case_timeline_events_carry_provenance(tmp_path):
    """Every other tool's rows are citable; these were not."""
    ev = case_timeline(_deps(tmp_path))["events"][0]
    assert ev["source_file"] and ev["source_locator"]
    assert "deleted" in ev
