"""Examiner review of the artifacts the classifier did not flag.

The classifier's recall is 0.958 on the annotated corpus, so about one
relevant artifact in 24 is never surfaced. Those misses are not reachable by
lowering the threshold: the composed score is near-bimodal and 45 of the 54
sit at exactly 0.0 among 551,809 others, so "show everything below 0.5" is
553,727 artifacts — the problem the tool exists to solve, restated. The
model's own probability does rank that population; reviewing the top 50
recovers 21 of the 54 misses, 42% of what is read.

So the queue is a fixed budget off a model-score ordering, and these tests
pin the three properties that make applying its results to evidence safe:
the classifier's own decision survives the override, a re-classification does
not revoke the examiner, and every decision reaches the custody chain.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from dd_core.custody import CustodyLog
from dd_core.enums import ArtifactCategory as C
from dd_core.enums import CustodyEvent
from dd_core.enums import SourceType as S
from dd_core.schema import ClassificationResult, NormalizedArtifact
from dd_lab.etl.warehouse import Warehouse


def _artifact(aid: str, *, relevant: bool, model_score: float,
              relevance_score: float = 0.0):
    return NormalizedArtifact(
        artifact_id=aid, case_id="c1", app="discord", source_type=S.MEMORY_STRING,
        source_file="memdump.raw", source_locator=f"pid1234:{aid}",
        key="mem.string", value=f"value for {aid}",
        record_ts=None, extracted_ts=datetime(2026, 9, 15, tzinfo=UTC),
        classification=ClassificationResult(
            category=C.GENERAL, confidence=0.0, classifier_version="test",
            relevant=relevant, relevance_score=relevance_score,
            relevance_model_score=model_score),
    )


@pytest.fixture
def wh(tmp_path):
    w = Warehouse(tmp_path / "case.duckdb")
    w.load_artifacts([
        # Flagged already — must never appear in the queue.
        _artifact("flagged", relevant=True, model_score=0.99, relevance_score=1.0),
        # Unflagged, with the model ordering them. All three sit at composed
        # score 0.0, which is exactly why the composed score cannot order them.
        _artifact("miss-likely", relevant=False, model_score=0.93),
        _artifact("miss-maybe", relevant=False, model_score=0.40),
        _artifact("noise", relevant=False, model_score=0.000001),
    ])
    w.build_views()
    yield w
    w.close()


def test_queue_holds_only_unflagged_artifacts(wh):
    ids = [r["artifact_id"] for r in wh.review_queue()]
    assert "flagged" not in ids
    assert set(ids) == {"miss-likely", "miss-maybe", "noise"}


def test_queue_is_ordered_by_model_probability_not_the_composed_score(wh):
    """All three rows share composed score 0.0, so an ordering that used it
    would be arbitrary. This is the whole basis of the feature."""
    assert [r["artifact_id"] for r in wh.review_queue()] == [
        "miss-likely", "miss-maybe", "noise"]
    assert {r["relevance_score"] for r in wh.review_queue()} == {0.0}


def test_a_budget_takes_the_best_candidates(wh):
    assert [r["artifact_id"] for r in wh.review_queue(limit=2)] == [
        "miss-likely", "miss-maybe"]


def test_approving_flips_relevant_so_the_artifact_reaches_the_report(wh):
    wh.record_review("miss-likely", "approved", "examiner-1")
    row = wh.query("SELECT relevant FROM artifacts WHERE artifact_id = 'miss-likely'")[0]
    assert row["relevant"] is True
    assert "miss-likely" in [r["artifact_id"]
                             for r in wh.query("SELECT * FROM v_high_priority")]


def test_the_classifiers_own_decision_survives_the_override(wh):
    """R3: the system discards nothing. The override has to be visible AS an
    override, and the classifier must stay independently scoreable afterwards
    — otherwise every accuracy figure silently becomes a measurement of the
    classifier plus the examiner."""
    wh.record_review("miss-likely", "approved", "examiner-1")
    row = wh.query("SELECT relevant, relevant_auto, review_decision, reviewed_by "
                   "FROM artifacts WHERE artifact_id = 'miss-likely'")[0]
    assert row["relevant"] is True
    assert row["relevant_auto"] is False
    assert row["review_decision"] == "approved"
    assert row["reviewed_by"] == "examiner-1"


def test_rejecting_leaves_the_artifact_unflagged_but_records_the_decision(wh):
    """A rejection is not a no-op: it takes the artifact out of the queue, so
    the examiner never re-reads it, and records that a human looked."""
    wh.record_review("noise", "rejected", "examiner-1")
    row = wh.query("SELECT relevant, review_decision FROM artifacts "
                   "WHERE artifact_id = 'noise'")[0]
    assert row["relevant"] is False
    assert row["review_decision"] == "rejected"
    assert "noise" not in [r["artifact_id"] for r in wh.review_queue()]


def test_reviewed_artifacts_leave_the_queue_but_remain_retrievable(wh):
    wh.record_review("miss-maybe", "rejected", "examiner-1")
    assert "miss-maybe" not in [r["artifact_id"] for r in wh.review_queue()]
    assert "miss-maybe" in [r["artifact_id"]
                            for r in wh.review_queue(include_reviewed=True)]


def test_re_reviewing_does_not_overwrite_the_original_automatic_decision(wh):
    """An examiner who approves then changes their mind must not leave
    `relevant_auto` holding their own earlier override."""
    wh.record_review("miss-likely", "approved", "examiner-1")
    wh.record_review("miss-likely", "rejected", "examiner-2")
    row = wh.query("SELECT relevant, relevant_auto, review_decision, reviewed_by "
                   "FROM artifacts WHERE artifact_id = 'miss-likely'")[0]
    assert row["relevant_auto"] is False, "the ORIGINAL classifier decision"
    assert row["relevant"] is False
    assert row["review_decision"] == "rejected"
    assert row["reviewed_by"] == "examiner-2"


def test_reclassification_does_not_revoke_an_examiners_decision(wh):
    """Re-classifying is routine — it happens whenever a rule changes. Without
    this, every /reclassify would silently undo the entire review queue, and
    nothing would report that it had."""
    wh.record_review("miss-likely", "approved", "examiner-1")
    wh.apply_classifications([
        ("miss-likely", {"category": "GENERAL", "confidence": 0.0,
                         "relevant": False, "relevance_score": 0.0,
                         "relevance_model_score": 0.93,
                         "classifier_version": "test-v2"}),
    ])
    row = wh.query("SELECT relevant, relevant_auto, review_decision, classifier_version "
                   "FROM artifacts WHERE artifact_id = 'miss-likely'")[0]
    assert row["relevant"] is True, "re-classification revoked a human decision"
    assert row["relevant_auto"] is False, "the fresh automatic decision is still recorded"
    assert row["classifier_version"] == "test-v2", "the rest of the row did update"


def test_reclassification_still_updates_an_unreviewed_artifact(wh):
    """The guard above must be narrow: it protects reviewed rows only."""
    wh.apply_classifications([
        ("miss-maybe", {"category": "CREDENTIAL", "confidence": 0.9,
                        "relevant": True, "relevance_score": 1.0,
                        "relevance_model_score": 0.4,
                        "classifier_version": "test-v2"}),
    ])
    row = wh.query("SELECT relevant, relevant_auto FROM artifacts "
                   "WHERE artifact_id = 'miss-maybe'")[0]
    assert row["relevant"] is True
    assert row["relevant_auto"] is True


def test_counts_track_progress(wh):
    wh.record_review("miss-likely", "approved", "e")
    wh.record_review("noise", "rejected", "e")
    c = wh.review_counts()
    assert c["pending"] == 1 and c["approved"] == 1 and c["rejected"] == 1
    assert c["relevant"] == 2  # the originally-flagged one, plus the approval


def test_unknown_artifact_and_unknown_decision_are_rejected(wh):
    with pytest.raises(KeyError):
        wh.record_review("no-such-artifact", "approved", "e")
    with pytest.raises(ValueError, match="decision must be one of"):
        wh.record_review("noise", "maybe", "e")


# ------------------------------------------------------------------- API


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import dd_lab.api.main as main
    from dd_core.casedir import CaseLayout

    root = tmp_path / "cases"
    layout = CaseLayout(root, "case-1")
    layout.create_evidence_skeleton()
    layout.create_lab_skeleton()
    layout.manifest.write_text("{}", encoding="utf-8")
    w = Warehouse(layout.warehouse)
    w.load_artifacts([
        _artifact("flagged", relevant=True, model_score=0.99, relevance_score=1.0),
        _artifact("miss-likely", relevant=False, model_score=0.93),
        _artifact("noise", relevant=False, model_score=0.000001),
    ])
    w.build_views()
    w.close()
    monkeypatch.setattr(main._settings, "cases_root", str(root))
    return TestClient(main.app), layout


def test_api_queue_excludes_flagged_and_orders_by_model_score(client):
    c, _ = client
    r = c.get("/api/cases/case-1/review-queue?limit=10")
    assert r.status_code == 200
    body = r.json()
    assert [row["artifact_id"] for row in body["rows"]] == ["miss-likely", "noise"]
    assert body["counts"]["pending"] == 2


def test_api_approval_flips_relevant_and_writes_the_custody_chain(client):
    """R2: a human overriding an automated decision about evidence has to be
    attributable. The chain must still verify afterwards."""
    c, layout = client
    r = c.post("/api/cases/case-1/review",
               json={"artifact_id": "miss-likely", "decision": "approved",
                     "reviewed_by": "examiner-1", "note": "session token"})
    assert r.status_code == 200, r.text
    assert r.json()["artifact"]["relevant"] is True

    entries = [e for e in CustodyLog(layout.dir).entries()
               if e.event == CustodyEvent.REVIEW_DECISION]
    assert len(entries) == 1
    assert entries[0].actor == "examiner-1"
    assert entries[0].data["artifact_id"] == "miss-likely"
    assert entries[0].data["decision"] == "approved"
    # The automated decision being overridden is in the log entry itself, so
    # it stands without the warehouse.
    assert entries[0].data["relevant_auto"] is False
    ok, bad = CustodyLog(layout.dir).verify_chain()
    assert ok, f"custody chain broken at {bad}"


def test_api_rejects_an_unknown_artifact_and_a_bad_decision(client):
    c, _ = client
    assert c.post("/api/cases/case-1/review",
                  json={"artifact_id": "nope", "decision": "approved"}
                  ).status_code == 404
    assert c.post("/api/cases/case-1/review",
                  json={"artifact_id": "noise", "decision": "sure-why-not"}
                  ).status_code == 400


def test_api_unknown_case_is_404(client):
    c, _ = client
    assert c.get("/api/cases/nope/review-queue").status_code == 404
