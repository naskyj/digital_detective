"""The relevance score saturates, so ordering by it alone leaves large ties.

`rule_evidence` is capped at 1.0 and the highest-weighted voter reaches the
cap unaided, so noisy-OR returns exactly `1 - (1 - 1.0)(1 - x) = 1.0` for ANY
model probability x. Measured on the 557,834-row annotated corpus, 2,559
artifacts share the score 1.0 and only 48% of them are relevant, while the
model probability separates that block almost perfectly (median 1.000 for the
relevant ones against 0.000 for the rest). Average precision over the whole
corpus is 0.638 ordering by the composed score and 0.930 once the model
probability breaks its ties - against 0.941 for the model score alone.

Every LIMITed query over relevant artifacts therefore decides *which* rows an
examiner sees, not merely their order: `v_high_priority` feeds the report's
high-priority section at LIMIT 100. These tests pin the tie-break at each
ordering site, because losing it would silently restore arbitrary ordering -
no error, no failing confusion matrix, just a worse list.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from dd_core.enums import ArtifactCategory as C
from dd_core.enums import SourceType as S
from dd_core.schema import ClassificationResult, NormalizedArtifact
from dd_lab.etl.warehouse import Warehouse

_SATURATED = 1.0


def _artifact(aid: str, model_score: float, *, confidence: float = 0.5):
    """An artifact pinned at the saturated relevance score, differing from its
    siblings only in the model probability — the population the tie-break has
    to order."""
    return NormalizedArtifact(
        artifact_id=aid, case_id="c1", app="discord", source_type=S.CACHE,
        source_file="discord/Cache/f", source_locator=f"loc-{aid}",
        key="cache.body.content", value=f"value for {aid}",
        record_ts=None, extracted_ts=datetime(2026, 9, 15, tzinfo=UTC),
        classification=ClassificationResult(
            category=C.CHAT_CONTENT, confidence=confidence,
            classifier_version="test", relevant=True,
            relevance_score=_SATURATED, relevance_model_score=model_score),
    )


@pytest.fixture
def wh(tmp_path):
    w = Warehouse(tmp_path / "case.duckdb")
    # Category confidence runs OPPOSITE to the model probability, and the rows
    # go in worst-first. Both matter: the ordering these tests replaced was
    # `relevance_score DESC, confidence DESC`, so an inverse confidence makes
    # a lost tie-break produce exactly the reverse of the right answer rather
    # than passing on insertion order. Equal confidences let the first version
    # of this test pass against the unfixed code.
    w.load_artifacts([
        _artifact("low", 0.01, confidence=0.9),
        _artifact("mid", 0.50, confidence=0.5),
        _artifact("high", 0.99, confidence=0.1),
    ])
    w.build_views()
    yield w
    w.close()


def test_v_high_priority_breaks_saturated_ties_by_model_score(wh):
    """The report's high-priority section reads this view at LIMIT 100, so on
    a real case the tie-break decides which artifacts reach the examiner."""
    order = [r["artifact_id"] for r in wh.query("SELECT * FROM v_high_priority")]
    assert order == ["high", "mid", "low"]


def test_v_high_priority_still_orders_by_relevance_score_first(wh):
    """The tie-break must only act *within* a tie: a genuinely lower relevance
    score outranks a high model probability, or this would stop being a
    tie-break and start being a different classifier."""
    wh.load_artifacts([_artifact("higher-score", 0.0, confidence=0.2)])
    wh.query("UPDATE artifacts SET relevance_score = 1.5 "
             "WHERE artifact_id = 'higher-score'")
    order = [r["artifact_id"] for r in wh.query("SELECT * FROM v_high_priority")]
    assert order[0] == "higher-score", "tie-break overrode the primary key"
    assert order[1:] == ["high", "mid", "low"]


def test_a_null_model_score_sorts_last_not_first(wh):
    """No model configured is a supported deployment (requirement R1), so the
    column is nullable. NULLS LAST keeps those rows from displacing scored
    ones at the top of the examiner's list."""
    wh.load_artifacts([_artifact("unscored", 0.0, confidence=0.95)])
    wh.query("UPDATE artifacts SET relevance_model_score = NULL "
             "WHERE artifact_id = 'unscored'")
    order = [r["artifact_id"] for r in wh.query("SELECT * FROM v_high_priority")]
    assert order == ["high", "mid", "low", "unscored"]


@pytest.mark.parametrize("sort", ["", "relevance"])
def test_artifact_browser_sorts_break_ties_by_model_score(wh, sort):
    """Both relevance orderings in the API, which back the artifact browser
    and the CSV export."""
    from dd_lab.api.main import _artifact_order_by

    rows = wh.query(
        f"SELECT artifact_id FROM artifacts ORDER BY {_artifact_order_by(sort)}")
    assert [r["artifact_id"] for r in rows] == ["high", "mid", "low"]


def test_confidence_sort_is_untouched_by_the_tie_break(wh):
    """Sorting by category confidence is a different question and must not
    silently acquire a relevance tie-break."""
    from dd_lab.api.main import _artifact_order_by

    assert "relevance_model_score" not in _artifact_order_by("confidence")
