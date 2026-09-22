"""classify_warehouse()'s re-classify path (classifier_backend_spec.md phase
1): iter_for_classification() was widened to SELECT app/source_type/deleted
alongside key/value/value_type, so re-classifying an already-ingested
warehouse reaches the same Classifier.classify() signature the ingest-time
classify_stream() path already gets for free from the NormalizedArtifact
it's holding. This is a real gap that would otherwise only show up once a
later phase actually branches on app/source_type/deleted - catch it now,
at the plumbing stage, while it's still a trivial fix."""

from __future__ import annotations

from datetime import UTC, datetime

from dd_core.enums import SourceType, ValueType
from dd_core.schema import NormalizedArtifact
from dd_lab.classify.ensemble import Classifier
from dd_lab.classify.pipeline import classify_warehouse
from dd_lab.etl.warehouse import Warehouse


def _artifact(key: str, value: str, *, app="discord", source_type=SourceType.CACHE, deleted=False):
    return NormalizedArtifact(
        artifact_id=f"a-{key}",
        case_id="c1",
        app=app,
        source_type=source_type,
        source_file="f",
        source_locator=key,
        key=key,
        value=value,
        value_type=ValueType.STRING,
        extracted_ts=datetime.now(UTC),
        deleted=deleted,
    )


def test_classify_warehouse_reads_app_source_type_deleted_without_erroring(tmp_path):
    wh = Warehouse(tmp_path / "warehouse" / "case.duckdb")
    try:
        wh.load_artifacts(
            [
                _artifact("cache.body.auth.token", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.x.y"),
                _artifact("local_storage.app.password", "hunter2!", deleted=True),
            ]
        )
        result = classify_warehouse(wh)
        assert result.classified == 2
        rows = wh.query("SELECT category, relevant FROM artifacts ORDER BY artifact_id")
        assert len(rows) == 2
        assert all(r["category"] for r in rows)
    finally:
        wh.close()


def test_classify_warehouse_round_trips_relevance_model_score(tmp_path):
    """relevance_model_score (the trained relevance model's own raw
    P(relevant), a new column alongside relevant/relevance_score) must
    survive the full classify_warehouse round trip - apply_classifications'
    hand-written temp-table/INSERT/UPDATE column lists are exactly the kind
    of place a new field silently gets dropped if not threaded through
    everywhere."""

    def fake_relevance(key, value, source_type):
        return 0.77

    wh = Warehouse(tmp_path / "warehouse" / "case.duckdb")
    try:
        wh.load_artifacts([_artifact("k", "v")])
        classify_warehouse(wh, Classifier(relevance_voter=fake_relevance))
        rows = wh.query("SELECT relevance_model_score, relevant, relevance_score FROM artifacts")
        assert len(rows) == 1
        assert rows[0]["relevance_model_score"] == 0.77
        assert rows[0]["relevant"] is True
        assert rows[0]["relevance_score"] == 0.77
    finally:
        wh.close()


def test_reuse_model_scores_skips_the_model_but_keeps_its_stored_score(tmp_path):
    """The expensive part of a re-classify is the relevance model (~27 hours
    of a measured ~27.5-hour run on the real 675k-artifact case). Its output
    depends only on artifact text, which doesn't change - so re-deriving
    relevance after a COMBINATION-LOGIC change must be able to reuse the
    stored score instead of paying that cost again."""
    calls = []

    class CountingVoter:
        def batch(self, items):
            calls.append(len(items))
            return [0.9 for _ in items]

    wh = Warehouse(tmp_path / "warehouse" / "case.duckdb")
    try:
        wh.load_artifacts([_artifact("k", "v")])
        # First pass: the model actually runs and its score is persisted.
        classify_warehouse(wh, Classifier(relevance_voter=CountingVoter()))
        assert calls == [1]
        stored = wh.query("SELECT relevance_model_score FROM artifacts")[0]
        assert stored["relevance_model_score"] == 0.9

        # Second pass with reuse: the model is NOT called again, but the
        # stored score still informs the result.
        calls.clear()
        classify_warehouse(
            wh, Classifier(relevance_voter=CountingVoter()), reuse_model_scores=True
        )
        assert calls == [], "the relevance model must not run when reusing stored scores"
        after = wh.query("SELECT relevance_model_score, relevant FROM artifacts")[0]
        assert after["relevance_model_score"] == 0.9  # preserved, not wiped
        assert after["relevant"] is True
    finally:
        wh.close()


def test_iter_for_classification_yields_app_source_type_deleted(tmp_path):
    wh = Warehouse(tmp_path / "warehouse" / "case.duckdb")
    try:
        wh.load_artifacts([_artifact("k", "v", app="slack", source_type=SourceType.INDEXEDDB, deleted=True)])
        batches = list(wh.iter_for_classification())
        assert len(batches) == 1
        row = batches[0][0]
        assert row["app"] == "slack"
        assert row["source_type"] == "indexeddb"
        assert row["deleted"] is True
    finally:
        wh.close()
