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


def test_only_unscored_processes_just_the_rows_missing_a_model_score(tmp_path):
    """Finishing an interrupted model run without redoing its completed
    work - on the real case that's 116,725 rows instead of 675,463."""
    scored_items = []

    class CountingVoter:
        def batch(self, items):
            scored_items.extend(k for k, _v, _st in items)
            return [0.8 for _ in items]

    wh = Warehouse(tmp_path / "warehouse" / "case.duckdb")
    try:
        wh.load_artifacts([_artifact("already", "v1"), _artifact("missing", "v2")])
        # Give exactly one row a stored score, as an interrupted run would have.
        wh.apply_classifications([
            ("a-already", {"category": "GENERAL", "confidence": 0.0, "relevance_model_score": 0.5}),
        ])

        classify_warehouse(wh, Classifier(relevance_voter=CountingVoter()), only_unscored=True)
        assert scored_items == ["missing"], "only the row without a stored score should be processed"
        # The already-scored row keeps its original score, untouched.
        rows = {r["key"]: r for r in wh.query("SELECT key, relevance_model_score FROM artifacts")}
        assert rows["already"]["relevance_model_score"] == 0.5
        assert rows["missing"]["relevance_model_score"] == 0.8
    finally:
        wh.close()


def test_only_unscored_terminates_even_when_a_row_never_gets_a_score(tmp_path):
    """The trap this pagination exists to avoid: rows the empty-body gate
    resolves never get a model score at all, so they stay NULL forever. A
    naive "take the first N unscored each time" loop would re-yield them
    endlessly. Keyset pagination on artifact_id always moves forward."""
    wh = Warehouse(tmp_path / "warehouse" / "case.duckdb")
    try:
        # "{}" is structurally empty -> the gate short-circuits before the
        # relevance model is ever consulted, so this row can never be scored.
        wh.load_artifacts([_artifact("empty", "{}"), _artifact("real", "actual content")])
        batches = list(wh.iter_for_classification(batch_size=1, only_unscored=True))
        assert len(batches) == 2, "must terminate, yielding each unscored row exactly once"

        # And the same through the real pipeline, which does write scores.
        result = classify_warehouse(
            wh, Classifier(relevance_voter=lambda k, v, st: 0.9), only_unscored=True
        )
        assert result.classified == 2
        still_null = wh.query(
            "SELECT count(*) AS n FROM artifacts WHERE relevance_model_score IS NULL"
        )[0]["n"]
        assert still_null == 1, "the gated row legitimately stays unscored - and that's fine"
    finally:
        wh.close()


def test_exclude_source_types_skips_them_but_leaves_them_classified(tmp_path):
    """Relevance-model inference is too slow on CPU to run over a whole
    large case (memory_string alone was ~547k of 675k on the real case), so
    a model pass can be scoped away from a source type. The excluded rows
    must still end up fully classified - just with rules-only relevance."""
    seen = []

    class CountingVoter:
        def batch(self, items):
            seen.extend(st for _k, _v, st in items)
            return [0.9 for _ in items]

    wh = Warehouse(tmp_path / "warehouse" / "case.duckdb")
    try:
        wh.load_artifacts([
            _artifact("k1", "some cache value", source_type=SourceType.CACHE),
            _artifact("k2", "a password is hunter2", source_type=SourceType.MEMORY_STRING),
        ])
        # The real sequence: everything is classified first (rules-only, fast),
        # THEN an expensive model pass runs scoped to the sources worth it.
        classify_warehouse(wh, Classifier())
        before = {r["source_type"]: r["category"] for r in wh.query(
            "SELECT source_type, category FROM artifacts")}
        assert before["memory_string"] is not None

        result = classify_warehouse(
            wh, Classifier(relevance_voter=CountingVoter()),
            exclude_source_types=("memory_string",),
        )
        assert seen == ["cache"], "memory_string must never reach the model"
        assert result.classified == 1  # only the non-excluded row was re-processed

        rows = {r["source_type"]: r for r in wh.query(
            "SELECT source_type, category, relevant, relevance_model_score FROM artifacts")}
        assert rows["cache"]["relevance_model_score"] == 0.9
        assert rows["memory_string"]["relevance_model_score"] is None
        # The excluded row KEEPS its earlier classification - skipped by the
        # MODEL, not wiped or left unclassified.
        assert rows["memory_string"]["category"] == before["memory_string"]
        assert rows["memory_string"]["relevant"] is not None
    finally:
        wh.close()


def test_source_types_processes_only_those_leaving_everything_else_untouched(tmp_path):
    """The inverse of exclude_source_types: run a long model pass over ONE
    huge source type in the background without rewriting anything else.
    Rows outside the filter must not be selected at all - not re-scored,
    not re-classified, not touched."""
    seen = []

    class CountingVoter:
        def batch(self, items):
            seen.extend(st for _k, _v, st in items)
            return [0.9 for _ in items]

    wh = Warehouse(tmp_path / "warehouse" / "case.duckdb")
    try:
        wh.load_artifacts([
            _artifact("k1", "some cache value", source_type=SourceType.CACHE),
            _artifact("k2", "a carved memory string", source_type=SourceType.MEMORY_STRING),
        ])
        # Give the cache row a score, as a prior scoped pass would have.
        wh.apply_classifications([
            ("a-k1", {"category": "GENERAL", "confidence": 0.1, "relevance_model_score": 0.42}),
        ])

        result = classify_warehouse(
            wh, Classifier(relevance_voter=CountingVoter()),
            only_unscored=True, source_types=("memory_string",),
        )
        assert seen == ["memory_string"], "only the named source type may reach the model"
        assert result.classified == 1

        rows = {r["source_type"]: r for r in wh.query(
            "SELECT source_type, relevance_model_score FROM artifacts")}
        assert rows["memory_string"]["relevance_model_score"] == 0.9
        # Untouched - still exactly what the earlier pass stored.
        assert rows["cache"]["relevance_model_score"] == 0.42
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
