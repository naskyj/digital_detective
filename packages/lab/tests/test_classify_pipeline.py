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
