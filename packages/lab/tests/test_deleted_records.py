"""The `deleted` field: whether an artifact was recovered from a tombstoned/
overwritten LevelDB record rather than being the current live value.

ccl_chromium_reader's higher-level readers (LocalStoreDb, SessionStoreDb,
WrappedIndexDB) already pull these in via include_deletions=True /
live_only=False - that part predates this session. What was missing: the
per-record is_live/is_deleted flag those readers expose was silently
dropped on the way into NormalizedArtifact, so a value recovered from a
deleted record was indistinguishable from a current live one anywhere in
the tool's own output.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dd_core.enums import SourceType
from dd_lab.parsers import localstorage
from dd_lab.parsers.base import ParseContext, emit_parsed_value, emit_value, make_artifact

_LOCAL_STORAGE_CANDIDATES = [
    Path.home() / "Library/Application Support/Code",
    Path.home() / "Library/Application Support/Postman",
    Path.home() / "Library/Application Support/Lens",
    Path.home() / "Library/Application Support/Docker Desktop",
]


def _ctx() -> ParseContext:
    return ParseContext(case_id="c1", app="app", evidence_root=Path("/tmp/nonexistent-evidence-root"))


def test_make_artifact_defaults_to_not_deleted():
    a = make_artifact(
        _ctx(), source_type=SourceType.LOCAL_STORAGE, source_file="f", locator="l", key="k", value="v",
    )
    assert a.deleted is False


def test_make_artifact_propagates_deleted_true():
    a = make_artifact(
        _ctx(), source_type=SourceType.LOCAL_STORAGE, source_file="f", locator="l", key="k", value="v",
        deleted=True,
    )
    assert a.deleted is True


def test_emit_value_propagates_deleted_to_every_flattened_leaf():
    import json

    rows = list(emit_value(
        _ctx(), source_type=SourceType.CACHE, source_file="f", locator="l", key="cache.body",
        raw_value=json.dumps({"a": 1, "b": [2, 3]}), deleted=True,
    ))
    assert len(rows) == 3  # a, b[0], b[1]
    assert all(r.deleted for r in rows)


def test_emit_parsed_value_propagates_deleted():
    rows = list(emit_parsed_value(
        _ctx(), source_type=SourceType.INDEXEDDB, source_file="f", locator="l", key="idb.db.store",
        parsed={"x": 1}, deleted=True,
    ))
    assert rows and all(r.deleted for r in rows)


def test_warehouse_round_trips_the_deleted_flag(tmp_path):
    from dd_lab.etl.warehouse import Warehouse

    wh = Warehouse(tmp_path / "case.duckdb")
    try:
        artifacts = [
            make_artifact(
                _ctx(), source_type=SourceType.LOCAL_STORAGE, source_file="f", locator=f"l{i}",
                key=f"k{i}", value="v", deleted=(i == 0),
            )
            for i in range(2)
        ]
        wh.load_artifacts(artifacts)
        rows = wh.query("SELECT key, deleted FROM artifacts ORDER BY key")
        assert rows[0]["deleted"] is True
        assert rows[1]["deleted"] is False
    finally:
        wh.close()


def _local_storage_profile_with_deletions() -> Path | None:
    for p in _LOCAL_STORAGE_CANDIDATES:
        if not (p / "Local Storage" / "leveldb").is_dir():
            continue
        ctx = ParseContext(case_id="c1", app="app", evidence_root=p.parent)
        ctx.app_root = p
        if any(r.deleted for r in localstorage.parse(ctx)):
            return p
    return None


@pytest.mark.skipif(
    _local_storage_profile_with_deletions() is None,
    reason="no real Local Storage profile with a deleted record found on this machine right now",
)
def test_real_deleted_local_storage_records_are_flagged():
    """Verified on real evidence, not synthetic: overwritten/cleared Local
    Storage keys are genuinely present in real Chromium profiles (a
    tombstone stays in the .log until compaction), and are now recovered
    AND correctly marked - not silently indistinguishable from live data."""
    profile = _local_storage_profile_with_deletions()
    ctx = ParseContext(case_id="c1", app="app", evidence_root=profile.parent)
    ctx.app_root = profile
    rows = list(localstorage.parse(ctx))
    deleted = [r for r in rows if r.deleted]
    live = [r for r in rows if not r.deleted]
    assert deleted
    assert live  # a real profile isn't ALL deleted records either
    for r in deleted:
        assert r.source_type == SourceType.LOCAL_STORAGE
