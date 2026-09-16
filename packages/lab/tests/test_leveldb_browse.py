"""Raw LevelDB physical-file browsing (dd_lab/parsers/leveldb_browse.py)
against a real Local Storage store on the dev machine. Skips cleanly if none
is available."""

from __future__ import annotations

from pathlib import Path

import pytest

from dd_lab.parsers.leveldb_browse import list_leveldb_files, read_leveldb_file

_CANDIDATES = [
    Path.home() / "Library/Application Support/Code/Local Storage/leveldb",
    Path.home() / "Library/Application Support/Postman/Local Storage/leveldb",
    Path.home() / ".config/Code/Local Storage/leveldb",
]


def _store() -> Path | None:
    for p in _CANDIDATES:
        if p.is_dir() and any(p.iterdir()):
            return p
    return None


pytestmark = pytest.mark.skipif(_store() is None, reason="no real Local Storage leveldb store available")


def test_list_leveldb_files_finds_real_physical_files():
    files = list_leveldb_files(_store())
    assert files
    for f in files:
        assert f.filename
        assert f.kind in ("ldb", "log")
        assert f.size_bytes > 0


def test_read_leveldb_file_returns_real_records_with_state():
    store = _store()
    files = list_leveldb_files(store)
    target = next((f for f in files if f.record_count), files[0])
    records, total, truncated = read_leveldb_file(store, target.filename, limit=10, offset=0)
    assert total == target.record_count
    assert records
    for r in records:
        assert r.state in ("live", "deleted", "unknown")
        assert isinstance(r.seq, int)


def test_read_leveldb_file_pagination_matches_total():
    store = _store()
    files = list_leveldb_files(store)
    target = next((f for f in files if (f.record_count or 0) >= 3), None)
    if target is None:
        pytest.skip("no store file with enough records to test pagination")
    page1, total, _ = read_leveldb_file(store, target.filename, limit=1, offset=0)
    page2, _, _ = read_leveldb_file(store, target.filename, limit=1, offset=1)
    assert len(page1) == 1
    assert len(page2) == 1
    assert page1[0].seq != page2[0].seq


def test_unknown_file_returns_empty_not_an_error():
    store = _store()
    records, total, truncated = read_leveldb_file(store, "999999.ldb", limit=10, offset=0)
    assert records == []
    assert total == 0
    assert truncated is False


def test_list_on_missing_directory_returns_empty():
    assert list_leveldb_files(Path("/nonexistent-store-dir")) == []
