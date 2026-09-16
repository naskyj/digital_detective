"""Raw SQLite database-file browsing (dd_lab/parsers/sqlite_browse.py)
against a real Cookies database on the dev machine. Skips cleanly if none is
available."""

from __future__ import annotations

from pathlib import Path

import pytest

from dd_lab.parsers.sqlite_browse import list_sqlite_tables, read_sqlite_table

_CANDIDATES = [
    Path.home() / "Library/Application Support/Cursor/Cookies",
    Path.home() / "Library/Application Support/Code/Cookies",
    Path.home() / "Library/Application Support/Antigravity/Cookies",
]


def _db() -> Path | None:
    for p in _CANDIDATES:
        if p.is_file():
            return p
    return None


pytestmark = pytest.mark.skipif(_db() is None, reason="no real Cookies sqlite database available")


def test_list_sqlite_tables_finds_real_tables():
    entry = list_sqlite_tables(_db())
    assert entry is not None
    assert entry.filename == "Cookies"
    assert entry.tables
    assert any(t.table == "meta" for t in entry.tables)


def test_read_sqlite_table_returns_real_rows():
    entry = list_sqlite_tables(_db())
    table = entry.tables[0].table
    columns, rows, total = read_sqlite_table(_db(), table, limit=5, offset=0)
    assert columns
    assert total == entry.tables[0].row_count
    assert len(rows) == min(5, total)
    if rows:
        assert len(rows[0]) == len(columns)


def test_unknown_table_name_returns_empty_not_sql_injection():
    columns, rows, total = read_sqlite_table(_db(), 'x"; DROP TABLE meta; --', limit=5, offset=0)
    assert columns == []
    assert rows == []
    assert total == 0
    # confirm the real table is still there and untouched
    entry = list_sqlite_tables(_db())
    assert any(t.table == "meta" for t in entry.tables)


def test_missing_file_returns_none():
    assert list_sqlite_tables(Path("/nonexistent/Cookies")) is None
