"""Raw SQLite database-file browsing.

Independent of the named parsers that read specific SQLite files for specific
purposes (`cookies.py`) - this lists every table in a database file, with row
counts, and reads a table's raw rows directly. Currently applies to the one
SQLite-backed evidence file the layout defines (`Network/Cookies` or legacy
`Cookies`); written generically so another database file can be pointed at it
later without changes here.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from pydantic import BaseModel

log = logging.getLogger("dd_lab.parsers.sqlite_browse")


class SqliteTableInfo(BaseModel):
    table: str
    row_count: int


class SqliteFileEntry(BaseModel):
    filename: str
    size_bytes: int
    tables: list[SqliteTableInfo]


def _connect(db_path: Path) -> sqlite3.Connection:
    # Same convention as cookies.py: immutable=1, since this is a static
    # evidence copy that will never change under us.
    return sqlite3.connect(f"file:{db_path}?immutable=1", uri=True)


def _table_names(conn: sqlite3.Connection) -> list[str]:
    return [
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    ]


def list_sqlite_tables(db_path: Path) -> SqliteFileEntry | None:
    """None if the file doesn't exist or isn't a readable SQLite database -
    never raises, this is a best-effort browsing aid."""
    if not db_path.is_file():
        return None
    try:
        conn = _connect(db_path)
    except sqlite3.Error:
        log.debug("could not open %s as sqlite", db_path)
        return None
    try:
        tables = []
        for t in _table_names(conn):
            try:
                n = conn.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0]
            except sqlite3.Error:
                n = 0
            tables.append(SqliteTableInfo(table=t, row_count=n))
        return SqliteFileEntry(filename=db_path.name, size_bytes=db_path.stat().st_size, tables=tables)
    except sqlite3.Error:
        log.exception("could not list tables in %s", db_path)
        return None
    finally:
        conn.close()


def read_sqlite_table(
    db_path: Path, table: str, *, limit: int = 200, offset: int = 0
) -> tuple[list[str], list[list[str]], int]:
    """(columns, rows, total) for one table. `table` is checked against the
    database's own table list before being interpolated into SQL - sqlite3
    can't parameterise an identifier, so this is the injection guard."""
    conn = _connect(db_path)
    try:
        if table not in set(_table_names(conn)):
            return [], [], 0
        total = conn.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]
        cur = conn.execute(f'SELECT * FROM "{table}" LIMIT ? OFFSET ?', (limit, offset))
        columns = [d[0] for d in cur.description]
        rows = [[_stringify(v) for v in row] for row in cur.fetchall()]
        return columns, rows, total
    finally:
        conn.close()


def _stringify(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, bytes):
        try:
            return v.decode("utf-8")
        except UnicodeDecodeError:
            return v.hex()
    return str(v)
