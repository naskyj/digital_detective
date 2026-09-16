"""Raw LevelDB physical-file browsing.

Independent of the higher-level per-store readers (`ccl_chromium_reader`'s
`LocalStoreDb` etc.) the ingest pipeline uses to build normalised artifacts —
this reads a LevelDB store's `.ldb`/`.sst`/`.log` files directly, one physical
file at a time, via `ccl_chromium_reader`'s own low-level `ccl_leveldb`
module. Lets an examiner see exactly what one physical file actually holds,
the same kind of raw view a LevelDB inspection tool gives, and names the
file every record it lists really came from.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel

log = logging.getLogger("dd_lab.parsers.leveldb_browse")

_EXTENSIONS = (".ldb", ".sst", ".log")


class LeveldbFileEntry(BaseModel):
    filename: str
    kind: str  # "ldb" | "log"
    size_bytes: int
    record_count: int | None = None  # None if the file couldn't be opened at all
    truncated: bool = False  # a parse error was hit partway through (counted so far)


class LeveldbRecord(BaseModel):
    key: str
    value: str
    state: str  # "live" | "deleted" | "unknown"
    seq: int


def _open_raw(f: Path, kind: str):
    from ccl_chromium_reader.storage_formats import ccl_leveldb as raw

    return raw.LogFile(f) if kind == "log" else raw.LdbFile(f)


def _kind_of(f: Path) -> str | None:
    suffix = f.suffix.lower()
    if suffix not in _EXTENSIONS:
        return None
    return "log" if suffix == ".log" else "ldb"


def list_leveldb_files(store_dir: Path) -> list[LeveldbFileEntry]:
    """Every `.ldb`/`.sst`/`.log` file in a LevelDB store directory, each with
    a record count read directly from that one file (never the merged,
    cross-file logical view the normal parsers produce)."""
    if not store_dir.is_dir():
        return []
    out: list[LeveldbFileEntry] = []
    for f in sorted(store_dir.iterdir()):
        kind = _kind_of(f)
        if kind is None or not f.is_file():
            continue
        n, truncated = _count(f, kind)
        out.append(LeveldbFileEntry(
            filename=f.name, kind=kind, size_bytes=f.stat().st_size,
            record_count=n, truncated=truncated,
        ))
    return out


def _count(f: Path, kind: str) -> tuple[int | None, bool]:
    try:
        handle = _open_raw(f, kind)
    except Exception:
        log.debug("could not open %s for counting", f)
        return None, False
    n, truncated = 0, False
    try:
        for _rec in handle:
            n += 1
    except Exception:
        # A .log file ending mid-write or with trailing zero-padding is
        # routine on live-acquired evidence (the same case
        # open_leveldb_resilient works around at the whole-store level) -
        # here it just means this one file's listing stops early rather
        # than the whole browser failing.
        truncated = True
    finally:
        handle.close()
    return n, truncated


def read_leveldb_file(
    store_dir: Path, filename: str, *, limit: int = 200, offset: int = 0
) -> tuple[list[LeveldbRecord], int, bool]:
    """(records_page, total, truncated) for one physical file's records."""
    f = store_dir / filename
    kind = _kind_of(f)
    if kind is None or not f.is_file():
        return [], 0, False
    try:
        handle = _open_raw(f, kind)
    except Exception:
        log.debug("could not open %s", f)
        return [], 0, False
    records: list[LeveldbRecord] = []
    truncated = False
    try:
        for rec in handle:
            key = rec.user_key if kind == "ldb" else rec.key
            records.append(LeveldbRecord(
                key=_decode(key), value=_decode(rec.value),
                state=rec.state.name.lower(), seq=rec.seq,
            ))
    except Exception:
        truncated = True
    finally:
        handle.close()
    total = len(records)
    return records[offset : offset + limit], total, truncated


def _decode(b: bytes) -> str:
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.hex()
