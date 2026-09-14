"""Parser plumbing shared by every source-specific parser.

A parser takes a `ParseContext` (which case, which app, where the evidence
lives) and yields `NormalizedArtifact` rows. Parsers never mutate evidence and
never raise on a single bad record — they log and continue, because forensic
input is routinely partial or corrupt.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from dd_core.enums import SourceType, ValueType
from dd_core.evidence import (
    CACHE,
    CODE_CACHE,
    COOKIES,
    COOKIES_LEGACY,
    INDEXEDDB,
    LOCAL_STATE,
    LOCAL_STORAGE,
    MEMORY,
    PREFERENCES,
    SERVICE_WORKER_CACHE,
    SESSION_STORAGE,
)
from dd_core.schema import NormalizedArtifact
from dd_lab.parsers.jsonflat import iter_leaves, try_parse_json

log = logging.getLogger("dd_lab.parsers")

__all__ = [
    "CACHE", "CODE_CACHE", "COOKIES", "COOKIES_LEGACY", "INDEXEDDB", "LOCAL_STATE",
    "LOCAL_STORAGE", "MEMORY", "PREFERENCES", "SERVICE_WORKER_CACHE", "SESSION_STORAGE",
    "ParseContext", "emit_value", "make_artifact", "open_leveldb_resilient",
]

# Values longer than this are truncated in the `value` column (full hash is kept).
MAX_VALUE_CHARS = 8192


@dataclass
class ParseContext:
    case_id: str
    app: str
    evidence_root: Path  # <case>/evidence
    app_root: Path = field(init=False)

    def __post_init__(self) -> None:
        self.evidence_root = Path(self.evidence_root)
        self.app_root = self.evidence_root / self.app

    def has(self, relative: str) -> bool:
        return (self.app_root / relative).exists()

    def path(self, relative: str) -> Path:
        return self.app_root / relative

    def rel(self, p: Path) -> str:
        """Path of an evidence file relative to <case>/evidence (for provenance)."""
        return Path(p).resolve().relative_to(self.evidence_root.resolve()).as_posix()


def _truncate(value: str) -> str:
    if len(value) > MAX_VALUE_CHARS:
        return value[:MAX_VALUE_CHARS] + f"…[+{len(value) - MAX_VALUE_CHARS} chars]"
    return value


def make_artifact(
    ctx: ParseContext,
    *,
    source_type: SourceType,
    source_file: str,
    locator: str,
    key: str,
    value: str,
    value_type: ValueType = ValueType.STRING,
    record_ts: datetime | None = None,
) -> NormalizedArtifact:
    return NormalizedArtifact(
        artifact_id=NormalizedArtifact.make_id(ctx.case_id, source_type, source_file, locator, key),
        case_id=ctx.case_id,
        app=ctx.app,
        source_type=source_type,
        source_file=source_file,
        source_locator=locator,
        key=key,
        value=_truncate(value),
        value_type=value_type,
        record_ts=record_ts,
        extracted_ts=datetime.now(UTC),
    )


def emit_value(
    ctx: ParseContext,
    *,
    source_type: SourceType,
    source_file: str,
    locator: str,
    key: str,
    raw_value: str,
    record_ts: datetime | None = None,
) -> Iterator[NormalizedArtifact]:
    """Emit one artifact for a plain value, or many if the value is a JSON document.

    This is the workhorse used by the storage parsers: a Local Storage value that
    happens to be JSON becomes one row per leaf, keyed `<key>.<leaf path>`.
    """
    parsed = try_parse_json(raw_value)
    if parsed is None:
        yield make_artifact(
            ctx, source_type=source_type, source_file=source_file, locator=locator,
            key=key, value=raw_value, record_ts=record_ts,
        )
        return
    produced = False
    for leaf_path, leaf_value, leaf_type in iter_leaves(parsed):
        produced = True
        full_key = f"{key}.{leaf_path}" if leaf_path and leaf_path != "$" else key
        yield make_artifact(
            ctx, source_type=source_type, source_file=source_file,
            locator=f"{locator}#{leaf_path}", key=full_key,
            value=leaf_value, value_type=leaf_type, record_ts=record_ts,
        )
    if not produced:
        yield make_artifact(
            ctx, source_type=source_type, source_file=source_file, locator=locator,
            key=key, value=raw_value, value_type=ValueType.JSON, record_ts=record_ts,
        )


def open_leveldb_resilient[T](
    leveldb_dir: Path, opener: Callable[[Path], T]
) -> tuple[T | None, Path | None]:
    """Open a LevelDB directory with `opener(leveldb_dir)`; on failure, retry
    against a copy with the `.log` (write-ahead log) file(s) dropped.

    Confirmed on real evidence (a live-acquired disk image), not hypothetical:
    a `.log` can end mid-write, or — very commonly — end with zero-byte block
    padding, and `ccl_chromium_reader`'s log-block reader treats an
    unrecognised block type as fatal (`ccl_leveldb.py::LogFile._get_batches`,
    `raise ValueError()  # Cannot happen` — it can, and did) which aborts
    opening the WHOLE store, not just the log. Retrying without the `.log`
    recovers everything already compacted into `.ldb`/`.sst` files — real data
    loss is then limited to whatever was still only in the WAL, instead of
    losing the entire store over one bad log file.

    Returns `(obj_or_None, cleanup_dir_or_None)` — the caller must
    `shutil.rmtree(cleanup_dir, ignore_errors=True)` after it's done with the
    object (once `cleanup_dir` is not None, the object's files live there, not
    at `leveldb_dir`).
    """
    try:
        return opener(leveldb_dir), None
    except Exception:
        log.warning(
            "could not open leveldb at %s; retrying without its .log file(s) "
            "(compacted data only)", leveldb_dir, exc_info=True,
        )

    tmp_dir = Path(tempfile.mkdtemp(prefix="dd_leveldb_"))
    copied_any = False
    for f in leveldb_dir.iterdir():
        if f.is_file() and f.suffix.lower() != ".log":
            shutil.copy2(f, tmp_dir / f.name)
            copied_any = True
    if not copied_any:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None, None
    try:
        return opener(tmp_dir), tmp_dir
    except Exception:
        log.exception("could not open leveldb at %s even without its .log file(s)", leveldb_dir)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None, None
