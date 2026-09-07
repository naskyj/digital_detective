"""Parser plumbing shared by every source-specific parser.

A parser takes a `ParseContext` (which case, which app, where the evidence
lives) and yields `NormalizedArtifact` rows. Parsers never mutate evidence and
never raise on a single bad record — they log and continue, because forensic
input is routinely partial or corrupt.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
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
    "ParseContext", "emit_value", "make_artifact",
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
