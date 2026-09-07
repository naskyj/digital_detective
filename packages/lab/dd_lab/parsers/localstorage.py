"""Local Storage (LevelDB at `<app>/Local Storage/leveldb`)."""

from __future__ import annotations

import logging
from collections.abc import Iterator

from dd_core.enums import SourceType
from dd_core.schema import NormalizedArtifact
from dd_lab.parsers.base import LOCAL_STORAGE, ParseContext, emit_value

log = logging.getLogger("dd_lab.parsers.localstorage")


def parse(ctx: ParseContext) -> Iterator[NormalizedArtifact]:
    if not ctx.has(LOCAL_STORAGE):
        return
    from ccl_chromium_reader.ccl_chromium_localstorage import LocalStoreDb

    leveldb_dir = ctx.path(LOCAL_STORAGE)
    source_file = ctx.rel(leveldb_dir)
    try:
        db = LocalStoreDb(leveldb_dir)
    except Exception:
        log.exception("could not open Local Storage leveldb at %s", leveldb_dir)
        return
    try:
        for rec in db.iter_all_records(include_deletions=True):
            try:
                locator = f"{rec.storage_key}\x1f{rec.script_key}\x1fseq={rec.leveldb_seq_number}"
                key = f"{rec.storage_key} :: {rec.script_key}"
                yield from emit_value(
                    ctx,
                    source_type=SourceType.LOCAL_STORAGE,
                    source_file=source_file,
                    locator=locator,
                    key=key,
                    raw_value=rec.value if rec.value is not None else "",
                )
            except Exception:
                log.exception("bad Local Storage record in %s", leveldb_dir)
    finally:
        db.close()
