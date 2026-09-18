"""Local Storage (LevelDB at `<app>/Local Storage/leveldb`)."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Iterator
from pathlib import Path

from dd_core.enums import SourceType
from dd_core.schema import NormalizedArtifact
from dd_lab.parsers.base import LOCAL_STORAGE, ParseContext, emit_value, open_leveldb_resilient

log = logging.getLogger("dd_lab.parsers.localstorage")


def parse(ctx: ParseContext) -> Iterator[NormalizedArtifact]:
    if not ctx.has(LOCAL_STORAGE):
        return
    from ccl_chromium_reader.ccl_chromium_localstorage import LocalStoreDb

    leveldb_dir = ctx.path(LOCAL_STORAGE)
    source_dir = ctx.rel(leveldb_dir)
    db, cleanup_dir = open_leveldb_resilient(leveldb_dir, LocalStoreDb)
    if db is None:
        return
    try:
        for rec in db.iter_all_records(include_deletions=True):
            try:
                locator = f"{rec.storage_key}\x1f{rec.script_key}\x1fseq={rec.leveldb_seq_number}"
                key = f"{rec.storage_key} :: {rec.script_key}"
                # rec.file is the PHYSICAL .ldb/.log file this record was
                # read from (a key isn't pinned to one file - compaction
                # moves it between .ldb files over time, same fact
                # /localstorage-files exposes for browsing). It's an
                # absolute path (into a resilient-open tmp copy when that
                # path was taken - see open_leveldb_resilient - which
                # preserves the original filename, just not the original
                # directory), so only its basename is evidence-safe to
                # record; ctx.rel(leveldb_dir) is what keeps this relative
                # to <case>/evidence/, same as every other source_file.
                source_file = f"{source_dir}/{Path(rec.file).name}"
                yield from emit_value(
                    ctx,
                    source_type=SourceType.LOCAL_STORAGE,
                    source_file=source_file,
                    locator=locator,
                    key=key,
                    raw_value=rec.value if rec.value is not None else "",
                    deleted=not rec.is_live,
                )
            except Exception:
                log.exception("bad Local Storage record in %s", leveldb_dir)
    finally:
        db.close()
        if cleanup_dir is not None:
            shutil.rmtree(cleanup_dir, ignore_errors=True)
