"""Session Storage (LevelDB at `<app>/Session Storage`)."""

from __future__ import annotations

import logging
from collections.abc import Iterator

from dd_core.enums import SourceType
from dd_core.schema import NormalizedArtifact
from dd_lab.parsers.base import SESSION_STORAGE, ParseContext, emit_value

log = logging.getLogger("dd_lab.parsers.sessionstorage")


def parse(ctx: ParseContext) -> Iterator[NormalizedArtifact]:
    if not ctx.has(SESSION_STORAGE):
        return
    from ccl_chromium_reader.ccl_chromium_sessionstorage import SessionStoreDb

    ss_dir = ctx.path(SESSION_STORAGE)
    source_file = ctx.rel(ss_dir)
    try:
        db = SessionStoreDb(ss_dir)
    except Exception:
        log.exception("could not open Session Storage at %s", ss_dir)
        return
    try:
        for rec in db.iter_all_records(include_deletions=True, include_orphans=True):
            try:
                host = rec.host or "?"
                locator = f"{host}\x1f{rec.key}\x1fseq={rec.leveldb_sequence_number}"
                yield from emit_value(
                    ctx,
                    source_type=SourceType.SESSION_STORAGE,
                    source_file=source_file,
                    locator=locator,
                    key=f"{host} :: {rec.key}",
                    raw_value=rec.value if rec.value is not None else "",
                )
            except Exception:
                log.exception("bad Session Storage record in %s", ss_dir)
    finally:
        db.close()
