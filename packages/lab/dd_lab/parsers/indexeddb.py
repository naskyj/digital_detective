"""IndexedDB (`<app>/indexeddb/*.indexeddb.leveldb`).

Chat apps (Discord, Slack, Element, …) keep message history, channel/user
directories and attachments metadata in IndexedDB, so this is usually the
richest source of CHAT_CONTENT / USER / GROUP_DETAIL artifacts. Each record's
value is typically an object and is flattened leaf-by-leaf.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
from collections.abc import Iterator

from dd_core.enums import SourceType
from dd_core.schema import NormalizedArtifact
from dd_lab.parsers.base import (
    INDEXEDDB,
    ParseContext,
    emit_parsed_value,
    emit_value,
    open_leveldb_resilient,
)

log = logging.getLogger("dd_lab.parsers.indexeddb")


def _leveldb_dirs(root):
    for child in sorted(root.iterdir()):
        if child.is_dir() and child.name.endswith(".leveldb"):
            blob = child.with_name(child.name.replace(".leveldb", ".blob"))
            yield child, (blob if blob.is_dir() else None)


def parse(ctx: ParseContext) -> Iterator[NormalizedArtifact]:
    if not ctx.has(INDEXEDDB):
        return
    from ccl_chromium_reader.ccl_chromium_indexeddb import WrappedIndexDB

    idb_root = ctx.path(INDEXEDDB)
    for leveldb_dir, blob_dir in _leveldb_dirs(idb_root):
        source_file = ctx.rel(leveldb_dir)
        widb, cleanup_dir = open_leveldb_resilient(
            leveldb_dir, lambda d, b=blob_dir: WrappedIndexDB(d, b)
        )
        if widb is None:
            continue
        try:
            for db_id in widb.database_ids:
                yield from _emit_database(ctx, widb, db_id, source_file)
        except Exception:
            log.exception("error walking IndexedDB %s", leveldb_dir)
        finally:
            with contextlib.suppress(Exception):
                widb.close()
            if cleanup_dir is not None:
                shutil.rmtree(cleanup_dir, ignore_errors=True)


def _emit_database(ctx, widb, db_id, source_file) -> Iterator[NormalizedArtifact]:
    try:
        db = widb[db_id.dbid_no] if hasattr(db_id, "dbid_no") else widb[db_id]
    except Exception:
        log.exception("cannot open IndexedDB database %r", db_id)
        return
    for store_name in db.object_store_names:
        try:
            store = db.get_object_store_by_name(store_name)
        except Exception:
            continue
        for rec in _safe_records(store):
            locator = f"{db.name}/{store_name}/{_key_repr(rec)}"
            value = getattr(rec, "value", None)
            if isinstance(value, (dict, list)):
                yield from emit_parsed_value(
                    ctx, source_type=SourceType.INDEXEDDB, source_file=source_file,
                    locator=locator, key=f"idb.{db.name}.{store_name}", parsed=value,
                )
            elif value is not None:
                yield from emit_value(
                    ctx, source_type=SourceType.INDEXEDDB, source_file=source_file,
                    locator=locator, key=f"idb.{db.name}.{store_name}", raw_value=str(value),
                )


def _safe_records(store):
    try:
        yield from store.iterate_records(live_only=False)
    except Exception:
        log.exception("error iterating object store %s", getattr(store, "name", "?"))


def _key_repr(rec) -> str:
    k = getattr(rec, "key", None)
    val = getattr(k, "value", k)
    return str(val)[:120]
