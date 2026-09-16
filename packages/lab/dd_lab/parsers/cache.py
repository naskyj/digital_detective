"""Chromium caches — the HTTP disk cache plus the Service Worker CacheStorage
(the Cache API) and the compiled-code cache.

Chat apps fetch message payloads, user directories and API JSON here — the disk
cache holds ordinary responses, and the Cache API (`caches.open(...)`) holds
what the app deliberately caches for offline use, which for Discord/Slack is a
lot. All three are Chromium simple/block-file caches, so one reader handles them.
JSON bodies are flattened leaf-by-leaf.
"""

from __future__ import annotations

import gzip
import logging
import zlib
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from dd_core.enums import SourceType, ValueType
from dd_core.hashing import stable_id
from dd_core.schema import NormalizedArtifact
from dd_lab.parsers.base import (
    CACHE,
    SERVICE_WORKER_CACHE,
    ParseContext,
    emit_value,
    make_artifact,
)

log = logging.getLogger("dd_lab.parsers.cache")

_TEXTUAL_CT = ("application/json", "text/", "application/javascript", "application/xml", "+json")

# Response bodies larger than this are sampled, not flattened leaf-by-leaf.
_MAX_FLATTEN_BYTES = 512_000


def _decompress(data: bytes, content_encoding: str) -> bytes:
    enc = (content_encoding or "").strip().lower()
    try:
        if enc == "gzip":
            return gzip.decompress(data)
        if enc == "br":
            import brotli

            return brotli.decompress(data)
        if enc == "deflate":
            return zlib.decompress(data, -zlib.MAX_WBITS)
        if enc == "zstd":
            import zstd

            return zstd.decompress(data)
    except Exception:
        log.debug("cache body decompression failed (encoding=%s)", enc)
    return data


def _iter_cache_dirs(ctx: ParseContext) -> Iterator[Path]:
    """Every directory under the app's evidence that is a Chromium HTTP cache.

    The compiled-code cache (`Code Cache/`) is preserved but not parsed — it
    holds V8 bytecode, no user data, and its metadata format trips up readers.
    """
    root = ctx.path(CACHE)
    if root.is_dir():
        yield (root / "Cache_Data") if (root / "Cache_Data").is_dir() else root
    # Service Worker CacheStorage — the Cache API. <origin>/<cache-name>/ leaves.
    sw = ctx.path(SERVICE_WORKER_CACHE)
    if sw.is_dir():
        for leaf in sorted(sw.rglob("*")):
            if leaf.is_dir() and (leaf / "index").is_file():
                yield leaf


def parse(ctx: ParseContext) -> Iterator[NormalizedArtifact]:
    from ccl_chromium_reader import ccl_chromium_cache as cc

    seen: set[str] = set()
    for cache_dir in _iter_cache_dirs(ctx):
        rp = str(cache_dir.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        source_file = ctx.rel(cache_dir)
        klass = cc.guess_cache_class(cache_dir)
        if klass is None:
            log.debug("not a recognised cache: %s", cache_dir)
            continue
        try:
            cache = klass(cache_dir)
        except Exception:
            log.exception("could not open cache at %s", cache_dir)
            continue
        try:
            for key in cache.keys():  # noqa: SIM118  (ChromiumCache, not a dict)
                try:
                    yield from _emit_entry(ctx, cache, key, source_file)
                except Exception:
                    log.debug("skipped unreadable cache entry %r in %s", key, cache_dir)
        finally:
            cache.close()


def _first(gen) -> str | None:
    try:
        return next(iter(gen or []), None)
    except Exception:
        return None


def url_from_cache_key(key: str) -> str:
    """A Chromium cache key is typically `<request-context>/<sub-key>/<url>`
    (e.g. "1/0/https://discord.com/api/..."); this pulls the URL back out.
    Falls back to the raw key when it doesn't look like that shape - some
    Cache API keys are bare URLs already. Shared by every place that shows a
    cache entry's URL (the parser, the cache-file inventory, and the
    artifact-browser API's derived `url` column for cache rows)."""
    return key.split("/", 2)[-1] if key.count("/") >= 2 else key


def _location(cache, key: str, stream: str):
    """The physical on-disk file (+ byte offset) backing one cache entry's
    metadata or body stream — the same fact a tool like ChromeCacheView shows
    as its "Filename" column, read via ccl_chromium_reader's own location
    API rather than re-derived. `stream` is "meta" or "body". None if the
    library can't resolve a location for this key (never fatal)."""
    try:
        getter = cache.get_location_for_metadata if stream == "meta" else cache.get_location_for_cachefile
        return _first(getter(key))
    except Exception:
        return None


def _emit_entry(ctx: ParseContext, cache, key: str, source_file: str) -> Iterator[NormalizedArtifact]:
    metas = cache.get_metadata(key)
    bodies = cache.get_cachefile(key)
    meta = next((m for m in metas if m), None)

    # The physical file (and, packed into the locator, its byte offset) that
    # actually holds each stream — metadata and body can genuinely live in
    # different physical files for the same key in the block-file format, so
    # each is resolved separately rather than assumed to match.
    meta_loc = _location(cache, key, "meta")
    body_loc = _location(cache, key, "body")
    meta_file = f"{source_file}/{meta_loc.file_name}" if meta_loc else source_file
    body_file = f"{source_file}/{body_loc.file_name}" if body_loc else source_file
    meta_locator = f"{key} @{meta_loc.file_name}:{meta_loc.offset}" if meta_loc else key
    body_locator = f"{key} @{body_loc.file_name}:{body_loc.offset}" if body_loc else key

    url = url_from_cache_key(key)
    request_time: datetime | None = getattr(meta, "request_time", None) if meta else None

    yield make_artifact(
        ctx, source_type=SourceType.CACHE, source_file=meta_file,
        locator=meta_locator, key="cache.url", value=url, value_type=ValueType.URL,
        record_ts=request_time,
    )

    if meta is not None:
        status = _first(meta.http_header_declarations)
        if status:
            yield make_artifact(
                ctx, source_type=SourceType.CACHE, source_file=meta_file,
                locator=meta_locator, key="cache.http_status", value=str(status), record_ts=request_time,
            )
        ct = (meta.get_attribute("content-type") or [""])[0]
        if ct:
            yield make_artifact(
                ctx, source_type=SourceType.CACHE, source_file=meta_file,
                locator=meta_locator, key="cache.content_type", value=ct, record_ts=request_time,
            )
        times = (("request_time", request_time), ("response_time", getattr(meta, "response_time", None)))
        for label, ts in times:
            if ts:
                yield make_artifact(
                    ctx, source_type=SourceType.CACHE, source_file=meta_file,
                    locator=meta_locator, key=f"cache.{label}", value=ts.isoformat(),
                    value_type=ValueType.TIMESTAMP, record_ts=ts,
                )
        content_encoding = (meta.get_attribute("content-encoding") or [""])[0]
    else:
        ct = ""
        content_encoding = ""

    body = next((b for b in (bodies or []) if b), None)
    if not body:
        return
    if content_encoding:
        body = _decompress(body, content_encoding)
    if not any(t in ct.lower() for t in _TEXTUAL_CT):
        return
    try:
        text = body.decode("utf-8", errors="replace")
    except Exception:
        return
    if len(body) > _MAX_FLATTEN_BYTES:
        yield make_artifact(
            ctx, source_type=SourceType.CACHE, source_file=body_file,
            locator=f"{body_locator}::body", key="cache.body_sample",
            value=text[:4000], value_type=ValueType.STRING, record_ts=request_time,
        )
        return
    yield from emit_value(
        ctx, source_type=SourceType.CACHE, source_file=body_file,
        locator=f"{body_locator}::body", key="cache.body", raw_value=text, record_ts=request_time,
    )


# --------------------------------------------------------------------- cache-file inventory
#
# A ChromeCacheView-style listing: one row per cache entry, naming the exact
# physical file and byte offset behind it, without reading/decoding bodies.
# Independent of the ingest pipeline — it reads straight off the evidence
# directory, so it works whether or not "Run analysis" has happened yet.


class CacheFileEntry(BaseModel):
    entry_id: str          # stable, URL-safe identifier for the "open" endpoint
    app: str
    cache_kind: str         # "http_cache" | "cache_api"
    cache_dir: str          # evidence-relative directory, e.g. "discord/Cache/Cache_Data"
    key: str                # the raw Chromium cache key
    physical_file: str | None = None   # e.g. "data_1", "f_00012a", or a simple-cache filename
    offset: int | None = None
    url: str
    http_status: str | None = None
    content_type: str | None = None
    content_encoding: str | None = None
    cache_control: str | None = None
    server: str | None = None
    size_bytes: int | None = None
    request_time: datetime | None = None
    response_time: datetime | None = None
    openable: bool = False   # a body stream location was resolvable


def _entry_size(cache, key: str) -> int | None:
    """Stream-1 (body) size without reading the body itself."""
    get_info = getattr(cache, "get_entry_info", None)
    if get_info is not None:  # simple-file cache: computed at index-build time
        try:
            infos = get_info(key)
            return infos[0].data_size if infos else None
        except Exception:
            return None
    try:  # block-file cache: EntryStore carries per-stream sizes directly
        return cache[key].data_sizes[1]
    except Exception:
        return None


def _describe_entry(ctx: ParseContext, cache, key: str, cache_dir_rel: str, kind: str) -> CacheFileEntry | None:
    try:
        meta = next((m for m in cache.get_metadata(key) if m), None)
    except Exception:
        meta = None
    body_loc = _location(cache, key, "body")
    url = url_from_cache_key(key)

    def attr(name: str) -> str | None:
        return (meta.get_attribute(name) or [None])[0] if meta else None

    return CacheFileEntry(
        entry_id=stable_id(ctx.case_id, ctx.app, cache_dir_rel, key),
        app=ctx.app, cache_kind=kind, cache_dir=cache_dir_rel, key=key,
        physical_file=body_loc.file_name if body_loc else None,
        offset=body_loc.offset if body_loc else None,
        url=url,
        http_status=_first(meta.http_header_declarations) if meta else None,
        content_type=attr("content-type"),
        content_encoding=attr("content-encoding"),
        cache_control=attr("cache-control"),
        server=attr("server"),
        size_bytes=_entry_size(cache, key),
        request_time=getattr(meta, "request_time", None) if meta else None,
        response_time=getattr(meta, "response_time", None) if meta else None,
        openable=body_loc is not None,
    )


def cache_dirs(ctx: ParseContext) -> list[str]:
    """Every recognised cache directory for this app, evidence-relative — the
    valid `cache_dir` values for `read_cache_entry` (used to validate the API's
    `dir` query param before opening anything)."""
    seen: set[str] = set()
    out: list[str] = []
    for d in _iter_cache_dirs(ctx):
        rp = str(d.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        out.append(ctx.rel(d))
    return out


def list_cache_files(ctx: ParseContext) -> Iterator[CacheFileEntry]:
    """Enumerate every cache entry across every cache directory for this app —
    the HTTP disk cache and the Cache API alike — as one row each. Metadata
    only; bodies are read on demand by `read_cache_entry`, not here."""
    from ccl_chromium_reader import ccl_chromium_cache as cc

    seen: set[str] = set()
    for cache_dir in _iter_cache_dirs(ctx):
        rp = str(cache_dir.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        cache_dir_rel = ctx.rel(cache_dir)
        kind = "cache_api" if "Service Worker" in cache_dir_rel else "http_cache"
        klass = cc.guess_cache_class(cache_dir)
        if klass is None:
            continue
        try:
            cache = klass(cache_dir)
        except Exception:
            log.exception("could not open cache at %s", cache_dir)
            continue
        try:
            for key in cache.keys():  # noqa: SIM118  (ChromiumCache, not a dict)
                try:
                    entry = _describe_entry(ctx, cache, key, cache_dir_rel, kind)
                    if entry:
                        yield entry
                except Exception:
                    log.debug("skipped unreadable cache entry %r in %s", key, cache_dir)
        finally:
            cache.close()


def read_cache_entry(ctx: ParseContext, cache_dir_rel: str, key: str) -> tuple[bytes | None, str, str]:
    """The decoded body of one cache entry, for "open"/preview in the UI.
    Returns (body_bytes | None, content_type, filename_hint). Re-opens the
    cache fresh (no handle held between calls) — this is a read-only, on-
    demand lookup, not part of the ingest pipeline."""
    from ccl_chromium_reader import ccl_chromium_cache as cc

    cache_dir = (ctx.evidence_root / cache_dir_rel)
    if not cache_dir.is_dir():
        return None, "application/octet-stream", "cache_entry"
    klass = cc.guess_cache_class(cache_dir)
    if klass is None:
        return None, "application/octet-stream", "cache_entry"
    try:
        cache = klass(cache_dir)
    except Exception:
        log.exception("could not open cache at %s", cache_dir)
        return None, "application/octet-stream", "cache_entry"
    try:
        meta = next((m for m in cache.get_metadata(key) if m), None)
        ct = ((meta.get_attribute("content-type") or [""])[0] if meta else "") or "application/octet-stream"
        enc = (meta.get_attribute("content-encoding") or [""])[0] if meta else ""
        bodies = cache.get_cachefile(key)
        # `if b` (truthy) would treat a genuinely empty body (b"" - a real,
        # valid 0-byte response, e.g. a 304/204) as "no body" and 404 an
        # entry the listing correctly marked openable. `is not None` is the
        # right check - an empty body is still a body.
        body = next((b for b in (bodies or []) if b is not None), None)
        if body is None:
            return None, ct, "cache_entry"
        if enc:
            body = _decompress(body, enc)
        url = url_from_cache_key(key)
        filename_hint = (url.rsplit("/", 1)[-1] or "cache_entry").split("?", 1)[0][:120] or "cache_entry"
        return body, ct, filename_hint
    except Exception:
        log.debug("could not read cache entry %r in %s", key, cache_dir)
        return None, "application/octet-stream", "cache_entry"
    finally:
        cache.close()
