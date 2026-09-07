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

from dd_core.enums import SourceType, ValueType
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


def _emit_entry(ctx: ParseContext, cache, key: str, source_file: str) -> Iterator[NormalizedArtifact]:
    metas = cache.get_metadata(key)
    bodies = cache.get_cachefile(key)
    meta = next((m for m in metas if m), None)

    url = key.split("/", 2)[-1] if key.count("/") >= 2 else key
    request_time: datetime | None = getattr(meta, "request_time", None) if meta else None

    yield make_artifact(
        ctx, source_type=SourceType.CACHE, source_file=source_file,
        locator=key, key="cache.url", value=url, value_type=ValueType.URL,
        record_ts=request_time,
    )

    if meta is not None:
        status = _first(meta.http_header_declarations)
        if status:
            yield make_artifact(
                ctx, source_type=SourceType.CACHE, source_file=source_file,
                locator=key, key="cache.http_status", value=str(status), record_ts=request_time,
            )
        ct = (meta.get_attribute("content-type") or [""])[0]
        if ct:
            yield make_artifact(
                ctx, source_type=SourceType.CACHE, source_file=source_file,
                locator=key, key="cache.content_type", value=ct, record_ts=request_time,
            )
        times = (("request_time", request_time), ("response_time", getattr(meta, "response_time", None)))
        for label, ts in times:
            if ts:
                yield make_artifact(
                    ctx, source_type=SourceType.CACHE, source_file=source_file,
                    locator=key, key=f"cache.{label}", value=ts.isoformat(),
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
            ctx, source_type=SourceType.CACHE, source_file=source_file,
            locator=f"{key}::body", key="cache.body_sample",
            value=text[:4000], value_type=ValueType.STRING, record_ts=request_time,
        )
        return
    yield from emit_value(
        ctx, source_type=SourceType.CACHE, source_file=source_file,
        locator=f"{key}::body", key="cache.body", raw_value=text, record_ts=request_time,
    )
