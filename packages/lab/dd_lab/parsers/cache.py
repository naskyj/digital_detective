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
import json
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
from dd_lab.parsers.jsonflat import try_parse_json

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


def url_filename_hint(url: str) -> str:
    """A human-readable filename guessed from a cache entry's URL (its last
    path segment, query string dropped, capped at 120 chars) - NOT the
    physical on-disk cache file (that's `source_file`/the byte offset
    resolved by `_location`; a completely different, unrelated fact - one
    is where Chromium physically stored the bytes, this is what the
    original resource would have been called). Originally inline in
    `read_cache_entry` (the "Open"/download filename); shared here so the
    labelling export's `filename` column derives it the same way instead of
    a second, possibly-drifting reimplementation."""
    return (url.rsplit("/", 1)[-1] or "cache_entry").split("?", 1)[0][:120] or "cache_entry"


def raw_key_from_locator(locator: str) -> str:
    """Recover the original Chromium cache key from ANY of the locator
    shapes `_emit_entry` produces: bare `key`, `key @file:offset`,
    `key @file:offset::body`, or `...::body#leaf.path` (and the
    no-location-resolved variants of the last two, `key::body` /
    `key::body#leaf.path`, when `_location` couldn't resolve a physical
    file). Strips exactly the suffixes `_emit_entry` adds, in the order it
    adds them - "#leaf" first (only body rows with a JSON value have it),
    then "::body" (only body rows have it at all), then " @file:offset"
    (only when `_location` resolved one). Splitting on " @" ALONE (an
    earlier, simpler approach) silently keeps a trailing "::body#leaf" any
    time location resolution fails, corrupting the recovered key - the
    order here is what avoids that. The one place every reader of a cache
    row's locator should go through, rather than reimplementing this
    per-caller (the artifact-browser API's derived `url` column and the
    labelling export's per-entry grouping both need exactly this)."""
    s = locator
    if "#" in s:
        s = s.split("#", 1)[0]
    if s.endswith("::body"):
        s = s[: -len("::body")]
    if " @" in s:
        s = s.split(" @", 1)[0]
    return s


def cache_offset_from_locator(locator: str) -> int | None:
    """The byte offset backing THIS row's own stream (meta or body) - the
    number after the physical filename in ` @file:offset`. None when
    `_location` couldn't resolve a physical location for this stream
    (never fatal there, so not an error here either)."""
    if " @" not in locator:
        return None
    after = locator.split(" @", 1)[1]  # "file:offset[::body[#leaf.path]]"
    file_and_offset = after.split("::", 1)[0]
    if ":" not in file_and_offset:
        return None
    try:
        return int(file_and_offset.rsplit(":", 1)[-1])
    except ValueError:
        return None


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


def _entry_metadata(ctx: ParseContext, cache, key: str, cache_dir_rel: str, kind: str) -> dict:
    """Everything about one cache entry EXCEPT its body - the single source
    of truth for both the live Cache Files inventory (CacheFileEntry, below)
    and the labelling export (CacheLabelRow) so the two column sets can
    never independently drift on how the same fact gets computed. Keys
    match CacheFileEntry's own field names exactly, so `CacheFileEntry(**d)`
    and `CacheLabelRow(**d, ...)` both unpack it directly."""
    try:
        meta = next((m for m in cache.get_metadata(key) if m), None)
    except Exception:
        meta = None
    body_loc = _location(cache, key, "body")
    url = url_from_cache_key(key)

    def attr(name: str) -> str | None:
        return (meta.get_attribute(name) or [None])[0] if meta else None

    return {
        "entry_id": stable_id(ctx.case_id, ctx.app, cache_dir_rel, key),
        "app": ctx.app, "cache_kind": kind, "cache_dir": cache_dir_rel, "key": key,
        "physical_file": body_loc.file_name if body_loc else None,
        "offset": body_loc.offset if body_loc else None,
        "url": url,
        "http_status": _first(meta.http_header_declarations) if meta else None,
        "content_type": attr("content-type"),
        "content_encoding": attr("content-encoding"),
        "cache_control": attr("cache-control"),
        "server": attr("server"),
        "size_bytes": _entry_size(cache, key),
        "request_time": getattr(meta, "request_time", None) if meta else None,
        "response_time": getattr(meta, "response_time", None) if meta else None,
    }


def _describe_entry(
    ctx: ParseContext, cache, key: str, cache_dir_rel: str, kind: str
) -> CacheFileEntry | None:
    m = _entry_metadata(ctx, cache, key, cache_dir_rel, kind)
    return CacheFileEntry(
        **m,
        openable=m["physical_file"] is not None,
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
        return body, ct, url_filename_hint(url)
    except Exception:
        log.debug("could not read cache entry %r in %s", key, cache_dir)
        return None, "application/octet-stream", "cache_entry"
    finally:
        cache.close()


# --------------------------------------------------------------------- labelling export
#
# One row per physical cache entry (or, for a JSON array-of-objects body, one
# row per element) - NOT one row per metadata field. `parse()`/`_emit_entry`
# above feeds the main warehouse, where url/http_status/content_type/
# request_time/response_time are each their own NormalizedArtifact (correct
# there - the artifact browser, findings, correlation and the classifier all
# want one fact per row). A hand-labelling/content-matching export is a
# different consumer with a different unit of analysis: a person (or a
# partner reconciling against a raw ChromeCacheView dump) thinks in terms of
# "this cached file", not "this file's request_time considered on its own" -
# treating those 5-6 fields as 5-6 separate "artifacts" inflated a real
# export ~7x over the true entry count (41,405 rows for 5,946 real entries,
# confirmed against a partner's independent raw-dump reconciliation) and
# scattered one entry's own facts across unrelated rows instead of keeping
# them together as columns of the same row.
#
# The body gets its own, narrower splitting rule than jsonflat.py's general
# one (which explodes ANY array, including a plain array of strings/numbers,
# and recurses into nested arrays) - see `_classify_body_for_labeling`: only
# a top-level JSON array whose elements are ALL objects is exploded, one row
# per element; a JSON object (however deeply nested inside), a primitive/
# mixed/empty array, non-JSON text, and binary content all stay exactly one
# row. This function reads bodies fresh from the evidence directory (like
# `list_cache_files`, independent of the ingest pipeline and its warehouse)
# rather than reconstructing shape from already-flattened warehouse rows,
# which would be ambiguous or outright lossy for exactly the cases this
# rule cares about getting right.

# JSON bodies (object/array/array-element) are capped generously - content-
# matching wants the real structure, well above MAX_VALUE_CHARS in base.py
# (tuned for the warehouse's own `value` column, a tighter row-size budget).
# Non-JSON TEXT bodies (JS/CSS/HTML/plain text) get a much smaller sample -
# a partner reconciling against a raw cache dump needs enough to confirm
# content, not an inlined copy of a whole minified bundle; the physical
# file itself (physical_file/offset/cache_dir) is the actual evidence.
_JSON_BODY_CHARS = 20_000
_TEXT_SAMPLE_CHARS = 2_000


class CacheLabelRow(BaseModel):
    """One row per physical cache entry (or one row per element of a JSON
    array-of-objects body) - column set is CacheFileEntry's own (see
    _entry_metadata) plus `value` (the decoded body) and the columns needed
    to describe how `value` relates to the whole body. `openable` is
    deliberately dropped - a live-browser navigation concept, meaningless
    in a static export."""

    entry_id: str          # stable id, unique per (case, app, cache_dir, key) - NOT per exported row
    app: str
    cache_kind: str         # "http_cache" | "cache_api"
    cache_dir: str          # evidence-relative directory
    key: str                # the raw Chromium cache key
    physical_file: str | None = None   # e.g. "data_1" - the file this entry's body lives in
    offset: int | None = None
    url: str
    filename: str           # human-readable, guessed from the URL (see url_filename_hint)
    http_status: str | None = None
    content_type: str | None = None
    content_encoding: str | None = None
    cache_control: str | None = None
    server: str | None = None
    size_bytes: int | None = None
    request_time: datetime | None = None
    response_time: datetime | None = None
    # Always False today - cache entries have no deletion-tracking the way
    # LevelDB tombstones do (a rotating disk cache just evicts, it doesn't
    # leave a recoverable record). Column kept for schema parity with the
    # rest of the export and as the landing spot for real detection later.
    deleted: bool = False
    body_kind: str           # "empty" | "binary" | "text" | "json_object" | "json_array" | "json_array_element"
    array_index: int | None = None   # set only when body_kind == "json_array_element"
    value: str = ""


def _cap_body(text: str, limit: int) -> str:
    if len(text) > limit:
        return text[:limit] + f"…[+{len(text) - limit} chars]"
    return text


def _flatten_newlines(text: str) -> str:
    """A raw JS/CSS/HTML/plain-text body sample can contain literal
    newlines - a properly RFC4180-quoted CSV field handles that fine, but a
    naive line-counting check (`wc -l`) or a non-compliant viewer doesn't,
    and will report a wildly inflated "row count" that has nothing to do
    with the actual number of CSV records (confirmed on a real export: a
    correctly-parsed 6,863-row file counted as 10,730+ raw lines). JSON
    bodies never hit this - json.dumps already escapes embedded newlines
    into a literal two-character "\\n" itself. Escaping here the same way,
    rather than collapsing to a space, keeps the sample byte-faithful
    (recoverable, and not silently merging two distinct tokens together)
    for a partner doing exact content matching against the raw dump."""
    return text.replace("\r\n", "\\n").replace("\r", "\\n").replace("\n", "\\n")


def _classify_body_for_labeling(text: str) -> tuple[str, list[str]]:
    """(body_kind, [row values]) for one decoded response body - one value
    unless body_kind is "json_array_element", in which case there's one per
    array entry, in order. See the module-level comment above for the exact
    rule; `try_parse_json` (jsonflat.py) is the same JSON-or-not sniff/parse
    the rest of the system uses, so "is this JSON" never drifts between the
    two rules that both apply it to a cache body."""
    parsed = try_parse_json(text)
    if parsed is None:
        return "text", [text]
    if isinstance(parsed, dict):
        return "json_object", [json.dumps(parsed, default=str)]
    # try_parse_json only ever returns a dict or a list (it requires the
    # first non-whitespace character to be "{" or "[") - a bare top-level
    # JSON scalar never reaches here, so this is the only remaining case.
    if parsed and all(isinstance(e, dict) for e in parsed):
        return "json_array_element", [json.dumps(e, default=str) for e in parsed]
    return "json_array", [json.dumps(parsed, default=str)]


def iter_cache_entries_for_labeling(ctx: ParseContext) -> Iterator[CacheLabelRow]:
    """Every cache entry for this app, across every cache directory (the HTTP
    disk cache and the Cache API alike) - one row each, or one row per
    element for a JSON array-of-objects body. See the module comment above."""
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
                    yield from _describe_entry_for_labeling(ctx, cache, key, cache_dir_rel, kind)
                except Exception:
                    log.debug("skipped unreadable cache entry %r in %s", key, cache_dir)
        finally:
            cache.close()


def _binary_placeholder(content_type: str, size: int) -> str:
    return f"[binary {content_type or 'unknown'}, {size} bytes]"


def _describe_entry_for_labeling(
    ctx: ParseContext, cache, key: str, cache_dir_rel: str, kind: str
) -> Iterator[CacheLabelRow]:
    m = _entry_metadata(ctx, cache, key, cache_dir_rel, kind)
    content_encoding = m["content_encoding"] or ""
    content_type = m["content_type"] or ""
    common = dict(m, filename=url_filename_hint(m["url"]))

    bodies = cache.get_cachefile(key)
    # `is not None`, not truthiness - a genuinely empty body (b"", a real
    # 304/204 response) is still a body, not a missing one (see
    # test_read_cache_entry_empty_body_is_a_real_empty_response_not_missing).
    body = next((b for b in (bodies or []) if b is not None), None)
    if body is None:
        yield CacheLabelRow(**common, body_kind="empty")
        return
    if content_encoding:
        body = _decompress(body, content_encoding)
    if not body:
        yield CacheLabelRow(**common, body_kind="empty")
        return
    if not any(t in content_type.lower() for t in _TEXTUAL_CT):
        # a real, non-empty entry (image/video/audio/font/wasm/...) - still
        # counted as one artifact (the physical file IS the evidence, even
        # without its bytes inlined here), just with a short placeholder
        # instead of raw bytes in a text column.
        yield CacheLabelRow(**common, body_kind="binary", value=_binary_placeholder(content_type, len(body)))
        return
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        yield CacheLabelRow(**common, body_kind="binary", value=_binary_placeholder(content_type, len(body)))
        return

    body_kind, values = _classify_body_for_labeling(text)
    if body_kind == "json_array_element":
        for i, v in enumerate(values):
            yield CacheLabelRow(**common, body_kind=body_kind, array_index=i, value=_cap_body(v, _JSON_BODY_CHARS))
    elif body_kind == "text":
        sample = _flatten_newlines(_cap_body(values[0], _TEXT_SAMPLE_CHARS))
        yield CacheLabelRow(**common, body_kind=body_kind, value=sample)
    else:
        yield CacheLabelRow(**common, body_kind=body_kind, value=_cap_body(values[0], _JSON_BODY_CHARS))
