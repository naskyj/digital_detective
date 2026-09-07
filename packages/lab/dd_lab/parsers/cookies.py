"""Cookies — the `<app>/cookies` SQLite database.

Cookie *values* are encrypted (AES-GCM on modern Chromium; the key is DPAPI-
wrapped on Windows and stored in `Local State`). Decryption is only reliable on
the original machine/account, so this parser records cookie metadata and marks
whether the value was recoverable. Actual decryption is a separate, opt-in step
run on the acquiring host (out of scope for the cross-platform parser).
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

from dd_core.enums import SourceType, ValueType
from dd_core.schema import NormalizedArtifact
from dd_lab.parsers.base import COOKIES, ParseContext, make_artifact

log = logging.getLogger("dd_lab.parsers.cookies")

_CHROME_EPOCH = datetime(1601, 1, 1, tzinfo=UTC)


def _chrome_time(micros: int | None) -> datetime | None:
    if not micros:
        return None
    try:
        return _CHROME_EPOCH + timedelta(microseconds=micros)
    except (OverflowError, OSError):
        return None


def parse(ctx: ParseContext) -> Iterator[NormalizedArtifact]:
    from dd_lab.parsers.base import COOKIES_LEGACY

    db_path = next(
        (ctx.path(p) for p in (COOKIES, COOKIES_LEGACY) if ctx.has(p)), None
    )
    if db_path is None:
        return
    source_file = ctx.rel(db_path)
    try:
        con = sqlite3.connect(f"file:{db_path}?immutable=1", uri=True)
    except sqlite3.Error:
        log.exception("could not open cookies db %s", db_path)
        return
    try:
        con.row_factory = sqlite3.Row
        cols = {r[1] for r in con.execute("PRAGMA table_info(cookies)")}
        has_enc = "encrypted_value" in cols
        rows = con.execute(
            "SELECT host_key, name, path, "
            "  CASE WHEN :he THEN length(encrypted_value) ELSE 0 END AS enc_len, "
            "  value, creation_utc, expires_utc, last_access_utc, is_secure, is_httponly "
            "FROM cookies",
            {"he": has_enc},
        )
        for r in rows:
            locator = f"{r['host_key']}\x1f{r['name']}\x1f{r['path']}"
            plaintext = r["value"] or ""
            recovered = bool(plaintext)
            value = plaintext if recovered else f"<encrypted {r['enc_len']} bytes>"
            yield make_artifact(
                ctx, source_type=SourceType.COOKIES, source_file=source_file,
                locator=locator, key=f"cookie.{r['host_key']}.{r['name']}",
                value=value,
                value_type=ValueType.STRING if recovered else ValueType.BINARY,
                record_ts=_chrome_time(r["creation_utc"]),
            )
            for label in ("creation_utc", "expires_utc", "last_access_utc"):
                ts = _chrome_time(r[label])
                if ts:
                    yield make_artifact(
                        ctx, source_type=SourceType.COOKIES, source_file=source_file,
                        locator=locator, key=f"cookie.{r['host_key']}.{r['name']}.{label}",
                        value=ts.isoformat(), value_type=ValueType.TIMESTAMP, record_ts=ts,
                    )
    except sqlite3.Error:
        log.exception("error reading cookies db %s", db_path)
    finally:
        with contextlib.suppress(sqlite3.Error):
            con.close()
