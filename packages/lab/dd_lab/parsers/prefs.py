"""Preferences and Local State — plain JSON config files.

`Preferences` holds per-profile settings; `Local State` holds browser-wide state
including (on Windows) the DPAPI-wrapped key used to encrypt cookies and Local
Storage. Both are pure JSON, so they flatten directly.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

from dd_core.enums import SourceType
from dd_core.schema import NormalizedArtifact
from dd_lab.parsers.base import LOCAL_STATE, PREFERENCES, ParseContext, emit_value

log = logging.getLogger("dd_lab.parsers.prefs")


def parse(ctx: ParseContext) -> Iterator[NormalizedArtifact]:
    for relative, prefix in ((PREFERENCES, "preferences"), (LOCAL_STATE, "local_state")):
        if not ctx.has(relative):
            continue
        fpath = ctx.path(relative)
        try:
            text = fpath.read_text("utf-8", errors="replace")
        except Exception:
            log.exception("could not read %s", fpath)
            continue
        source_type = SourceType.PREFERENCES if prefix == "preferences" else SourceType.LOCAL_STATE
        yield from emit_value(
            ctx,
            source_type=source_type,
            source_file=ctx.rel(fpath),
            locator=prefix,
            key=prefix,
            raw_value=text,
        )
