"""Native (non-Chromium) app data: config/log files no named parser claims.

Two sources land here, both real gaps confirmed against actual evidence
behaviour rather than assumed:

- `native/...` - a curated, extension-allowlisted pick from a WebView2/MSIX-
  hosted app's own directory tree (see `dd_core.evidence.plan_native_root_copy`
  and `iter_profiles_for_entry`'s docstring) - e.g. Roblox's own logs/XML/JSON
  config, sitting well outside the Chromium profile embedded deep inside that
  same tree, which is the *only* thing this project images for a non-Electron
  app otherwise.
- any *unclaimed* file loose in a normal Electron app's own profile root -
  e.g. Discord's own `settings.json`, distinct from Chromium's `Preferences`
  and already copied today by `plan_profile_copy()`'s whole-profile copy, but
  never decoded by any named parser.

`dd_core.evidence.source_type_for_path` is the single source of truth for
"is this path already claimed by a named parser" (it's the same function the
signed manifest's own per-file inventory uses) - reused here rather than a
second, easily-drifting copy of the same prefix list.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

from dd_core.enums import SourceType
from dd_core.evidence import MISC_EXTENSIONS, source_type_for_path
from dd_core.schema import NormalizedArtifact
from dd_lab.parsers.base import ParseContext, emit_parsed_value, emit_value

log = logging.getLogger("dd_lab.parsers.appdata")

MAX_FILES = 2_000
MAX_LINES_PER_FILE = 50_000
MIN_LINE_LEN = 4


def _xml_to_obj(el: ET.Element) -> dict:
    """A minimal, generic element -> dict conversion (attributes prefixed
    "@", text under "#text", same-tag children collected into a list) fed
    through the *existing* jsonflat leaf-walk (emit_parsed_value) rather
    than reimplementing array-explode/object-whole logic a second time for
    XML specifically."""
    obj: dict = {}
    for k, v in el.attrib.items():
        obj[f"@{k}"] = v
    text = (el.text or "").strip()
    if text:
        obj["#text"] = text
    children: dict[str, list] = {}
    for child in el:
        children.setdefault(child.tag, []).append(_xml_to_obj(child))
    for tag, items in children.items():
        obj[tag] = items if len(items) > 1 else items[0]
    return obj


def _parse_one_file(ctx: ParseContext, f: Path, rel: str) -> Iterator[NormalizedArtifact]:
    source_file = ctx.rel(f)
    suffix = f.suffix.lower()
    key_base = f"appdata.{rel}"
    try:
        if suffix == ".json":
            text = f.read_text("utf-8", errors="replace")
            yield from emit_value(
                ctx, source_type=SourceType.APP_DATA, source_file=source_file,
                locator=rel, key=key_base, raw_value=text,
            )
        elif suffix == ".xml":
            tree = ET.parse(f)  # noqa: S314 - evidence, not untrusted network input
            yield from emit_parsed_value(
                ctx, source_type=SourceType.APP_DATA, source_file=source_file,
                locator=rel, key=key_base, parsed=_xml_to_obj(tree.getroot()),
            )
        else:
            # .log / .txt / .ini / .cfg - no reliable shared structure across
            # apps to assume; one artifact per substantive line, generic
            # `key=value` split where a line has one (.ini/.cfg's own shape),
            # else the bare line as the value.
            text = f.read_text("utf-8", errors="replace")
            for n, line in enumerate(text.splitlines()):
                if n >= MAX_LINES_PER_FILE:
                    break
                line = line.strip()
                if len(line) < MIN_LINE_LEN:
                    continue
                if suffix in (".ini", ".cfg") and "=" in line and not line.startswith(("#", ";")):
                    k, _, v = line.partition("=")
                    k = k.strip()
                    if k:
                        yield from emit_value(
                            ctx, source_type=SourceType.APP_DATA, source_file=source_file,
                            locator=f"{rel}#line{n}", key=f"{key_base}.{k}", raw_value=v.strip(),
                        )
                        continue
                yield from emit_value(
                    ctx, source_type=SourceType.APP_DATA, source_file=source_file,
                    locator=f"{rel}#line{n}", key=f"{key_base}.line{n}", raw_value=line,
                )
    except Exception:
        log.exception("could not parse app-data file %s", f)


def parse(ctx: ParseContext) -> Iterator[NormalizedArtifact]:
    if not ctx.app_root.is_dir():
        return
    seen = 0
    for f in sorted(ctx.app_root.rglob("*")):
        if seen >= MAX_FILES:
            log.warning("appdata parser: MAX_FILES (%d) reached for %s, stopping", MAX_FILES, ctx.app)
            break
        if not f.is_file() or f.suffix.lower() not in MISC_EXTENSIONS:
            continue
        rel = f.relative_to(ctx.app_root).as_posix()
        if source_type_for_path(f"{ctx.app}/{rel}") != "app_data":
            continue  # already claimed by a named parser
        seen += 1
        yield from _parse_one_file(ctx, f, rel)
