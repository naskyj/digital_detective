"""Extract candidate strings from a memory dump.

Streaming ASCII + UTF-16LE extraction with light filtering. For Electron targets
the renderer heap is full of chat text, tokens, URLs and user records, so these
strings feed straight into the same classification ensemble as the disk
artifacts.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

_CHUNK = 4 * 1024 * 1024
_ASCII_RUN = re.compile(rb"[\x20-\x7e]{%d,}")
_UTF16_RUN = re.compile(rb"(?:[\x20-\x7e]\x00){%d,}")

# keep strings that look like they carry information, drop base64 mush / repeats
_INTERESTING = re.compile(
    r"(https?://|wss?://|@|[Tt]oken|[Bb]earer|password|secret|"
    r"\b\d{1,3}(?:\.\d{1,3}){3}\b|[A-Za-z]:\\|"
    r"[A-Za-z]{4,}\s+[A-Za-z]{4,}\s+[A-Za-z]{4,})"
)


def iter_strings(path: str | Path, min_len: int = 6, max_strings: int = 400_000) -> Iterator[str]:
    ascii_re = re.compile(_ASCII_RUN.pattern % min_len)
    utf16_re = re.compile(_UTF16_RUN.pattern % min_len)
    seen: set[str] = set()
    produced = 0
    tail = b""
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            buf = tail + chunk
            for m in ascii_re.finditer(buf):
                if (s := _keep(m.group(0).decode("ascii", "ignore"), seen)) is not None:
                    produced += 1
                    yield s
                    if produced >= max_strings:
                        return
            for m in utf16_re.finditer(buf):
                text = m.group(0).decode("utf-16-le", "ignore")
                if (s := _keep(text, seen)) is not None:
                    produced += 1
                    yield s
                    if produced >= max_strings:
                        return
            tail = buf[-256:]


def _keep(s: str, seen: set[str]) -> str | None:
    s = s.strip()
    if len(s) < 6 or len(s) > 4000:
        return None
    if s in seen:
        return None
    # skip low-information runs (single repeated char, no letters at all)
    if len(set(s)) < 4:
        return None
    if not _INTERESTING.search(s):
        return None
    seen.add(s)
    return s
