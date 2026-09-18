"""Extract candidate strings from a memory dump.

Streaming ASCII + UTF-16LE extraction with light filtering. For Electron targets
the renderer heap is full of chat text, tokens, URLs and user records, so these
strings feed straight into the same classification ensemble as the disk
artifacts.

The shape filters below (``_DROP_PREFIX``, the asset/schema URL exclusion,
``_is_code_or_pki``, ``_looks_like_sentence``, ``_JSON_KEY``, ``_TOAST_TAG``)
are ported - and, where noted, tightened - from a reference extractor
(mem_artifact_extractor.py, see mem_artifact_extractor_EXPLAINED.md) that was
tuned by hand against a per-process (renderer-heap) memory dump. Only its
Phase-1 SHAPE rules are used here ("does this look like a readable artifact,
not machine noise") - its Phase-2 ground-truth fingerprint matching is a
recall-scoring harness for that one case, not a relevance filter, and is
deliberately not ported. Its chat-line rule (`Name: message`) is also not
ported: measured against this project's real evidence, a *full* physical-
memory image rather than a single process's heap, it net-increased kept
volume by picking up driver/RPC diagnostics and shader code that never
appear inside a browser process. The remaining work of deciding whether a
surviving string is actually *relevant* to a case stays with the
classification ensemble downstream, same as before: this module still only
filters on shape, never on meaning.
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
    r"\b\d{1,3}(?:\.\d{1,3}){3}\b|[A-Za-z]:\\)"
)

# --- unconditional noise-shape drops -------------------------------------

# Browser/OS telemetry & metric-counter names - never user content.
_DROP_PREFIX = (
    "PageLoad.", "Microsoft.", "chrome.", "V8 ", "Mojo", "HKEY", "Blink.",
    "Net.", "Sqlite", "OptimizationGuide", "SubresourceFilter", "PROCESSOR",
    "PSModule",
)
# A pure hex blob (hash/key/binary) - not readable evidence.
_HEX_BLOB = re.compile(r"[0-9a-fA-F]{16,}")

# Static-asset, telemetry, and schema/namespace URLs - not user content.
# Deliberately does NOT exclude image/CDN URLs (cdn.discordapp/avatars,
# files.slack, /attachments): those carry user/channel IDs and filenames.
_NOISE_URL = re.compile(
    r"\.(?:css|js|woff2?|ttf|otf|svg|ico|map)(?:[?#]|$)|/assets/|"
    r"a\.slack-edge|gstatic|gvt1|doubleclick|googletagmanager|"
    r"google-analytics|/sentry|sentry\.io|segment\.io|segment\.com|"
    r"/beacon|/telemetry|fonts\.(?:googleapis|gstatic)|/typography|"
    r"schemas\.(?:microsoft|xmlsoap|android)\.com|www\.w3\.org|"
    r"osgwiki\.com|/wiki/|purl\.org|ns\.adobe|\bmsn\.com|xboxlive|"
    r"learn\.microsoft|windows\.msn",
    re.I,
)

_CODE_SIG = re.compile(
    r"\.(?:dll|exe|cpp|node|pdb)\b|__cdecl|__stdcall|__fastcall|"
    r"webpackchunk|use strict|=>|::|#\d+\.\d+|\\jtirules\\|"
    r"this program cannot|@param|adbe vector",
    re.I,
)
_PKI_SIG = re.compile(
    r"digicert|time-stamp pca|global root|windows production|verisign|"
    r"sectigo|certificate authority",
    re.I,
)


def _is_code_or_pki(s: str) -> bool:
    if _CODE_SIG.search(s) or _PKI_SIG.search(s):
        return True
    if "(" not in s and "{" not in s:
        return False
    punct = sum(1 for c in s if c in "{}();<>=:*/\\")
    return punct / len(s) > 0.12


# --- extra positive interest signals --------------------------------------

_JSON_KEY = re.compile(
    r'"(?:content|message|body|text|username|displayName|display_name|'
    r'description|draft|title|about|value)"\s*:',
    re.I,
)
_TOAST_TAG = re.compile(r"<(text|title|body)[^>]*>[^<]{3,}</\1>", re.I)

# NOTE: the reference script also adds a `Name: message` chat-line rule here.
# Measured against this project's real evidence (a full physical-memory
# image, not a single process's renderer heap - the domain the reference
# script was tuned against) it net-INCREASED kept volume: without an app
# process boundary, `word: word` also matches driver/RPC diagnostics, shader
# register declarations, and error-code strings that never appear inside a
# browser process. Left out here; worth revisiting if/when this pipeline
# also carves per-process dumps rather than only full-image ones.

# Stricter replacement for the old "3 words of 4+ letters" heuristic: also
# requires the run to actually read like prose (mostly lowercase, has a
# space) and excludes env-var assignments / source-doc & format-string
# patterns that otherwise sneak through a bare word-count check. Unlike the
# reference script's plain "findall >= 4 words anywhere", this keeps the
# original adjacency requirement (words back to back, whitespace-only
# between them) - dropping that requirement is what caused the false
# positives above: Windows APPX manifest attributes ('ActivatableClassId="..."
# ThreadingModel="both"') have plenty of 3+ letter words scattered through
# them, just never four of them in a row.
_WORD_RUN = re.compile(r"[A-Za-z]{3,}(?:\s+[A-Za-z]{3,}){3,}")  # 4+ words in a row
_ENV_ASSIGN = re.compile(r"^[A-Za-z0-9_()]+=")
_CODEDOC = re.compile(
    r"require\(script|script\.Parent|Roact|getScreen|withNavigation|"
    r"@\{\d+[:x}]|\{\d+:x\}"
)


def _looks_like_sentence(s: str) -> bool:
    if _ENV_ASSIGN.match(s) or _CODEDOC.search(s):
        return False
    if not _WORD_RUN.search(s):
        return False
    return " " in s and sum(c.islower() for c in s) / len(s) > 0.35


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
    # unconditional noise-shape drops - telemetry, hashes, code/PKI signatures,
    # static-asset & schema URLs. None of these are ever evidence.
    if s.startswith(_DROP_PREFIX):
        return None
    if _HEX_BLOB.fullmatch(s):
        return None
    if _NOISE_URL.search(s):
        return None
    if _is_code_or_pki(s):
        return None
    # positive interest signals
    interesting = bool(_INTERESTING.search(s)) or bool(_JSON_KEY.search(s)) or bool(_TOAST_TAG.search(s))
    if not interesting and _looks_like_sentence(s):
        interesting = True
    if not interesting:
        return None
    seen.add(s)
    return s
