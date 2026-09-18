"""Explode JSON *arrays* into one artifact per entry. Objects are NOT split.

Electron apps store most of their interesting state as JSON — cache response
bodies, Local Storage values, IndexedDB records, Preferences. An **object**
is one artifact, whole, however deep — it is never split by key. Only an
**array** is exploded, one artifact per entry, recursing into any array
nested inside another array. A value nested inside an object is only
reachable this way if an array is found before an object is - the FIRST
object the walk reaches stops it right there, whole:

    ["a", "b", {"name": "bob"}]
    ->  [0]  = "a"
        [1]  = "b"
        [2]  = {"name": "bob"}      <- one whole object, not split further

    {"name": "bob", "friends": ["x", "y"]}
    ->  $    = {"name": "bob", "friends": ["x", "y"]}
               the WHOLE document, one artifact - an object was hit
               immediately, so "friends" is never reached to be exploded

Array indices always use unquoted brackets (`[0]`); object keys that aren't
safe bare identifiers (containing brackets, dots, spaces, ...) use quoted
brackets instead (`["[python]"]`), never a dot - relevant wherever an array
of objects is being walked and a path needs to name which array entry, not
which object key (VS Code's own settings really do use `"[python]"` as a
literal object key, language-scoped defaults, which is where this distinction
originally came from, back when object keys were still walked too).
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from dd_core.enums import ValueType

# Guard against pathological documents (arrays nested very deep, or huge).
# A forensic classifier gains nothing from tens of thousands of entries out
# of one API blob; past the cap the remainder is emitted as a single JSON leaf.
MAX_DEPTH = 16
MAX_LEAVES = 2_000


def _classify_scalar(v: object) -> tuple[str, ValueType]:
    if v is None:
        return "", ValueType.NULL
    if isinstance(v, bool):
        return ("true" if v else "false"), ValueType.BOOLEAN
    if isinstance(v, (int, float)):
        return str(v), ValueType.NUMBER
    text = str(v)
    lowered = text.strip().lower()
    if lowered.startswith(("http://", "https://", "ws://", "wss://")):
        return text, ValueType.URL
    return text, ValueType.STRING


def iter_leaves(obj: object, prefix: str = "") -> Iterator[tuple[str, str, ValueType]]:
    """Yield (key_path, value_text, value_type) for every artifact in obj -
    one per array entry (recursively, for an array nested in an array), or
    exactly one for obj itself if it's an object or a scalar (see the
    module docstring)."""
    yield from _walk(obj, prefix, 0, [0])


def _json_default(obj: object) -> str:
    """`json.dumps(..., default=)` fallback for a value plain JSON can't
    encode - real evidence, not theoretical: some IndexedDB records
    (ccl_chromium_reader hands them back already deserialized) carry raw
    `bytes` values nested inside an otherwise-JSON-shaped object (blob
    references, binary payloads). Before objects became one artifact whole,
    only individual scalar leaves ever reached json encoding, and
    `_classify_scalar` stringifies anything non-JSON-native via `str()`
    before that point; now a whole object is dumped in one call, so the
    same str()-fallback has to be given to json.dumps directly, or one
    bytes value anywhere inside an otherwise-fine document raises
    TypeError and (since parse() catches broadly per leveldb file/database,
    not per record) can silently drop every record after it - confirmed on
    real evidence: Slack's IndexedDB store lost thousands of records this
    way before this fix."""
    return str(obj)


def _walk(
    obj: object, path: str, depth: int, budget: list[int]
) -> Iterator[tuple[str, str, ValueType]]:
    if budget[0] >= MAX_LEAVES:
        return
    if depth > MAX_DEPTH:
        budget[0] += 1
        yield path or "$", json.dumps(obj, default=_json_default)[:2000], ValueType.JSON
        return

    if isinstance(obj, dict):
        # An object is one artifact, whole, never split by key - only an
        # array (below) gets exploded into per-entry artifacts. An empty
        # object is still one artifact ("{}"), same as any other object.
        budget[0] += 1
        yield path or "$", json.dumps(obj, default=_json_default), ValueType.JSON
        return
    elif isinstance(obj, (list, tuple)):
        if not obj:
            budget[0] += 1
            yield path or "$", "[]", ValueType.JSON
            return
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]", depth + 1, budget)
    else:
        budget[0] += 1
        text, vtype = _classify_scalar(obj)
        yield path or "$", text, vtype


def try_parse_json(raw: str | bytes) -> object | None:
    """Return the parsed object, or None if it isn't JSON."""
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None
    s = raw.strip()
    if not s or s[0] not in "{[":
        return None
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return None
