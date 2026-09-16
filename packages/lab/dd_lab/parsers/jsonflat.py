"""Flatten nested JSON into leaf (key-path, value) pairs.

Electron apps store most of their interesting state as JSON — cache response
bodies, Local Storage values, IndexedDB records, Preferences. The classifier
works best on `key = value` granularity (the same shape the original system's
labelled spreadsheet used), so every JSON document is exploded into one row per
scalar leaf, with a dotted/bracketed path as the key.

    {"user": {"name": "bob", "friends": ["x", "y"]}}
    ->  user.name          = "bob"
        user.friends[0]     = "x"
        user.friends[1]     = "y"

Array indices always use unquoted brackets (`[0]`); object keys that aren't
safe bare identifiers (containing brackets, dots, spaces, ...) use quoted
brackets instead (`["[python]"]`), never a dot. VS Code's own settings really
do use `"[python]"` as a literal object key (language-scoped defaults); a
dict key can, in principle, be anything, including a string that reads like
an array index (`"[0]"`). Without this distinction those collide with real
array indices and can't be told apart after flattening — this is what makes
`key["oddly named"]` vs `key[0]` unambiguous everywhere, not just at the
outermost segment.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator

from dd_core.enums import ValueType

# Guard against pathological documents (deeply nested / huge arrays). A forensic
# classifier gains nothing from tens of thousands of leaves out of one API blob;
# past the cap the remainder of the document is emitted as a single JSON leaf.
MAX_DEPTH = 16
MAX_LEAVES = 2_000

_SAFE_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _key_segment(k: str, *, is_first: bool) -> str:
    """One object-key path segment, safe to concatenate directly onto the
    path built so far (no extra separator needed by the caller)."""
    if _SAFE_KEY.match(k):
        return k if is_first else f".{k}"
    escaped = k.replace("\\", "\\\\").replace('"', '\\"')
    return f'["{escaped}"]'


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
    """Yield (key_path, value_text, value_type) for every scalar leaf in obj."""
    yield from _walk(obj, prefix, 0, [0])


def _walk(
    obj: object, path: str, depth: int, budget: list[int]
) -> Iterator[tuple[str, str, ValueType]]:
    if budget[0] >= MAX_LEAVES:
        return
    if depth > MAX_DEPTH:
        budget[0] += 1
        yield path or "$", json.dumps(obj)[:2000], ValueType.JSON
        return

    if isinstance(obj, dict):
        if not obj:
            budget[0] += 1
            yield path or "$", "{}", ValueType.JSON
            return
        for k, v in obj.items():
            child = f"{path}{_key_segment(str(k), is_first=not path)}"
            yield from _walk(v, child, depth + 1, budget)
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
