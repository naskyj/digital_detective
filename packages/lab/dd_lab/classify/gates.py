"""Gates: checks that override the voting ensemble entirely rather than
casting a vote themselves - classifier_backend_spec.md's "empty-body
pre-gate" and the deleted/edited relevance boost.

Neither of these is a voter (see rules.py/entropy.py/ner.py for those): a
gate either short-circuits classification before any voter runs (the
empty-body check) or adjusts the already-combined relevance_score after the
fact (the deleted boost) - they never contribute to the weighted-sum score
`_combine()` produces, so they need no voter weight in ensemble.py's
_W_* constants.
"""

from __future__ import annotations

import json

# Chromium/IndexedDB's own literal placeholder for a JS `undefined` value,
# distinct from JSON `null` - seen verbatim in real IndexedDB/LocalStorage
# dumps (structured-clone serialization writes it out as this exact string).
_UNDEFINED_LITERAL = "<Undefined>"


def _is_empty_json_value(v: object) -> bool:
    """A JSON value counts as empty if it's None/""/[]/{}, or a dict/list
    whose every value is itself empty (recursively) - e.g. the spec's own
    examples: {"data": []}, {"consents": [], "nextCursor": null},
    {"selectedChannelIds": {}, "knownThreadIds": []}. Any real scalar
    (a non-empty string, a number, a boolean) anywhere in the structure
    makes the whole thing non-empty - this is deliberately not "the object
    has few keys", only "the object carries no actual content"."""
    if v is None or v == "" or v == [] or v == {}:
        return True
    if isinstance(v, dict):
        return all(_is_empty_json_value(x) for x in v.values())
    if isinstance(v, list):
        return all(_is_empty_json_value(x) for x in v)
    return False


def is_structurally_empty(value: str) -> bool:
    """True for a value that carries no actual content, regardless of what
    the artifact's key or URL says - classifier_backend_spec.md: "If the
    decoded value is structurally empty, label not relevant regardless of
    any ID in the key or URL." Runs before any voter (see
    Classifier.classify()) - validated on real cache data as catching ~35
    false positives where a case/workspace ID lived in the URL of an API
    response with no actual payload."""
    s = value.strip()
    if s == "" or s == "{}" or s == "[]" or s.lower() == "null" or s == _UNDEFINED_LITERAL:
        return True
    if s[:1] not in "{[":
        return False  # not JSON-shaped; a bare non-empty scalar is never "empty"
    try:
        parsed = json.loads(s)
    except (ValueError, TypeError):
        return False
    return _is_empty_json_value(parsed)


# Recovered/tombstoned records are inherently more interesting to an
# investigator than a live one (deletion is itself a signal) - a fixed,
# documented bump rather than another weighted vote, since it's not
# evidence of *what* the record is about, only that it's worth a second
# look. Applied once, after _combine(), only when the category isn't
# already GENERAL (deleting a GENERAL/irrelevant record is not itself
# interesting).
DELETED_RELEVANCE_BOOST = 0.15


def boosted_relevance(relevance_score: float, *, deleted: bool, is_general: bool) -> float:
    if not deleted or is_general:
        return relevance_score
    return round(min(1.0, relevance_score + DELETED_RELEVANCE_BOOST), 3)
