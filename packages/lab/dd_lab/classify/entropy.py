"""Entropy/type-shape voter — a cheap, generic tie-breaker.

Every other voter (regex_hits/keyword_hits: rules.py, ner_vote: ner.py) needs
some pre-existing signal to fire: a key name, a keyword, a recognisable
pattern, a name-shaped span. This one looks only at the raw SHAPE of a bare
value with none of that - a session token or hash with no "token"/"secret" in
its key, a plain sentence with no chat-ish key name, a lone boolean/version
string that isn't evidence of anything. Proposed by a labelling partner
reviewing real misses: exactly the bare, unlabelled values that fall through
every other voter and land at GENERAL/0.0, invisible to a reviewer skimming
hundreds of thousands of rows.

Three shapes, each a weak, single-vote signal (never a strong determinant on
its own - see _W_ENTROPY in ensemble.py):

  - A long, unbroken, mixed alphanumeric token with high per-character
    Shannon entropy reads as a random credential/session-token/id shape ->
    CREDENTIAL.
  - Several real word-shaped runs in a row reads as a sentence -> CHAT_CONTENT.
  - A bare boolean/enum/version/number has no evidentiary content on its own
    -> GENERAL, an explicit vote (not silence) so it can actually out-weigh a
    weak false-positive hit from another voter, which is the whole point of a
    tie-breaker.

Anything else abstains (returns None) rather than guessing.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from dd_core.enums import ArtifactCategory as C

# A single unbroken token (no whitespace at all) - natural language and
# structured config values are practically never one long unbroken run, so
# this is the cheap pre-filter before bothering with entropy at all.
_TOKEN = re.compile(r"^\S+$")
_HAS_LETTER = re.compile(r"[A-Za-z]")
_HAS_DIGIT = re.compile(r"\d")

# Below this length, "looks random" is unreliable (too few characters for
# frequency-based entropy to mean much, and short random-looking strings are
# common and mostly harmless - e.g. a 6-char hex color or CSS class).
_MIN_TOKEN_LEN = 16
# Empirically, structured mixed alnum tokens (UUIDs, hashes, session ids,
# API keys) land at 3.7+ bits/char; a fixed English word or a repetitive
# non-random string lands lower. Deliberately conservative - a false
# negative here just falls through to "no vote", not a wrong one.
_ENTROPY_THRESHOLD = 3.7

# 4+ real word-shaped runs (2+ letters) separated by whitespace, in a row -
# "looks like a sentence" (same idea as dd_lab/memory/strings.py's own
# _WORD_RUN sentence heuristic, reimplemented here rather than imported -
# extraction-time noise filtering and classification-time category voting
# are different concerns with different tuning needs, even though the shape
# looks similar).
_WORD_RUN = re.compile(r"[A-Za-z]{2,}(?:\s+[A-Za-z]{2,}){3,}")

# A bare boolean/null/enum-ish token, a SHORT plain integer, or a version
# string - no evidentiary content on its own, regardless of context.
#
# Deliberately narrower than "any bare number":
#   - the integer branch is capped at 4 digits - a long digit run is far more
#     likely to be a card number, phone number, or snowflake/asset id (all
#     genuinely evidentiary) than an innocuous small count/index, so it's
#     excluded rather than risk actively downgrading one.
#   - the version-string branch requires EXACTLY two dots (three segments,
#     "X.Y.Z"/"v10.0.19045") - not "1-3 dots", which would also match a
#     coordinate ("51.5072", one dot) or an IPv4 address ("192.168.1.44",
#     three dots). Both are genuinely evidentiary (LOCATION_INFO/
#     NETWORK_DETAIL) and must never be pushed toward "not relevant" by a
#     tie-breaker that only looked at the dot-shape.
_BOOLEAN_OR_ENUM = re.compile(r"^(?:true|false|yes|no|null|none|nil|-?\d{1,4}|v?\d+\.\d+\.\d+)$", re.I)

_W_CREDENTIAL = 0.5
_W_CHAT = 0.4
_W_NOT_RELEVANT = 0.3


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    counts = Counter(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def entropy_vote(key: str, value: str) -> tuple[C, float] | None:  # noqa: ARG001 - key kept for a uniform voter signature
    """(category, strength) from the value's shape alone, or None to abstain."""
    v = value.strip()
    if not v:
        return None

    if _BOOLEAN_OR_ENUM.match(v):
        return C.GENERAL, _W_NOT_RELEVANT

    if (
        _TOKEN.match(v)
        and len(v) >= _MIN_TOKEN_LEN
        and _HAS_LETTER.search(v)
        and _HAS_DIGIT.search(v)
        and _shannon_entropy(v) >= _ENTROPY_THRESHOLD
    ):
        return C.CREDENTIAL, _W_CREDENTIAL

    if _WORD_RUN.search(v):
        return C.CHAT_CONTENT, _W_CHAT

    return None
