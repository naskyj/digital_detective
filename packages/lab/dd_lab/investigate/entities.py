"""Extract entities of interest from a single artifact (key, value).

Scoped deliberately narrow: unambiguous identifiers and links rather than
generic username guessing, because every hit becomes a citation in a court-
facing report and a false positive there is worse than a missed one. Add more
patterns as real cases show a need (see docs/MODEL_TRAINING.md for the
weak-supervision precedent — same philosophy: precision first).

Each `EntityHit.normalized` is what `correlate.py` searches the warehouse for,
so cross-app / cross-source links work (a Roblox asset id embedded in a Cache
JSON blob, the same digits inside a memory-string carve, the same digits as an
isolated value in an IndexedDB row).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --------------------------------------------------------------------- types


@dataclass(frozen=True)
class EntityHit:
    entity_type: str  # e.g. "discord_user_id", "roblox_user_id", "email", "discord_invite"
    normalized: str  # the value correlate.py searches for
    span: str  # the matched substring, for the report/citation


# --------------------------------------------------------------------- regex

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_DISCORD_INVITE = re.compile(
    r"\b(?:discord\.gg|discord(?:app)?\.com/invite)/([A-Za-z0-9-]{2,32})\b", re.I
)
_DISCORD_USER_URL = re.compile(r"\bdiscord(?:app)?\.com/users/(\d{15,20})\b", re.I)
# Discord's own mention syntax in message text/embeds, e.g. "<@1473705902501531761>"
# (a nickname mention is "<@!id>"; a role mention "<@&id>" does NOT match this,
# deliberately - a role isn't a person). Found via real captured chat content
# (a Cache-stored referenced_message) where a raw snowflake never appears.
_DISCORD_MENTION = re.compile(r"<@!?(\d{15,20})>")
_ROBLOX_USER_URL = re.compile(r"\broblox\.com/users/(\d{1,15})\b", re.I)
_ROBLOX_ASSET = re.compile(r"\brbxassetid[:=/]{1,3}(\d{1,15})\b", re.I)
_SNOWFLAKE = re.compile(r"^\d{15,20}$")  # Discord snowflake shape (also matches raw Roblox ids)

# key-name hints that turn a bare numeric value into a typed id instead of "just a number"
_ID_KEY_HINT = re.compile(
    r"(author|user|recipient|guild|channel|owner|sender|member|snowflake|dm_)?id\b", re.I
)
_ROBLOX_KEY_HINT = re.compile(r"roblox|rbx", re.I)


def extract_entities(app: str, key: str, value: str) -> list[EntityHit]:
    """`app`/`key` give context a regex on `value` alone can't (e.g. a bare
    17-digit number is a Discord id only when the key or app says so)."""
    if not value:
        return []
    hits: list[EntityHit] = []

    for m in _EMAIL.finditer(value):
        hits.append(EntityHit("email", m.group(0).lower(), m.group(0)))

    for m in _DISCORD_INVITE.finditer(value):
        hits.append(EntityHit("discord_invite", m.group(1), m.group(0)))

    for m in _DISCORD_USER_URL.finditer(value):
        hits.append(EntityHit("discord_user_id", m.group(1), m.group(0)))

    for m in _DISCORD_MENTION.finditer(value):
        hits.append(EntityHit("discord_user_id", m.group(1), m.group(0)))

    for m in _ROBLOX_USER_URL.finditer(value):
        hits.append(EntityHit("roblox_user_id", m.group(1), m.group(0)))

    for m in _ROBLOX_ASSET.finditer(value):
        hits.append(EntityHit("roblox_asset_id", m.group(1), m.group(0)))

    # a bare snowflake-shaped value: only treated as an id when key or app
    # context says so, to avoid tagging every long numeric string in a dump.
    stripped = value.strip()
    if _SNOWFLAKE.match(stripped):
        app_l, key_l = app.lower(), key.lower()
        if "discord" in app_l and _ID_KEY_HINT.search(key_l):
            hits.append(EntityHit("discord_user_id", stripped, stripped))
        elif ("roblox" in app_l and _ID_KEY_HINT.search(key_l)) or _ROBLOX_KEY_HINT.search(key_l):
            hits.append(EntityHit("roblox_user_id", stripped, stripped))

    return hits
