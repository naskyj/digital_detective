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

# Slack's own token-prefix scheme (xoxb- bot, xoxp- legacy user, xoxc- client,
# xoxa- app-level, xoxe-/xoxr- refresh/rotation, ...) - distinctive enough on
# its own, no app/key context needed, same as the URL/mention patterns above.
# A real xoxc- client token was found in real evidence this session.
_SLACK_TOKEN = re.compile(r"\bxox[a-z]-[0-9a-zA-Z-]{8,}\b")
# Team/user ids ("T"/"U" + 8-10 alnum, e.g. real evidence: T0C1J4SDLG2,
# U0C1DHP5PPX) are NOT distinctive enough to scan unscoped - a hex/hash
# fragment can share this shape by pure chance. Only trusted with "slack" in
# the app, same reasoning as the bare-snowflake case below for Discord/Roblox.
_SLACK_TEAM_ID = re.compile(r"\bT[A-Z0-9]{8,10}\b")
_SLACK_USER_ID = re.compile(r"\bU[A-Z0-9]{8,10}\b")

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

    for m in _SLACK_TOKEN.finditer(value):
        hits.append(EntityHit("slack_token", m.group(0), m.group(0)))

    # team/user ids are generic-shaped enough that they need app context -
    # scan both key and value together, since real evidence shows the id
    # embedded in a compound key string as often as in the value itself.
    if "slack" in app.lower():
        haystack = f"{key} {value}"
        for m in _SLACK_TEAM_ID.finditer(haystack):
            hits.append(EntityHit("slack_team_id", m.group(0), m.group(0)))
        for m in _SLACK_USER_ID.finditer(haystack):
            hits.append(EntityHit("slack_user_id", m.group(0), m.group(0)))

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
