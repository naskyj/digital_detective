"""Per-app classification profiles (classifier_backend_spec.md's DISCORD/
SLACK/ROBLOX profile sections) - real LevelDB/IndexedDB/cache key names and
value shapes, mapped onto the shared ArtifactCategory taxonomy, scoped to
one app so a Slack-shaped id doesn't fire on Discord evidence and vice
versa.

Category mapping note: the spec's own per-app tables label each bucket
with its *shared-taxonomy* name (identity/communication/draft/
account_settings/credential/navigation_activity/media/timestamp/
financial). Phase 1 (see enums.py's ArtifactCategory docstring) already
decided how those shared-taxonomy names land on this project's categories:
identity -> USER, communication -> CHAT_CONTENT, credential -> CREDENTIAL,
timestamp -> TIMESTAMP, financial -> PAYMENT_INFO; draft/account_settings/
navigation_activity/media are the new members added in that phase. Two
buckets the spec names but doesn't put in its shared taxonomy list
(Discord's "cross_app/activity", Roblox's "locale/state") are mapped to
the closest existing category (NAVIGATION_ACTIVITY and SYSTEM_DETAIL
respectively - SYSTEM_DETAIL already has a "locale" key keyword at weight
0.4, so this is consistent with existing precedent, not a new judgment
call from nothing).

Matching note: every rule here is searched against a *combined* `key +
value` haystack, not key-only or value-only. The spec's own tables mix
two different real shapes under one label - a genuine LevelDB/IndexedDB
record key (e.g. "DraftStore", literally the store name) and a JSON
*field* name from a record whose value this project's own parsers keep
whole (jsonflat.py explodes arrays but never splits an object - see that
module's docstring - so "robux"/"transaction"/"birthMonth" etc. live
inside `value` as JSON text, not as their own artifact `key`, unlike a
flat LevelDB key). Rather than track, per pattern, exactly which shape a
given real parser produces (this project's own extraction, not Axiom/
DLEAPP's, which is what the spec's own tables were validated against),
every rule searches `f"{key} {value}"` together - the same combined-
haystack approach infra_gazetteer.py and investigate/entities.py's Slack
id matching already use for this identical ambiguity.

These hits are unioned into regex_hits()'s own return list (see rules.py)
rather than tracked as a 6th voter - mechanically they're schema-key/
value-shape matches, identical in kind to what that function already does
generically; being app-scoped doesn't change the mechanism, so they reuse
_W_REGEX, no new weight constant.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from dd_core.enums import ArtifactCategory as C
from dd_core.patterns import CONTENT_FIELD_PAT

if TYPE_CHECKING:
    from dd_lab.classify.rules import RuleHit

# Each rule is (compiled pattern, category, weight), searched against the
# combined key+value haystack (see module docstring). Weight tiers,
# consistent with rules.py's own hand-picked-per-pattern style: a
# distinctive, app-specific LevelDB store name (DraftStore,
# MultiAccountStore, ...) or a highly distinctive value literal
# (.ROBLOSECURITY, a Slack token prefix) is high-precision -> 0.85-0.9; a
# generic field name that could plausibly appear in unrelated JSON
# (content, text, created) is weaker on its own -> 0.6; a broad value
# *shape* (a bare digit-run, a Slack id prefix) is weaker still, since the
# shape alone is common -> 0.5.

_DISCORD_RULES: tuple[tuple[re.Pattern, C, float], ...] = (
    (re.compile(r"user_?id|sender_?id|author_?id|discriminator|global_?name|username|"
                r"display_?name|\bavatar\b|\bbot\b", re.I), C.USER, 0.6),
    (re.compile(r"MultiAccountStore|user_id_cache|fingerprint|installation_id_v3|"
                r"ARISE_PROFILE|ARISE_DEVICE_ID|AudioContextSettingsMigrated:\d+"), C.USER, 0.85),
    (re.compile(r"\b\d{17,20}\b"), C.USER, 0.5),  # Discord snowflake shape
    # CONTENT_FIELD_PAT, not a bare "content": the plain substring matched
    # `cache.content_type` (an HTTP header, value `image/png`) on 5,891 real
    # artifacts, and `AdContentSeenStore`/`DismissibleContentFrameworkStore`
    # (Discord's own config stores) besides. See its docstring.
    (re.compile(rf"{CONTENT_FIELD_PAT}|message|message_?id|channel_?id|guild_?id|_?ThreadID|"
                r"in[_ ]?reply[_ ]?to|mentions?|reaction|reactors|pinned|message_?type|"
                r"embedded_?content", re.I), C.CHAT_CONTENT, 0.6),
    (re.compile(r"DraftStore"), C.DRAFT, 0.85),
    (re.compile(r"ThemeStore|UserSettingsProtoStore-Cache"), C.ACCOUNT_SETTINGS, 0.85),
    (re.compile(r"\btoken\b|\btokens\b|ARISEENGINE_AUTHTOKEN", re.I), C.CREDENTIAL, 0.85),
    (re.compile(r"SelectedGuildStore|SelectedChannelStore|RecentVoiceChannelStore|"
                r"QuickSwitcherStore|viewedThreadIds|ChannelSectionStore2|channelHistory|"
                r"voiceChannelHistory|lastConnectedTime"), C.NAVIGATION_ACTIVITY, 0.85),
    (re.compile(r"EmojiStoreV2|StickersPersistedStoreV2|TopEmojiStore|NewlyAddedEmojiStore"),
     C.NAVIGATION_ACTIVITY, 0.7),  # reaction/sticker *usage* timestamps, not the reaction itself
    # "cross_app/activity" - not in the spec's own shared taxonomy list;
    # mapped to NAVIGATION_ACTIVITY, the closest existing category (see
    # module docstring).
    (re.compile(r"RunningGameStore|ActivityTrackingStore|GameDisplayModeStorage|"
                r"EmbeddedActivities|everLaunchedActivities|"
                r"UserFlowAnalyticsStore-create_guild"), C.NAVIGATION_ACTIVITY, 0.8),
    (re.compile(r"attachment_?url|attachment_?name|embedded_?content|"
                r"lastImageSaveDirectory|ClipsStore|storageLocation", re.I), C.MEDIA, 0.6),
    (re.compile(r"cdn\.discordapp|media\.discordapp", re.I), C.MEDIA, 0.7),
    (re.compile(r"message_?sent|last_?edited|createdAt|editedTimestamp|call_?end", re.I),
     C.TIMESTAMP, 0.6),
)

_SLACK_RULES: tuple[tuple[re.Pattern, C, float], ...] = (
    (re.compile(r"member_?id|user_?id|team_?id|workspace_?id|user_?name|display_?name|"
                r"real_?name|full_?name|email", re.I), C.USER, 0.6),
    (re.compile(r"localConfig_v2"), C.USER, 0.7),  # holds name/url/user_id
    (re.compile(r"\b[UWTCBD]0[A-Z0-9]{7,}\b"), C.USER, 0.5),  # Slack id shape
    # canvas/collab stores - the spec's own text: these carry the user+team
    # id in their bootData, "-> identity".
    (re.compile(r"idb\.canvas-[UT]0[A-Z0-9]+|reduxPersistence|objectStore-[UT]0|"
                r"sundryStorage"), C.USER, 0.7),
    (re.compile(rf"{CONTENT_FIELD_PAT}|message|\btext\b|blocks|channel_?name|channel_?id|"
                r"thread_?ts|_?ThreadID|message_?status|recipient", re.I), C.CHAT_CONTENT, 0.6),
    (re.compile(r"persist-v1::.*::drafts|unifiedDrafts|withheldDraftIds|destinations"),
     C.DRAFT, 0.85),
    (re.compile(r"persist-v1::.*::customStatus|device-permissions-state"),
     C.ACCOUNT_SETTINGS, 0.85),
    (re.compile(r"persist-v1::.*::recentlyJoinedChannels|dateJoined|activitySession_"),
     C.NAVIGATION_ACTIVITY, 0.85),
    (re.compile(r"\bxox[bcp]-[\w-]+"), C.CREDENTIAL, 0.9),
    (re.compile(r"localConfig_v2"), C.CREDENTIAL, 0.6),  # also holds a token, alongside the USER hit above
    (re.compile(r"permanent_?link|file_?name|\btitle\b|filesize|created_?by", re.I), C.MEDIA, 0.6),
    (re.compile(r"files(?:-tmb)?\.slack\.com", re.I), C.MEDIA, 0.7),
    (re.compile(r"\bF0[A-Z0-9]{7,}\b"), C.MEDIA, 0.5),  # Slack file id shape
    (re.compile(r"message_?date|created|last_?read|updated|date_?joined", re.I), C.TIMESTAMP, 0.6),
)

_ROBLOX_RULES: tuple[tuple[re.Pattern, C, float], ...] = (
    (re.compile(r"user_?id|userId|username|display_?name|displayName|\bname\b", re.I),
     C.USER, 0.6),
    (re.compile(r"data\.userById\.id|verifiedEmail"), C.USER, 0.7),
    (re.compile(r"roblox-user-profiles"), C.USER, 0.7),  # SessionStorage cached profile -> identity
    (re.compile(r"\b\d{7,}\b"), C.USER, 0.5),  # Roblox id shape
    (re.compile(r"contentAgeRestriction|birth_?(?:month|day|year)|birthMonth|birthDay|"
                r"birthYear|ageGroup|estimatedAge|verifiedAge|countryName|allow\w+", re.I),
     C.ACCOUNT_SETTINGS, 0.7),
    (re.compile(r"\bsubject\b|sender_?user_?id|recipient_?user_?id|message_?id", re.I),
     C.CHAT_CONTENT, 0.6),
    (re.compile(r"place_?id|placeId|universe_?id|universeId|game_?instance_?id|join(?:ed)?|"
                r"rcc_?server|party_?id", re.I), C.NAVIGATION_ACTIVITY, 0.7),
    (re.compile(r"PresenceData|Roblox\.RealTime\.Events\.Notification|"
                r"Roblox\.RealTime\.Topic\.LocalStorage\.(?:Subscribe|Notification)"),
     C.NAVIGATION_ACTIVITY, 0.85),
    (re.compile(r"visit_?count|last_?visit_?time", re.I),
     C.NAVIGATION_ACTIVITY, 0.5),  # browser History fields
    (re.compile(r"RobloxCookies\.dat|appStorage\.json"), C.CREDENTIAL, 0.85),
    (re.compile(r"arkose|x-ark-arid|AUTHTOKEN", re.I), C.CREDENTIAL, 0.85),
    (re.compile(r"\.ROBLOSECURITY"), C.CREDENTIAL, 0.9),
    (re.compile(r"hbaDB|hba_keys"), C.CREDENTIAL, 0.7),  # device keys, low content value per spec
    # "locale/state" - not in the spec's own shared taxonomy list; mapped to
    # SYSTEM_DETAIL, which already has a generic "locale" key keyword at
    # weight 0.4 (rules.py) - consistent with existing precedent.
    (re.compile(r"RobloxLocaleCode|Roblox\.CrossTabCommunication"), C.SYSTEM_DETAIL, 0.7),
    (re.compile(r"\brobux\b|transaction|purchase|product_?id|\bprice\b|benefit_?id|"
                r"masked_credit_card", re.I), C.PAYMENT_INFO, 0.6),
    (re.compile(r"rbxcdn\.com", re.I), C.MEDIA, 0.7),
    (re.compile(r"created|updated|response_?received|joined|last_?online", re.I),
     C.TIMESTAMP, 0.6),
)

_PROFILES: tuple[tuple[str, tuple[tuple[re.Pattern, C, float], ...]], ...] = (
    ("discord", _DISCORD_RULES),
    ("slack", _SLACK_RULES),
    ("roblox", _ROBLOX_RULES),
)


def app_profile_hits(app: str, key: str, value: str) -> list["RuleHit"]:
    """[] for any app that isn't Discord/Slack/Roblox-shaped - same
    substring-on-app-name convention investigate/entities.py already uses
    for its Slack-only patterns, so an unrecognized/other app is a no-op,
    not an error."""
    # Deferred import: rules.py imports app_profile_hits from this module
    # (to union into its own regex_hits()), so a module-level import back
    # of RuleHit would be circular - both modules are fully loaded by the
    # time anything actually calls this function, so importing here (not
    # at module scope) resolves cleanly.
    from dd_lab.classify.rules import RuleHit

    app_l = app.lower()
    rules = next((r for name, r in _PROFILES if name in app_l), None)
    if rules is None:
        return []
    haystack = f"{key} {value}"
    hits: list[RuleHit] = []
    for pat, cat, weight in rules:
        m = pat.search(haystack)
        if m:
            hits.append(RuleHit(cat, weight, m.group(0)[:80]))
    return hits
