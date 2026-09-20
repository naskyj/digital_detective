"""Memory-derived extraction (classifier_backend_spec.md's "MEMORY-DERIVED
ADDITIONS" section) - URL-embedded ids, WebAuthn records, Windows toast
content. Source-agnostic (unlike app_profiles.py), only consulted for the
content-tier source types (see ensemble.py)."""

from __future__ import annotations

from dd_core.enums import SourceType
from dd_lab.classify.ensemble import Classifier
from dd_lab.memory.extract import memory_derived_hits

C = Classifier()

_DISCORD_ID = "1473705902501531761"
_DISCORD_ID2 = "1473705902501531762"


# --------------------------------------------------------------------- URL-embedded ids


def test_discord_avatar_url_is_user():
    value = f"https://cdn.discordapp.com/avatars/{_DISCORD_ID}/abcdef123456.png"
    r = C.classify("mem.string.record", value, source_type=SourceType.MEMORY_STRING)
    assert r.category == "USER"


def test_discord_attachment_url_is_media():
    value = f"https://cdn.discordapp.net/attachments/{_DISCORD_ID}/{_DISCORD_ID2}/evidence.png"
    r = C.classify("mem.string.record", value, source_type=SourceType.MEMORY_STRING)
    assert r.category == "MEDIA"


def test_slack_file_url_is_media():
    value = "https://files-tmb.slack.com/files-pri/T0-F0ABCDEFG-1234/thumb.jpg"
    r = C.classify("mem.string.record", value, source_type=SourceType.MEMORY_STRING)
    assert r.category == "MEDIA"


def test_roblox_users_url_is_user():
    r = C.classify("mem.string.record", "https://www.roblox.com/users/123456789/profile",
                    source_type=SourceType.MEMORY_STRING)
    assert r.category == "USER"


def test_roblox_games_url_is_navigation_activity():
    r = C.classify("mem.string.record", "https://www.roblox.com/games/9876543/some-game",
                    source_type=SourceType.MEMORY_STRING)
    assert r.category == "NAVIGATION_ACTIVITY"


# --------------------------------------------------------------------- WebAuthn / passkey


def test_webauthn_record_is_credential():
    value = '{"rpId":"discord.com","userId":"abc123","name":"bob@example.com","displayName":"bob"}'
    r = C.classify("mem.string.record", value, source_type=SourceType.MEMORY_STRING)
    assert r.category == "CREDENTIAL"


def test_single_webauthn_key_alone_is_not_distinctive_enough():
    # "name" alone is far too generic to fire on its own - requires 2+ of
    # the distinctive key names together.
    hits = memory_derived_hits('{"name": "just a name field"}')
    assert not any(h.category == "CREDENTIAL" for h in hits)


def test_webauthn_record_survives_truncation():
    # a memory carve cut off mid-record shouldn't lose the hit just
    # because it's not valid JSON anymore.
    value = '{"rpId":"discord.com","credentialId":"xyz789'  # no closing brace
    hits = memory_derived_hits(value)
    assert any(h.category == "CREDENTIAL" for h in hits)


# --------------------------------------------------------------------- Windows toast


def test_toast_text_is_notification():
    value = '<toast><visual><binding><text id="1">lets keep this between us</text></binding></visual></toast>'
    r = C.classify("mem.string.record", value, source_type=SourceType.MEMORY_STRING)
    assert r.category == "NOTIFICATION"
    assert any("lets keep this between us" in s for s in r.evidence_spans)


def test_toast_launch_target_is_notification():
    hits = memory_derived_hits('<toast launch="discord://channel/123">')
    assert any(h.category == "NOTIFICATION" for h in hits)


# --------------------------------------------------------------------- routing gate


def test_memory_derived_hits_not_applied_for_structured_source_type():
    # Without memory_derived_hits' Discord-avatar-specific rule, nothing
    # else classifies a bare CDN URL as USER (the generic bare-URL regex
    # hits NETWORK_DETAIL instead) - a precise enough signal that this
    # phase's gating actually held, without asserting on evidence_spans
    # text (the generic bare-URL hit's own span legitimately *contains*
    # "avatars" as a substring of the whole captured URL, regardless of
    # whether the dedicated rule ran).
    value = f"https://cdn.discordapp.com/avatars/{_DISCORD_ID}/abcdef123456.png"
    r = C.classify("local_storage.record", value, source_type=SourceType.LOCAL_STORAGE)
    assert r.category != "USER"


def test_memory_derived_hits_not_applied_when_no_source_type():
    value = f"https://cdn.discordapp.com/avatars/{_DISCORD_ID}/abcdef123456.png"
    r = C.classify("cache.record", value)
    assert r.category != "USER"
