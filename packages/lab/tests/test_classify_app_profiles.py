"""Per-app profiles (classifier_backend_spec.md's DISCORD/SLACK/ROBLOX
sections) - real LevelDB/IndexedDB key names and value shapes, scoped to
one app. Hits are unioned into regex_hits() (see rules.py), so these go
through Classifier.classify() like any other structural match."""

from __future__ import annotations

import pytest

from dd_lab.classify.app_profiles import app_profile_hits
from dd_lab.classify.ensemble import Classifier

C = Classifier()


# --------------------------------------------------------------------- Discord

def test_discord_draft_store_key_is_draft():
    r = C.classify("local_storage.DraftStore", '{"draft": "hey wanna"}', app="discord")
    assert r.category == "DRAFT"


def test_discord_multi_account_store_key_is_user():
    r = C.classify("local_storage.MultiAccountStore", '{"userId": "123"}', app="discord")
    assert r.category == "USER"


def test_discord_snowflake_value_is_user():
    r = C.classify("local_storage.some_id", "1473705902501531761", app="discord")
    assert r.category == "USER"


def test_discord_cdn_url_is_media():
    # deliberately not using "avatar" in the key - the spec's own Discord
    # identity bucket lists "avatar" as an identity signal (a profile
    # picture reference names whose profile it is), so a key containing
    # it correctly pulls toward USER too; this test isolates the CDN-URL
    # value match on its own.
    r = C.classify(
        "cache.record.thumbnail", "https://cdn.discordapp.com/avatars/123/abc.png", app="discord"
    )
    assert r.category == "MEDIA"


def test_discord_theme_store_is_account_settings():
    r = C.classify("local_storage.ThemeStore", '{"theme": "dark"}', app="discord")
    assert r.category == "ACCOUNT_SETTINGS"


# --------------------------------------------------------------------- Slack

def test_slack_custom_status_key_is_account_settings():
    r = C.classify(
        "local_storage.persist-v1::abc::customStatus", '{"text": "brb", "emoji": ":coffee:"}',
        app="slack",
    )
    assert r.category == "ACCOUNT_SETTINGS"


def test_slack_token_value_is_credential():
    r = C.classify("local_storage.localConfig_v2", "xoxc-1234567890-abcdefghij", app="slack")
    assert r.category == "CREDENTIAL"


def test_slack_id_shape_value_is_user():
    r = C.classify("indexeddb.some.field", "U0C1DHP5PPX", app="slack")
    assert r.category == "USER"


def test_slack_files_url_is_media():
    r = C.classify("cache.record.link", "https://files.slack.com/files-pri/T0-F0ABCDEFG/photo.png",
                    app="slack")
    assert r.category == "MEDIA"


# --------------------------------------------------------------------- Roblox

def test_roblox_robux_key_is_payment_info():
    r = C.classify("cache.record.commerce", '{"robux": 500, "transaction": "purchase"}', app="roblox")
    assert r.category == "PAYMENT_INFO"


@pytest.mark.parametrize("value", [
    "robux@2x.png",                                     # an icon filename
    "the kernel TRANSACTION manager had to abort",      # a Windows error string
    "PURCHASES are Currently Disabled",                 # UI copy
    "best price guaranteed",                            # marketing prose
])
def test_roblox_commerce_words_in_prose_are_not_payment_info(value):
    """The generic commerce words are field names in Roblox's own stores, so
    they only count as evidence when they appear as one. As bare substrings
    they produced 1,226 false positives and no true positives against the
    gold labels - every hit was an icon name, an OS error string or UI copy."""
    hits = app_profile_hits("roblox", "cache.record.body", value)
    assert not [h for h in hits if h.category == "PAYMENT_INFO"], value


@pytest.mark.parametrize("value", [
    '{"robux": 500}', "robux=500", '{"price": 9.99}', '{"purchases": []}',
    '{"masked_credit_card": "****1234"}',   # distinctive: no separator needed
    "benefit_id",
])
def test_roblox_commerce_fields_are_payment_info(value):
    hits = app_profile_hits("roblox", "cache.record.commerce", value)
    assert [h for h in hits if h.category == "PAYMENT_INFO"], value


def test_roblox_security_cookie_value_is_credential():
    r = C.classify("cookie.roblox.com..ROBLOSECURITY", ".ROBLOSECURITYv2_ABCDEF", app="roblox")
    assert r.category == "CREDENTIAL"


def test_roblox_birth_month_key_is_account_settings():
    r = C.classify("cache.record.account", '{"birthMonth": 6}', app="roblox")
    assert r.category == "ACCOUNT_SETTINGS"


def test_roblox_id_shape_value_is_user():
    r = C.classify("indexeddb.some.field", "123456789", app="roblox")
    assert r.category == "USER"


def test_roblox_locale_key_is_system_detail():
    r = C.classify("local_storage.RobloxLocaleCode", "en-us", app="roblox")
    assert r.category == "SYSTEM_DETAIL"


# --------------------------------------------------------------------- cross-app leakage guard


def test_discord_draft_store_key_does_not_fire_for_slack():
    """The exact same Discord-shaped key, called with app='slack', must not
    fire the Discord-only rule - mirrors investigate/entities.py's existing
    'only trusted with the matching app' guard."""
    hits = app_profile_hits("slack", "local_storage.DraftStore", '{"draft": "hey"}')
    assert not any(h.category == "DRAFT" for h in hits)


def test_roblox_robux_key_does_not_fire_for_discord():
    hits = app_profile_hits("discord", "cache.record.commerce", '{"robux": 500}')
    assert not any(h.category == "PAYMENT_INFO" for h in hits)


@pytest.mark.parametrize("app", ["", "unknown_app", "vscode"])
def test_unrecognized_app_returns_no_hits(app):
    assert app_profile_hits(app, "local_storage.DraftStore", '{"draft": "hey"}') == []


def test_app_profile_hits_matches_on_app_name_substring_not_exact_equality():
    # real app identifiers in this project are sanitised/suffixed
    # (e.g. "discord", "Roblox__WebView2_via_UniversalApp_WebView2") - the
    # match must be substring-based like entities.py's own convention.
    hits = app_profile_hits("Roblox__WebView2_via_UniversalApp_WebView2", "cache.record.commerce",
                             '{"robux": 500}')
    assert any(h.category == "PAYMENT_INFO" for h in hits)


def test_app_profile_content_rules_use_the_shared_content_pattern():
    """Same anti-drift pin as test_investigate's shared-email one: "content"
    is matched in three places (here, _KEY_KEYWORDS' tail, and the JSON
    leaf-key gazetteer), and the last time a rule lived in two copies a fix
    landed in one and silently missed the other. Both app profiles that
    carry a content rule must build it from dd_core.patterns."""
    from dd_core.enums import ArtifactCategory
    from dd_core.patterns import CONTENT_FIELD_PAT
    from dd_lab.classify.app_profiles import _DISCORD_RULES, _SLACK_RULES

    for name, rules in (("discord", _DISCORD_RULES), ("slack", _SLACK_RULES)):
        chat = [p for p, cat, _w in rules if cat == ArtifactCategory.CHAT_CONTENT]
        assert chat, name
        assert any(CONTENT_FIELD_PAT in p.pattern for p in chat), name
