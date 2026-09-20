"""Negative infra gazetteer + ID-in-plumbing carve-out, and the cross-app
Chromium fields (classifier_backend_spec.md). Neither is a voter - both
suppress/add to the result after (infra gazetteer) or alongside (Chromium
fields, which are ordinary regex hits) the normal vote."""

from __future__ import annotations

import pytest

from dd_lab.classify.ensemble import Classifier
from dd_lab.classify.infra_gazetteer import is_infra_noise

C = Classifier()

_DISCORD_ID = "1473705902501531761"  # 19-digit snowflake shape


# --------------------------------------------------------------------- is_infra_noise unit tests


def test_sdk_console_log_line_with_an_id_is_noise():
    assert is_infra_noise("cache.body.line", f"[SOCKET-MANAGER] connected uid={_DISCORD_ID}") is True


def test_telemetry_pixel_url_with_an_id_is_noise():
    assert is_infra_noise("cache.url", f"https://ecsv2.example.com/www/e.png?uid={_DISCORD_ID}") is True


def test_housekeeping_api_route_with_an_id_is_noise():
    assert is_infra_noise("cache.url", f"https://discord.com/api/v9/users/{_DISCORD_ID}/profile") is True


def test_plumbing_pattern_alone_with_no_id_is_not_flagged_noise():
    # nothing here for the carve-out to act on - the regular voters already
    # score this as GENERAL on their own, no gate needed.
    assert is_infra_noise("cache.body.line", "[SOCKET-MANAGER] connected") is False


def test_id_alone_with_no_plumbing_pattern_is_not_noise():
    assert is_infra_noise("cache.body.user_id", _DISCORD_ID) is False


@pytest.mark.parametrize(
    "value",
    [
        f"https://cdn.discordapp.com/attachments/{_DISCORD_ID}/999/photo.png",
        f'{{"content": "hey", "author_id": "{_DISCORD_ID}"}}',
        f"https://files-tmb.slack.com/file/{_DISCORD_ID}",
    ],
)
def test_content_exception_still_lets_real_evidence_through(value):
    # /avatars/<17-20 digit id> is a plumbing hit (profile/asset CDN) but
    # also a content-exception match - the exception must win.
    assert is_infra_noise("cache.url", value) is False


def test_avatar_url_with_a_real_user_id_is_the_content_exception_not_plumbing():
    value = f"https://cdn.discordapp.com/avatars/{_DISCORD_ID}/abcdef.png"
    # /avatars/ alone is plumbing, but /avatars/\d{17,20} is explicitly
    # carved out as a real user avatar reference.
    assert is_infra_noise("cache.url", value) is False


# --------------------------------------------------------------------- end-to-end through classify()


def test_classify_suppresses_a_noisy_sdk_log_line_even_with_an_id():
    r = C.classify("cache.body.log_line", f"[HUDDLE-SDK] session ready for {_DISCORD_ID}")
    assert r.category == "GENERAL"
    assert r.relevant is False
    assert r.relevance_score == 0.0


def test_classify_still_flags_a_genuine_attachment_url():
    r = C.classify(
        "cache.url", f"https://cdn.discordapp.com/attachments/{_DISCORD_ID}/999/evidence.png"
    )
    assert r.category != "GENERAL"


# --------------------------------------------------------------------- cross-app Chromium fields


def test_local_state_os_crypt_encrypted_key_is_credential():
    value = '{"os_crypt": {"encrypted_key": "QVBQMY9z...=="}, "other": "stuff"}'
    r = C.classify("local_state", value)
    assert r.category == "CREDENTIAL"


def test_preferences_device_id_salt_is_system_detail():
    value = '{"electron": {"media": {"device_id_salt": "abc123salt"}}}'
    r = C.classify("preferences", value)
    assert r.category == "SYSTEM_DETAIL"


@pytest.mark.parametrize(
    "cookie_name",
    ["ROBLOSECURITY", "ARID", "__dcfduid", "persona_device_id", "_ga", "_ga_ABC123XYZ", "__stripe_mid"],
)
def test_cookie_name_allowlist_is_credential(cookie_name):
    r = C.classify(f"cookie.discord.com.{cookie_name}", "some-opaque-encrypted-value")
    assert r.category == "CREDENTIAL"


def test_cookie_name_not_on_the_allowlist_is_not_flagged():
    # __cf_bm/cf_clearance/__cflb/hmt_id are the spec's explicit
    # non-evidence list (CDN/bot-management, pure config) - simply absent
    # from the allowlist, so they fall through to whatever (if anything)
    # else fires, never forced to CREDENTIAL.
    r = C.classify("cookie.discord.com.__cf_bm", "some-opaque-value")
    assert r.category != "CREDENTIAL"


def test_ga_allowlist_entry_is_anchored_not_a_bare_substring():
    """_ga is short enough that a naive substring-in-key search would
    false-positive on plenty of unrelated keys (e.g. "organization"
    contains "ga") - the cookie-name match is anchored to the actual
    trailing cookie-name segment, not a fuzzy substring anywhere in key."""
    r = C.classify("preferences.organization_settings", '{"unrelated": "value"}')
    assert r.category != "CREDENTIAL"
