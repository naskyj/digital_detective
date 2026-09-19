"""Classifier ensemble — synthetic, deterministic, fast."""

from __future__ import annotations

import pytest

from dd_lab.classify.ensemble import Classifier

C = Classifier()


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        # credentials
        ("cache.body.auth.token", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIx.abcdEF", "CREDENTIAL"),
        ("local_storage.app.password", "hunter2!", "CREDENTIAL"),
        ("idb.keyval.store.apiKey", "AIzaSyA1234567890abcdefghijklmnopqrstuv", "CREDENTIAL"),
        # payment
        ("cache.body.card.number", "4111111111111111", "PAYMENT_INFO"),
        ("cache.body.billing.last4", "4242", "PAYMENT_INFO"),
        ("prefs.payment.expMonth", "11", "PAYMENT_INFO"),
        # location
        ("cache.body.user.latitude", "51.5072", "LOCATION_INFO"),
        ("session.geo.coordinates", "51.5072,-0.1276", "LOCATION_INFO"),
        # network
        ("prefs.network.ipAddress", "192.168.1.44", "NETWORK_DETAIL"),
        ("local_state.os_crypt.macAddress", "a4:83:e7:1b:2c:3d", "NETWORK_DETAIL"),
        # user
        ("idb.messages.store.author.username", "kimkarter074", "USER"),
        ("cache.body.account.email", "suspect@example.com", "USER"),
        # chat
        ("idb.messages.store.content", "hey are you free to chat later", "CHAT_CONTENT"),
        ("cache.body.thread.message", "meet me after school", "CHAT_CONTENT"),
        # group
        ("idb.guilds.store.guildId", "8471293", "GROUP_DETAIL"),
        # system
        ("prefs.system.osVersion", "Windows-10-10.0.19045-SP0", "SYSTEM_DETAIL"),
        # timestamp
        ("cache.body.created_at", "2025-06-22T20:37:52Z", "TIMESTAMP"),
        # nothing interesting
        ("prefs.window.height", "800", "GENERAL"),
    ],
)
def test_expected_category(key, value, expected):
    result = C.classify(key, value)
    assert result.category == expected, (
        f"{key!r} -> got {result.category} "
        f"(regex={result.regex_vote} keyword={result.keyword_vote} ner={result.ner_vote})"
    )


def test_confidence_bounds_and_votes_recorded():
    r = C.classify("local_storage.app.password", "s3cr3t")
    assert 0.0 <= r.confidence <= 1.0
    assert r.keyword_vote == "CREDENTIAL"
    assert r.classifier_version.startswith("ensemble-")


def test_sensitive_category_wins_ties():
    # both a credential key hint and generic text; CREDENTIAL should win
    r = C.classify("cache.body.data.secret", "some description text here that is long")
    assert r.category in {"CREDENTIAL"}


def test_model_voter_is_optional_and_used_when_present():
    def fake_model(key, value):
        return __import__("dd_core.enums", fromlist=["ArtifactCategory"]).ArtifactCategory.GAME_DETAIL, 0.99

    clf = Classifier(model_voter=fake_model)
    r = clf.classify("some.random.key", "value")
    assert r.model_vote == "GAME_DETAIL"
    assert r.category == "GAME_DETAIL"


# --------------------------------------------------------------------- relevant/relevance_score
#
# A second, explicit output alongside category - "does this matter to the
# investigation", not "what kind of thing is this". Computed from the same
# per-voter scores category already uses (see ensemble.py::_combine), not a
# separate ensemble pass.


def test_relevant_true_for_a_real_category_hit():
    r = C.classify("local_storage.app.password", "hunter2")
    assert r.category == "CREDENTIAL"
    assert r.relevant is True
    assert r.relevance_score > 0.0


def test_relevant_false_when_nothing_fires_at_all():
    r = C.classify("some.opaque.key", "xyz")
    assert r.category == "GENERAL"
    assert r.relevant is False
    assert r.relevance_score == 0.0


def test_relevant_false_for_a_bare_boolean_or_short_number():
    # the entropy voter's explicit "not relevant" vote (see test_classify_entropy.py)
    r = C.classify("prefs.window.height", "800")
    assert r.category == "GENERAL"
    assert r.relevant is False
    assert r.relevance_score == 0.0


def test_relevance_score_is_never_positive_when_category_is_general():
    """However GENERAL was reached (empty scores, or GENERAL winning a real
    vote), relevance_score must be pinned to 0.0 - never a positive
    "relevance" for the category that means "nothing said this mattered"."""
    for key, value in [("some.opaque.key", "xyz"), ("prefs.window.height", "800"), ("k", "true")]:
        r = C.classify(key, value)
        if r.category == "GENERAL":
            assert r.relevance_score == 0.0
            assert r.relevant is False


def test_relevant_true_for_a_bare_timestamp_value_type():
    from dd_core.enums import ValueType

    r = C.classify("cache.body.some_field", "2025-06-22T20:37:52Z", ValueType.TIMESTAMP)
    assert r.category == "TIMESTAMP"
    assert r.relevant is True
    assert r.relevance_score > 0.0
