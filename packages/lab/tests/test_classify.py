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
