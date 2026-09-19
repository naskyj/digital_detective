"""dd_lab/classify/entropy.py — the entropy/type-shape voter. Pure logic,
no real evidence needed (same convention as test_classify.py). Every case
here was traced against test_classify.py's own existing parametrized cases
before shipping, specifically to catch shape collisions this voter must
NOT fire on (a credit card number, an IP address, a coordinate) - a real
false-positive found and fixed before this landed, not hypothetical."""

from __future__ import annotations

import pytest

from dd_core.enums import ArtifactCategory as C
from dd_lab.classify.entropy import entropy_vote


def test_abstains_on_short_values():
    assert entropy_vote("k", "hunter2!") is None  # below the min token length


def test_abstains_on_empty_value():
    assert entropy_vote("k", "") is None
    assert entropy_vote("k", "   ") is None


def test_high_entropy_token_votes_credential():
    cat, strength = entropy_vote("k", "aZ9k2Lm8pQ1xR7vN3wT6")
    assert cat == C.CREDENTIAL
    assert 0.0 < strength <= 1.0


def test_jwt_shaped_value_votes_credential():
    jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdEFghijKLmnoPQRstuv"
    cat, _ = entropy_vote("k", jwt)
    assert cat == C.CREDENTIAL


def test_pure_digit_string_never_votes_credential_even_when_long():
    # no letters at all - not the "random mixed alnum token" shape, however
    # long. A long digit run is handled (or not) by the regex/keyword
    # voters, not this one.
    assert entropy_vote("k", "4111111111111111") is None


def test_word_run_votes_chat_content():
    cat, strength = entropy_vote("k", "hey are you free to chat later tonight")
    assert cat == C.CHAT_CONTENT
    assert 0.0 < strength <= 1.0


def test_single_word_does_not_vote_chat_content():
    assert entropy_vote("k", "hello") is None


@pytest.mark.parametrize(
    "value",
    ["true", "FALSE", "yes", "no", "null", "none", "nil", "800", "-3", "4242", "1.2.3", "v10.0.19045"],
)
def test_boolean_enum_short_number_and_version_vote_general(value):
    cat, strength = entropy_vote("k", value)
    assert cat == C.GENERAL
    assert 0.0 < strength <= 1.0


@pytest.mark.parametrize(
    "value",
    [
        "4111111111111111",  # a card number - long digit run, not a short int
        "192.168.1.44",  # IPv4 - three dots/four segments, not a version's two
        "51.5072",  # a coordinate - one dot, not a version's exact two
        "-0.1276",
        "1473705902501531761",  # a Discord-snowflake-shaped long digit run
    ],
)
def test_evidentiary_looking_numbers_and_addresses_never_vote_general(value):
    """The real bug caught before this shipped: a naive 'looks like a bare
    number/dotted shape' rule would have pushed a card number, an IP, and a
    coordinate toward GENERAL - all three are genuinely evidentiary
    (PAYMENT_INFO/NETWORK_DETAIL/LOCATION_INFO elsewhere in the ensemble)
    and must never be nudged the other way by this tie-breaker."""
    result = entropy_vote("k", value)
    assert result is None or result[0] != C.GENERAL
