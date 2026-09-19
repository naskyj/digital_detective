"""Gates (classifier_backend_spec.md): the empty-body pre-gate and the
deleted/edited relevance boost - neither is a voter, both override the
ensemble's own math (see gates.py's module docstring)."""

from __future__ import annotations

import pytest

from dd_lab.classify.ensemble import Classifier
from dd_lab.classify.gates import boosted_relevance, is_structurally_empty

C = Classifier()


# --------------------------------------------------------------- empty-body pre-gate

_EMPTY_VALUES = [
    "",
    "   ",
    "{}",
    "[]",
    "null",
    "NULL",
    "<Undefined>",
    '{"data": []}',
    '{"consents": [], "nextCursor": null}',
    '{"selectedChannelIds": {}, "knownThreadIds": []}',
    '{"a": {"b": [null, {}]}}',  # nested-empty still counts as empty
]


@pytest.mark.parametrize("value", _EMPTY_VALUES)
def test_is_structurally_empty_true(value):
    assert is_structurally_empty(value) is True


_NON_EMPTY_VALUES = [
    "hello",
    "0",  # a real scalar, not one of the recognised empty literals
    "false",
    '{"data": ["x"]}',
    '{"consents": [], "nextCursor": "abc123"}',
    '{"data": [{"id": "abc"}]}',  # a real leaf value inside the array makes it non-empty
]


@pytest.mark.parametrize("value", _NON_EMPTY_VALUES)
def test_is_structurally_empty_false(value):
    assert is_structurally_empty(value) is False


def test_empty_body_gate_overrides_a_real_id_in_the_key_or_url():
    """The spec's own scenario: an API response's URL carries a real case
    ID, but the decoded body has no content - must still be GENERAL, not
    relevant, regardless of the ID. The gate only looks at `value` (the
    decoded body); the ID here lives in `key`, which regex_hits/keyword_hits
    would otherwise happily match - proving the gate really does run before
    (and instead of) those voters, not just on values that happen to have
    no ID anywhere nearby."""
    r = C.classify(
        "cache.url=https://api.example.com/case/123456789012345678", '{"data": [], "nextCursor": null}'
    )
    assert r.category == "GENERAL"
    assert r.relevant is False
    assert r.relevance_score == 0.0


def test_empty_body_gate_skips_every_voter():
    r = C.classify("cache.body.token", "{}")
    assert r.regex_vote is None
    assert r.keyword_vote is None
    assert r.ner_vote is None
    assert r.entropy_vote is None
    assert r.evidence_spans == []


def test_a_non_empty_body_is_unaffected_by_the_gate():
    r = C.classify("cache.body.auth.token", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIx.abcdEF")
    assert r.category == "CREDENTIAL"


# --------------------------------------------------------------- deleted/edited boost


def test_boosted_relevance_bumps_when_deleted_and_not_general():
    assert boosted_relevance(0.5, deleted=True, is_general=False) == pytest.approx(0.65)


def test_boosted_relevance_caps_at_one():
    assert boosted_relevance(0.95, deleted=True, is_general=False) == 1.0


def test_boosted_relevance_no_change_when_not_deleted():
    assert boosted_relevance(0.5, deleted=False, is_general=False) == 0.5


def test_boosted_relevance_no_change_when_general_even_if_deleted():
    """Deleting a GENERAL/irrelevant record isn't itself interesting -
    relevance stays pinned to 0.0 for GENERAL regardless of deleted."""
    assert boosted_relevance(0.0, deleted=True, is_general=True) == 0.0


def test_classify_applies_the_deleted_boost_end_to_end():
    key, value = "local_storage.app.password", "hunter2!"
    live = C.classify(key, value, deleted=False)
    deleted = C.classify(key, value, deleted=True)
    assert deleted.relevance_score > live.relevance_score
    assert deleted.relevance_score == pytest.approx(min(1.0, live.relevance_score + 0.15), abs=1e-6)
