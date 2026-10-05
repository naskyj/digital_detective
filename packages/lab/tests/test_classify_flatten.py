"""JSON key-path flattening + content-bearing field-name gazetteer
(classifier_backend_spec.md) - the spec's single biggest validated cache-
recall fix (+7pts, precision held). rules.py::flatten_leaf_keys never
touches stored artifact granularity (jsonflat.py's own "objects stay
whole" design is untouched) - it's purely internal to the classifier."""

from __future__ import annotations

import pytest

from dd_core.enums import SourceType
from dd_lab.classify.ensemble import Classifier
from dd_lab.classify.rules import flatten_leaf_keys

C = Classifier()


# --------------------------------------------------------------------- flatten_leaf_keys


def test_flatten_nested_object():
    assert flatten_leaf_keys('{"author": {"username": "bob"}}') == ["author.username"]


def test_flatten_array_of_objects_uses_bracket_index():
    paths = flatten_leaf_keys('{"attachments": [{"url": "x"}, {"url": "y"}]}')
    assert paths == ["attachments[0].url", "attachments[1].url"]


def test_flatten_non_json_returns_empty():
    assert flatten_leaf_keys("just a plain string") == []


def test_flatten_unparseable_json_shaped_text_returns_empty():
    assert flatten_leaf_keys("{not: valid json,}") == []


def test_flatten_scalar_json_returns_empty():
    # "42" and "true" are valid JSON but not object/array-shaped - nothing
    # to flatten (and is_structurally_empty in gates.py already handles
    # the "empty JSON" case separately).
    assert flatten_leaf_keys("42") == []


# --------------------------------------------------------------------- content_field_hits via classify()


def test_nested_message_content_fires_chat_content_when_routed_as_cache():
    r = C.classify(
        "cache.record", '{"message": {"content": "lets meet at 5"}}', source_type=SourceType.CACHE
    )
    assert r.category == "CHAT_CONTENT"
    assert r.relevant is True


def test_nested_author_username_fires_contact_pii_when_routed_as_cache():
    r = C.classify(
        "cache.record", '{"author": {"username": "bobsmith99"}}', source_type=SourceType.CACHE
    )
    assert r.category == "CONTACT_PII"


def test_attachments_url_leaf_does_not_crash_and_flattening_runs():
    # "url" itself isn't in the content-field gazetteer (deliberately - the
    # gazetteer excludes generic field names), so this shouldn't force a
    # category on its own, but must not error either.
    r = C.classify(
        "cache.record", '{"attachments": [{"url": "https://cdn.example.com/x.png"}]}',
        source_type=SourceType.CACHE,
    )
    assert r.category  # produced a real result, didn't raise


def test_same_nested_json_with_local_storage_source_type_does_not_get_flatten_treatment():
    """Phase 4's routing holds: content_field_hits only runs for the
    content tier, never for a structured source_type."""
    value = '{"message": {"content": "lets meet at 5"}}'
    r = C.classify("local_storage.record", value, source_type=SourceType.LOCAL_STORAGE)
    assert r.category != "CHAT_CONTENT"


def test_no_source_type_does_not_get_flatten_treatment_either():
    """content_field_hits is gated on explicit _CONTENT_SOURCES membership,
    unlike the NER/model/entropy routing - source_type=None (every
    pre-phase-5 caller) does not get this brand-new feature."""
    value = '{"message": {"content": "lets meet at 5"}}'
    r = C.classify("cache.record", value)
    assert r.category != "CHAT_CONTENT"


def test_generic_excluded_field_names_dont_fire():
    """name|value|status|body are deliberately excluded from the gazetteer
    (spec's own finding: they fire on config JSON with no recall benefit)."""
    r = C.classify(
        "cache.record", '{"config": {"name": "theme", "value": "dark", "status": "ok"}}',
        source_type=SourceType.CACHE,
    )
    assert r.category != "CHAT_CONTENT"
    assert r.category != "CONTACT_PII"


# --------------------------------------------------------------------- word-boundary hardening


def test_the_word_password_in_prose_is_not_a_credential():
    """This test previously asserted the opposite — that "your password was
    reset successfully" is a CREDENTIAL — on the reasoning that the word
    appears as a real word rather than inside a longer token.

    Measured against the gold labels, that reading produced 59 false
    positives and no true positives. The matches were UI copy ("Windows Web
    Password Credential", "Resend Password Reset Email"), an image filename
    (`password-hero@2x.png`, 26 of the 59), help-article URLs and API names —
    text *about* passwords, never a password. A real password is a field
    value, so the rule now requires a populated field; the key-name gazetteer
    still carries `password` at 0.9 for the case where the artifact itself is
    the password."""
    r = C.classify("cache.record.note", "your password was reset successfully")
    assert r.category != "CREDENTIAL"


@pytest.mark.parametrize("value", [
    '{"username":"a","password":"hunter2"}',
    "password=s3cret",
    '"passwd": "abc123"',
    "password: letmein",
])
def test_a_populated_password_field_is_still_a_credential(value):
    """The constraint must not cost the case it exists for: a password field
    inside a larger JSON blob the parsers keep whole (Local State,
    Preferences — see the os_crypt note in rules.py)."""
    from dd_lab.classify.rules import C as Cat
    from dd_lab.classify.rules import regex_hits

    assert [h for h in regex_hits("mem.string", value) if h.category == Cat.CREDENTIAL], value


@pytest.mark.parametrize("value", ['{"password": ""}', '{"password": null}', '"passwd":'])
def test_an_empty_password_field_is_not_a_credential(value):
    """A field that is present and empty is not evidence of a password — the
    same reasoning the empty-value gate applies to a whole artifact."""
    from dd_lab.classify.rules import C as Cat
    from dd_lab.classify.rules import regex_hits

    assert not [h for h in regex_hits("mem.string", value) if h.category == Cat.CREDENTIAL], value


def test_value_keyword_does_not_fire_inside_an_unrelated_longer_word():
    # "password" must not fire merely because it's a substring of a longer
    # token - word-boundary matching, not substring-in.
    r = C.classify("cache.record.note", "unpasswordable nonsense text here")
    assert r.category != "CREDENTIAL"
