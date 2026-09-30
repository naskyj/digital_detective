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


@pytest.mark.parametrize(
    "value",
    [
        "noconnection@2x.png",   # retina-image naming - the real offender
        "chat@2x.png",
        "connection@3x.png",
        "icon@3x.svg",
        "sprite@2x.webp",
        "x@y.woff2",
        "a@b.js",
    ],
)
def test_image_and_asset_filenames_are_not_emails(value):
    """`name@2x.png` is syntactically identical to an email address, and the
    naive email pattern matched it. Measured on a real 675k-artifact case:
    24,255 artifacts matched `@<something>.<file-ext>`, 24,141 flagged
    relevant, and ALL of the top 100 cross-source "correlated identifiers"
    were bogus emails of this shape - which made the correlation view
    useless."""
    r = C.classify("mem.string", value, source_type="memory_string")
    assert r.regex_vote != "USER", f"{value!r} was treated as an email address"


@pytest.mark.parametrize(
    "value",
    [
        "suspect@example.com",
        "appro@openssl.org",
        "user@mail.co.uk",
        "First.Last+tag@sub.domain.io",
    ],
)
def test_real_emails_still_detected(value):
    """The fix above must not cost real email recall."""
    r = C.classify("cache.body.account.email", value)
    assert r.category == "USER"


@pytest.mark.parametrize(
    "value",
    [
        "PA50PA60PA61PB61PA61P",           # raw binary carved out of memory
        "PC81PD82PC82PC82PD81PE92PD81PC7",
        "AB12CDEFGHIJKLMNOP",              # right shape, wrong checksum
        "GB82WEST12345698765431",          # a real IBAN with one digit changed
    ],
)
def test_iban_shaped_strings_without_a_valid_checksum_are_not_payment_info(value):
    """Two letters, two digits, then alphanumerics is far too common a shape
    to use as evidence on its own: measured against the gold labels it
    produced 601 false positives and zero true positives, every one of them
    binary carved out of a memory dump. The mod-97 checksum is part of the
    IBAN standard, so validating it costs no genuine recall."""
    r = C.classify("mem.string", value, source_type="memory_string")
    assert r.regex_vote != "PAYMENT_INFO", f"{value!r} was treated as an IBAN"


@pytest.mark.parametrize(
    "value",
    [
        "GB82WEST12345698765432",   # the canonical example from ISO 13616
        "DE89370400440532013000",
        "FR1420041010050500013M02606",
        "NL91ABNA0417164300",
    ],
)
def test_real_ibans_still_detected(value):
    r = C.classify("cache.body.billing.iban", value)
    assert r.category == "PAYMENT_INFO"


def test_a_valid_iban_is_found_even_after_a_junk_candidate():
    """`search()` stops at the first match, so a checksum validator that
    rejects it must not abandon the pattern - memory-carved values routinely
    hold several shape-alike strings before the real one."""
    r = C.classify("mem.string", "PA50PA60PA61PB61PA61P GB82WEST12345698765432",
                   source_type="memory_string")
    assert r.regex_vote == "PAYMENT_INFO"


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


# --------------------------------------------------------------------- relevance_voter
#
# A DIFFERENT slot from model_voter above: model_voter predicts a category,
# relevance_voter predicts relevant/not_relevant directly and feeds
# relevance_score/relevant, never category. See ensemble.py's module
# docstring for the full design.


def test_relevance_voter_is_optional_and_contributes_to_relevance_score():
    def fake_relevance(key, value, source_type):
        return 0.9  # high P(relevant)

    clf = Classifier(relevance_voter=fake_relevance)
    r = clf.classify("some.random.key", "value")
    # rule_evidence is 0 here (nothing else fires on this key/value) -
    # noisy-OR with rule_evidence=0 reduces to exactly the model's own
    # score: 1 - (1-0)*(1-0.9) = 0.9.
    assert r.relevance_model_score == 0.9
    assert r.relevance_score == 0.9
    assert r.relevant is True
    # relevance_voter never influences category - it stays whatever the
    # rule-based voters decided (GENERAL here, since nothing else fired).
    assert r.category == "GENERAL"


def test_relevance_voter_can_make_something_relevant_that_rules_missed():
    """The core point of adding this voter: catch relevance the rule-based
    voters miss entirely. Category can legitimately stay GENERAL (rules
    found nothing) while relevance_score/relevant reflect the trained
    model's own opinion - a real change from the no-relevance-model
    behavior other tests in this file rely on (see
    test_relevance_score_is_never_positive_when_category_is_general's own
    comment: that invariant only holds when no relevance model is
    configured)."""

    def fake_relevance(key, value, source_type):
        return 0.95

    clf = Classifier(relevance_voter=fake_relevance)
    r = clf.classify("some.totally.opaque.key", "nothing rule-shaped here")
    assert r.category == "GENERAL"
    assert r.relevant is True
    assert r.relevance_score > 0.0


def test_relevance_combination_is_noisy_or_not_an_average():
    """Two independent moderate signals should reinforce each other, not
    dilute toward their average - the property that makes noisy-OR the
    right combination here instead of a plain weighted average (see
    ensemble.py::classify's own comment for the bug a weighted average
    had). A credential keyword hit gives some rule_evidence > 0; stack a
    moderately-confident fake relevance model on top and the combined
    score must exceed BOTH inputs individually."""

    def fake_relevance(key, value, source_type):
        return 0.4

    clf = Classifier(relevance_voter=fake_relevance)
    r = clf.classify("local_storage.app.password", "hunter2")
    rule_only = C.classify("local_storage.app.password", "hunter2").relevance_score
    assert r.relevance_score > rule_only
    assert r.relevance_score > 0.4
    assert r.relevance_score == round(1 - (1 - rule_only) * (1 - 0.4), 3)


def test_relevance_voter_runs_even_for_structured_source_types():
    """Unlike NER/model_voter/entropy, relevance_voter is NOT skipped for
    _STRUCTURED_SOURCES - confirmed with the project owner: it was trained
    across all 8 real source files (structured ones included), and
    relevance is the primary signal now, so skipping ~70% of rows the way
    the category-only fuzzy voters do would defeat the point."""
    from dd_core.enums import SourceType

    calls = []

    def fake_relevance(key, value, source_type):
        calls.append(source_type)
        return 0.8

    clf = Classifier(relevance_voter=fake_relevance)
    r = clf.classify("k", "v", source_type=SourceType.LOCAL_STORAGE)
    assert calls == [SourceType.LOCAL_STORAGE]
    assert r.relevance_model_score == 0.8


def test_classify_batch_matches_classify_one_row_at_a_time():
    """classify_batch() must produce identical results to calling classify()
    in a loop - it's a perf optimization (batches the relevance-model call),
    not a behavior change. Exercises a real mix: an empty-body-gated row
    (skips the relevance model entirely), a real category hit, and a
    GENERAL row - all through a fake relevance_voter with no .batch()
    method, so this also confirms the per-item fallback works."""

    def fake_relevance(key, value, source_type):
        return {"password": 0.9}.get(key, 0.2)

    clf = Classifier(relevance_voter=fake_relevance)
    rows = [
        {"key": "x", "value": "{}"},  # empty-body gate
        {"key": "local_storage.app.password", "value": "hunter2"},  # real hit
        {"key": "some.opaque.key", "value": "xyz"},  # GENERAL
    ]
    batched = clf.classify_batch(rows)
    singles = [clf.classify(r["key"], r["value"]) for r in rows]
    assert len(batched) == 3
    for b, s in zip(batched, singles, strict=True):
        assert b.category == s.category
        assert b.relevant == s.relevant
        assert b.relevance_score == s.relevance_score
        assert b.relevance_model_score == s.relevance_model_score


def test_classify_batch_uses_a_real_batch_method_when_the_voter_has_one():
    """When the configured relevance_voter exposes a `.batch()` method
    (duck-typed, matching OnnxRelevanceModel's real shape), classify_batch()
    must call it ONCE for every row needing scoring, not per-row - the
    actual point of this method existing."""
    calls = []

    class FakeBatchVoter:
        def __call__(self, key, value, source_type):
            raise AssertionError("classify_batch should never use the single-item call path here")

        def batch(self, items):
            calls.append(list(items))
            return [0.9 for _ in items]

    clf = Classifier(relevance_voter=FakeBatchVoter())
    rows = [{"key": f"k{i}", "value": f"v{i}"} for i in range(5)]
    results = clf.classify_batch(rows)
    assert len(calls) == 1  # one batched call, not five
    assert len(calls[0]) == 5
    assert all(r.relevance_model_score == 0.9 for r in results)


def test_classify_batch_skips_empty_body_gated_rows_in_the_model_batch():
    """An empty-body-gated row must not be sent to the relevance model at
    all - same as classify() (see the empty-body gate's own comment:
    "runs FIRST... overrides everything"), just verified at batch
    granularity too."""
    calls = []

    class FakeBatchVoter:
        def batch(self, items):
            calls.append(list(items))
            return [0.9 for _ in items]

    clf = Classifier(relevance_voter=FakeBatchVoter())
    rows = [
        {"key": "a", "value": "{}"},  # empty-body gated - must be excluded
        {"key": "b", "value": "real content here"},
    ]
    results = clf.classify_batch(rows)
    assert len(calls[0]) == 1  # only the non-gated row was sent to the model
    assert results[0].relevant is False and results[0].relevance_model_score is None
    assert results[1].relevance_model_score == 0.9


def test_classify_batch_with_no_relevance_voter_configured():
    rows = [{"key": "local_storage.app.password", "value": "hunter2"}, {"key": "k", "value": "v"}]
    results = C.classify_batch(rows)
    assert len(results) == 2
    assert all(r.relevance_model_score is None for r in results)
    assert results[0].relevant is True


# --------------------------------------------------------------------- model veto
#
# Found by measuring a real full run (675,463 artifacts): noisy-OR can only
# ever RAISE relevance, so a confident model "not relevant" was
# arithmetically incapable of lowering a saturated rule_evidence - 156,697
# of 213,016 artifacts flagged relevant were flagged despite the model
# scoring them under 0.01 (content_type headers, minified JS bundles).


#: A real minified webpack bundle body from the 675k case, trimmed. Rules
#: saturate on it (the word "content" appears in the module map) and cannot
#: tell it from a message - only the model can. The OTHER example the
#: comment above cites, `cache.content_type`, is no longer one: it is now
#: excluded at the rule level (see dd_core.patterns.CONTENT_FIELD_PAT), so
#: it can't stand in for "rules saturate" any more.
_JS_BUNDLE = (
    '"use strict";(globalThis.rspackChunkwebapp||=[]).push([["prompt-skill-add-popup"],'
    '{j4Lk(l){l.exports={skillList:1,content:2}}}]);'
)


def test_confident_model_negative_vetoes_a_saturated_rule_hit():
    def fake_relevance(key, value, source_type):
        return 0.000005  # what the real model actually returned for bodies like this

    clf = Classifier(relevance_voter=fake_relevance)
    # Exactly reproduces a real row from the 675k-artifact case: rules alone
    # saturate this (the actual bug - minified JS bundles sitting at the top
    # of the report as CHAT_CONTENT).
    args = ("cache.body", _JS_BUNDLE)
    kwargs = {"source_type": "cache", "app": "Slack"}
    rules_only = C.classify(*args, **kwargs)
    assert rules_only.relevance_score > 0.95 and rules_only.relevant is True

    # With the model's confident negative, it is no longer relevant.
    r = clf.classify(*args, **kwargs)
    assert r.relevance_score < 0.5
    assert r.relevant is False


def test_veto_does_not_touch_the_most_sensitive_categories():
    """The model has 0.86 recall - it misses ~14% of genuinely relevant
    artifacts - so it must never get the final word on the categories where
    a false negative is a serious forensic error. These are exactly the top
    3 of rules.py's CATEGORY_PRIORITY."""
    def fake_relevance(key, value, source_type):
        return 0.000005

    clf = Classifier(relevance_voter=fake_relevance)
    r = clf.classify("local_storage.app.password", "hunter2")
    assert r.category == "CREDENTIAL"
    assert r.relevant is True, "a confident model negative must not suppress a credential"
    assert r.relevance_score >= 0.5


def test_veto_only_fires_when_the_model_is_actually_confident():
    """A genuinely borderline model score (~0.3-0.5) is nowhere near the
    veto threshold and must leave the noisy-OR result alone."""
    def borderline(key, value, source_type):
        return 0.3

    clf = Classifier(relevance_voter=borderline)
    args, kwargs = ("cache.body", _JS_BUNDLE), {"source_type": "cache", "app": "Slack"}
    r = clf.classify(*args, **kwargs)
    baseline = C.classify(*args, **kwargs)
    # noisy-OR of the rule evidence with 0.3 - raised, never vetoed down
    assert r.relevance_score >= baseline.relevance_score
    assert r.relevant is True


def test_entropy_alone_never_makes_something_relevant():
    """Entropy is category-only (see _W_ENTROPY's comment in ensemble.py) -
    it must never be able to push relevance_score/relevant on its own, even
    though it can and does influence category."""
    # No rule/keyword/NER/relevance-model signal at all for this input;
    # if entropy fired and contributed to relevance, this would go positive.
    r = C.classify("some.opaque.key", "xyz")
    assert r.relevance_score == 0.0
    assert r.relevant is False


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
    """Holds for `C` (module-level fixture, no relevance_voter configured):
    category==GENERAL only happens when the rule-based voters found
    nothing at all, which also means rule_evidence is 0 - so with no
    relevance model to contribute otherwise, relevance_score is 0 too.
    This stops being a hard invariant once a real relevance_voter IS
    configured - see test_relevance_voter_can_make_something_relevant_that_rules_missed,
    which is the actual point of that voter."""
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


# --------------------------------------------------------------- app/source_type/deleted plumbing
# classifier_backend_spec.md phase 1: Classifier.classify() grows optional
# app/source_type/deleted kwargs. This phase is pure plumbing - none of them
# drive any voting/gating decision yet (that lands in later phases), so
# passing them must be a strict no-op against today's behavior.


def test_new_kwargs_are_accepted_and_dont_change_the_result():
    from dd_core.enums import SourceType

    key, value = "cache.body.auth.token", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIx.abcdEF"
    baseline = C.classify(key, value)
    with_kwargs = C.classify(
        key, value, app="discord", source_type=SourceType.CACHE, deleted=True
    )
    assert with_kwargs.category == baseline.category
    assert with_kwargs.confidence == baseline.confidence
    assert with_kwargs.relevant == baseline.relevant
    assert with_kwargs.relevance_score == baseline.relevance_score


def test_new_kwargs_accept_plain_strings_too():
    # pipeline.py's classify_warehouse() re-classify path reads source_type
    # back out of DuckDB as a plain string, not a SourceType instance - the
    # signature must tolerate that, not just the enum.
    r = C.classify("k", "v", app="roblox", source_type="cache", deleted=False)
    assert r.category  # didn't raise, produced a real result


# ------------------------------------------------- "content" as a header, not content
#
# `cache.content_type` is an HTTP entity header describing a fetched
# resource (`image/png`, `application/json; charset=utf-8`), and a bare
# "content" substring matched it from three independent directions: the
# _KEY_KEYWORDS tail, the Discord/Slack app profiles, and the JSON
# leaf-key gazetteer. 5,891 artifacts on the real 675k case, the single
# largest false grouping inside CHAT_CONTENT. The model vetoed them back
# down to relevance 0 - but only because those rows happened to be cheap
# enough to score; memory_string rows carry rules-only relevance, so a
# rule this broad has nothing standing behind it there.


@pytest.mark.parametrize(
    "key",
    [
        "cache.content_type",       # the real key, from the real case
        "cache.content-type",       # `\b` accepts the hyphen form, hence the explicit guard
        "cache.contentType",        # camelCase
        "cache.content_length",
        "cache.content_encoding",
        "cache.content_disposition",
    ],
)
def test_http_entity_header_keys_are_not_chat_content(key):
    r = C.classify(key, "image/png", source_type="cache", app="Slack")
    assert r.category != "CHAT_CONTENT"
    assert r.relevance_score == 0.0 and r.relevant is False


@pytest.mark.parametrize("mime", ["text/css", "text/html;charset=utf-8", "text/plain"])
def test_text_mime_types_are_not_chat_content(mime):
    """The Slack profile's `\btext\b` matched the VALUE of a content_type
    header, not just a `"text":` field name - 115 real artifacts. Same
    "header, not content" mistake as the key-side one above, one token
    over."""
    r = C.classify("cache.content_type", mime, source_type="cache", app="Slack")
    assert r.category != "CHAT_CONTENT"


def test_a_real_slack_text_field_still_classifies_as_chat_content():
    """The guard above is `(?!/)` precisely so this keeps working: a Slack
    message's `text` field is followed by a quote, never a slash."""
    r = C.classify("idb.messages.blocks", '{"type": "rich_text", "text": "see you at 5"}',
                   source_type="indexeddb", app="Slack")
    assert r.category == "CHAT_CONTENT"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("cache.body.message.content", "see you at 5"),
        ("indexeddb.messages.content", "hey there"),
        ("cache.body.embedded_content", "shared a photo"),
        ("cache.body.referenced_message.content", "Account sharing is against ToS."),
    ],
)
def test_real_content_fields_still_classify_as_chat_content(key, value):
    """The exclusion above must cost no recall on the thing it sits next to:
    an underscore BEFORE "content" is a separator, not a word continuation,
    so `embedded_content` stays a content field while `content_encoding`
    does not."""
    r = C.classify(key, value, source_type="cache", app="Discord")
    assert r.category == "CHAT_CONTENT"
    assert r.relevant is True


def test_content_inside_a_longer_word_is_not_an_app_profile_content_hit():
    """Discord's own config stores - `AdContentSeenStore`,
    `DismissibleContentFrameworkStore` - hold counters and seen-ids, not
    messages, and the app profile's bare "content" scored them 0.6 as chat
    content.

    Asserted against app_profile_hits rather than the whole ensemble on
    purpose: _KEY_KEYWORDS' own "content" (0.5) matches the NORMALIZED key
    tail, which has had every punctuation character stripped
    ("httpsdiscordappcomadcontentseenstore") and so has no word boundary
    left to exclude on - a documented property of that substrate, not
    something this fix can reach. On the real case those rows still moved
    off CHAT_CONTENT once the 0.6 hit was gone, because a competing voter
    outranked the surviving 0.5; that is luck, not a guarantee, so the
    narrow claim is the one worth pinning.
    """
    from dd_lab.classify.app_profiles import app_profile_hits

    for key in (
        "https://discordapp.com :: AdContentSeenStore",
        "https://discordapp.com :: DismissibleContentFrameworkStore",
    ):
        hits = app_profile_hits("discord", key, '{"_state": {"numberOfDCsShownToday": 2}}')
        assert not [h for h in hits if h.category == "CHAT_CONTENT"], key
