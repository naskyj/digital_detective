"""Per-source-type voter routing (classifier_backend_spec.md): structured
key/value sources (LevelDB/IndexedDB/cookies/prefs/local_state/memory
process metadata) skip the NER/model/entropy voters outright - the spec's
own back-test found those voters gained 0 recall and cost precision there.
Content-dependent sources (cache/memory strings/app_data) keep every
voter, same as before this phase existed."""

from __future__ import annotations

from dd_core.enums import SourceType
from dd_lab.classify.ensemble import Classifier

C = Classifier()

# A value that reliably fires the entropy voter's "looks like a sentence"
# heuristic (4+ word-shaped runs in a row, no other signal) - see
# test_classify_entropy.py for the same shape.
_SENTENCE = "we should meet up later this evening okay"


def test_structured_source_skips_ner_model_entropy_voters():
    r = C.classify("cache.body.notes", _SENTENCE, source_type=SourceType.LOCAL_STORAGE)
    assert r.ner_vote is None
    assert r.model_vote is None
    assert r.entropy_vote is None


def test_content_source_still_runs_the_entropy_voter():
    r = C.classify("cache.body.notes", _SENTENCE, source_type=SourceType.CACHE)
    assert r.entropy_vote == "CHAT_CONTENT"


def test_no_source_type_defaults_to_full_voter_set():
    """Every pre-phase-4 caller never passed source_type at all - must keep
    behaving exactly like the content tier (full voter set), not silently
    get routed into the structured tier and lose recall."""
    r = C.classify("cache.body.notes", _SENTENCE)
    assert r.entropy_vote == "CHAT_CONTENT"


def test_unrecognized_source_type_string_falls_back_to_full_voter_set():
    r = C.classify("cache.body.notes", _SENTENCE, source_type="not_a_real_source_type")
    assert r.entropy_vote == "CHAT_CONTENT"


def test_source_type_accepts_a_plain_string_not_just_the_enum():
    # classify_warehouse's re-classify path reads source_type back out of
    # DuckDB as a plain string.
    r = C.classify("cache.body.notes", _SENTENCE, source_type="local_storage")
    assert r.entropy_vote is None


def test_regex_and_keyword_voters_are_unaffected_by_routing():
    """The schema layer (regex/keyword) isn't touched by routing - only
    NER/model/entropy are skipped for structured sources."""
    key, value = "cache.body.auth.token", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIx.abcdEF"
    structured = C.classify(key, value, source_type=SourceType.LOCAL_STORAGE)
    content = C.classify(key, value, source_type=SourceType.CACHE)
    assert structured.category == content.category == "CREDENTIAL"
    assert structured.regex_vote == content.regex_vote == "CREDENTIAL"


def test_memory_process_is_routed_as_structured():
    r = C.classify("mem.proc.name", _SENTENCE, source_type=SourceType.MEMORY_PROCESS)
    assert r.entropy_vote is None


def test_app_data_is_routed_as_content():
    r = C.classify("appdata.native/log.line0", _SENTENCE, source_type=SourceType.APP_DATA)
    assert r.entropy_vote == "CHAT_CONTENT"
