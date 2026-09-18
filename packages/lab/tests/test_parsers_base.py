"""Unit coverage for the shared parser plumbing in dd_lab/parsers/base.py -
in particular emit_value()/emit_parsed_value()'s key-path construction
(join_leaf_key), which every storage parser (Local Storage, Session Storage,
IndexedDB, Cache, Preferences) relies on.

The policy (see jsonflat.py's own docstring): an object is always one
artifact, whole, never split by key; only an array is exploded, one
artifact per entry, recursing into any array nested inside another array.
join_leaf_key is still the single place that must get "key.[0]" vs "key[0]"
right for that array-of-arrays case, so it only needs fixing once."""

from __future__ import annotations

import json
from pathlib import Path

from dd_core.enums import SourceType
from dd_lab.parsers.base import ParseContext, emit_parsed_value, emit_value, join_leaf_key


def _ctx() -> ParseContext:
    return ParseContext(case_id="c1", app="app", evidence_root=Path("/tmp/nonexistent-evidence-root"))


def _keys(raw_value: str) -> list[str]:
    return [
        a.key for a in emit_value(
            _ctx(), source_type=SourceType.CACHE, source_file="app/Cache/Cache_Data",
            locator="k1", key="cache.body", raw_value=raw_value,
        )
    ]


def _parsed_keys(parsed: object) -> list[str]:
    return [
        a.key for a in emit_parsed_value(
            _ctx(), source_type=SourceType.INDEXEDDB, source_file="app/IndexedDB/x.leveldb",
            locator="db/store/k1", key="idb.db.store", parsed=parsed,
        )
    ]


def test_top_level_array_key_has_no_stray_dot_before_the_bracket():
    """The bug: a top-level JSON array produced keys like "cache.body.[0]"
    instead of "cache.body[0]" - iter_leaves() already returns a leaf path
    that starts with its own bracket for array elements at the root."""
    keys = _keys(json.dumps(["a", "b", "c"]))
    assert keys == ["cache.body[0]", "cache.body[1]", "cache.body[2]"]
    assert not any(".[" in k for k in keys)


def test_array_inside_an_object_is_never_reached():
    """An object stops the walk right there, whole - a nested array buried
    inside one is never separately exploded, unlike an array nested inside
    ANOTHER array (test_array_of_arrays_key_path)."""
    keys = _keys(json.dumps({"friends": ["x", "y"]}))
    assert keys == ["cache.body"]


def test_object_is_one_artifact_whole_however_deep():
    artifacts = list(emit_value(
        _ctx(), source_type=SourceType.CACHE, source_file="app/Cache/Cache_Data",
        locator="k1", key="cache.body", raw_value=json.dumps({"user": {"name": "bob"}}),
    ))
    assert [a.key for a in artifacts] == ["cache.body"]
    assert json.loads(artifacts[0].value) == {"user": {"name": "bob"}}


def test_object_containing_raw_bytes_does_not_raise():
    """Real bug found on real evidence: dumping a whole object in one
    json.dumps() call (now that objects are never split by key) raises
    TypeError the moment ANY value nested inside it is a raw `bytes` object
    - which real IndexedDB records genuinely contain sometimes (blob
    references, binary payloads, ccl_chromium_reader hands them back
    already deserialized). Confirmed: this silently dropped thousands of a
    real Slack IndexedDB store's records once it hit one such record,
    because parse() catches per leveldb file, not per record."""
    artifacts = _parsed_keys({"name": "bob", "blob": b"\x00\x01raw bytes"})
    assert artifacts == ["idb.db.store"]  # didn't raise; one artifact, whole


def test_array_of_objects_is_one_artifact_per_entry_not_split_further():
    """Each array entry is its own artifact - including when that entry is
    itself an object, which then stops there whole (not split by key)."""
    artifacts = list(emit_value(
        _ctx(), source_type=SourceType.CACHE, source_file="app/Cache/Cache_Data",
        locator="k1", key="cache.body", raw_value=json.dumps([{"name": "bob"}, {"name": "eve"}]),
    ))
    assert [a.key for a in artifacts] == ["cache.body[0]", "cache.body[1]"]
    assert [json.loads(a.value) for a in artifacts] == [{"name": "bob"}, {"name": "eve"}]


def test_non_json_value_keeps_the_base_key_unchanged():
    keys = _keys("just a plain string, not json")
    assert keys == ["cache.body"]


def test_empty_json_container_keeps_the_base_key_unchanged():
    assert _keys("[]") == ["cache.body"]
    assert _keys("{}") == ["cache.body"]


def test_array_of_arrays_key_path():
    keys = _keys(json.dumps([["a", "b"], ["c", "d"]]))
    assert keys == ["cache.body[0][0]", "cache.body[0][1]", "cache.body[1][0]", "cache.body[1][1]"]


def test_object_with_bracket_like_keys_round_trips_intact():
    """A real case found on real evidence: VS Code's settings use "[python]"
    as a literal object key (language-scoped defaults) - the kind of key
    that used to read exactly like an array-index bug once dot-joined into
    a path. That ambiguity doesn't arise any more: the whole object
    (bracket-like keys included) is now one artifact, never path-joined at
    all, so it round-trips through json.dumps/json.loads untouched."""
    doc = {"configurationDefaults": {"[python]": {"editor": "x"}}, "[0]": "literal key, not an index"}
    artifacts = list(emit_value(
        _ctx(), source_type=SourceType.CACHE, source_file="app/Cache/Cache_Data",
        locator="k1", key="cache.body", raw_value=json.dumps(doc),
    ))
    assert [a.key for a in artifacts] == ["cache.body"]
    assert json.loads(artifacts[0].value) == doc


def test_join_leaf_key_matches_the_documented_rules():
    assert join_leaf_key("idb.db.store", "") == "idb.db.store"
    assert join_leaf_key("idb.db.store", "$") == "idb.db.store"
    assert join_leaf_key("idb.db.store", "name") == "idb.db.store.name"
    assert join_leaf_key("idb.db.store", "[0]") == "idb.db.store[0]"
    assert join_leaf_key("idb.db.store", '["[python]"]') == 'idb.db.store["[python]"]'


def test_emit_parsed_value_matches_emit_value_for_the_same_document():
    """The real bug this guards against: indexeddb.py used to build keys
    with its own inline "." join instead of going through emit_value() at
    all, so it kept the "key.[0]" bug even after emit_value() was fixed.
    Both entry points must now produce identical key shapes for the same
    document - emit_parsed_value() is what IndexedDB actually calls."""
    doc = {"a": [{"b": ["x", "y"]}], "[python]": {"editor": "x"}}
    from_value = [k.replace("cache.body", "idb.db.store") for k in _keys(json.dumps(doc))]
    from_parsed = _parsed_keys(doc)
    assert from_parsed == from_value


def test_emit_parsed_value_array_of_arrays():
    keys = _parsed_keys([["a", "b"], ["c", "d"]])
    assert keys == ["idb.db.store[0][0]", "idb.db.store[0][1]", "idb.db.store[1][0]", "idb.db.store[1][1]"]
