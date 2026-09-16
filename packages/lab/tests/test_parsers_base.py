"""Unit coverage for the shared parser plumbing in dd_lab/parsers/base.py -
in particular emit_value()/emit_parsed_value()'s key-path construction
(join_leaf_key), which every storage parser (Local Storage, Session Storage,
IndexedDB, Cache, Preferences) relies on - the single place that must get
"key.[0]" vs "key[0]" right, so it only needs fixing once."""

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


def test_nested_array_key_is_unaffected():
    keys = _keys(json.dumps({"friends": ["x", "y"]}))
    assert keys == ["cache.body.friends[0]", "cache.body.friends[1]"]


def test_plain_object_keys_still_dotted():
    keys = _keys(json.dumps({"user": {"name": "bob"}}))
    assert keys == ["cache.body.user.name"]


def test_non_json_value_keeps_the_base_key_unchanged():
    keys = _keys("just a plain string, not json")
    assert keys == ["cache.body"]


def test_empty_json_container_keeps_the_base_key_unchanged():
    assert _keys("[]") == ["cache.body"]
    assert _keys("{}") == ["cache.body"]


def test_array_of_arrays_key_path():
    keys = _keys(json.dumps([["a", "b"], ["c", "d"]]))
    assert keys == ["cache.body[0][0]", "cache.body[0][1]", "cache.body[1][0]", "cache.body[1][1]"]


def test_literal_bracket_object_key_is_quoted_not_dotted():
    """A real case found on real evidence: VS Code's settings use "[python]"
    as a literal object key (language-scoped defaults). Dot-joining it
    ("configurationDefaults.[python]") reads exactly like the array-index
    bug even though it isn't one - quoting it removes the ambiguity."""
    keys = _keys(json.dumps({"configurationDefaults": {"[python]": {"editor": "x"}}}))
    assert keys == ['cache.body.configurationDefaults["[python]"].editor']
    assert not any(".[" in k for k in keys)


def test_object_key_that_is_literally_an_index_string_is_distinguishable():
    """The genuinely pathological case: a dict key whose string value is
    "[0]" - identical in appearance to a real array index once flattened
    naively. Quoting makes it distinguishable from cache.body[0][0]."""
    keys = _keys(json.dumps([{"[0]": "not an index, a literal key"}]))
    assert keys == ['cache.body[0]["[0]"]']
    assert keys != _keys(json.dumps([["not an index, a literal key"]]))  # a genuine 2D array, for contrast


def test_object_keys_with_special_characters_are_quoted():
    keys = _keys(json.dumps({"weird key with spaces": 1, "dot.in.key": 2, "": 3}))
    assert keys == [
        'cache.body["weird key with spaces"]',
        'cache.body["dot.in.key"]',
        'cache.body[""]',
    ]


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
