"""Unit coverage for the shared parser plumbing in dd_lab/parsers/base.py -
in particular emit_value()'s key-path construction, which every storage
parser (Local Storage, Session Storage, IndexedDB, Cache) relies on."""

from __future__ import annotations

import json
from pathlib import Path

from dd_core.enums import SourceType
from dd_lab.parsers.base import ParseContext, emit_value


def _ctx() -> ParseContext:
    return ParseContext(case_id="c1", app="app", evidence_root=Path("/tmp/nonexistent-evidence-root"))


def _keys(raw_value: str) -> list[str]:
    return [
        a.key for a in emit_value(
            _ctx(), source_type=SourceType.CACHE, source_file="app/Cache/Cache_Data",
            locator="k1", key="cache.body", raw_value=raw_value,
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
