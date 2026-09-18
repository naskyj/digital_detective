"""Unit coverage for dd_lab/parsers/cache.py's locator-parsing helpers
(raw_key_from_locator, cache_offset_from_locator) - pure string logic, no
real evidence needed. Consolidated here so the artifact-browser API and the
labelling export both derive a cache row's grouping key/physical offset the
same way, instead of two independent reimplementations drifting apart."""

from __future__ import annotations

from dd_lab.parsers.cache import cache_offset_from_locator, raw_key_from_locator


def test_raw_key_from_locator_bare_key():
    assert raw_key_from_locator("1/0/https://discord.com/api/x") == "1/0/https://discord.com/api/x"


def test_raw_key_from_locator_with_resolved_offset():
    assert raw_key_from_locator("KEY @data_3:12345") == "KEY"


def test_raw_key_from_locator_body_with_resolved_offset():
    assert raw_key_from_locator("KEY @data_1:99::body") == "KEY"


def test_raw_key_from_locator_body_leaf_with_resolved_offset():
    assert raw_key_from_locator("KEY @data_1:99::body#some.leaf[0]") == "KEY"


def test_raw_key_from_locator_body_leaf_without_resolved_offset():
    """The case a naive `.split(' @', 1)[0]` gets wrong: no location was
    resolved, so there's no " @" to split on at all - the "::body#leaf"
    suffix has to be stripped by name, not by position relative to " @"."""
    assert raw_key_from_locator("KEY::body#some.leaf[0]") == "KEY"
    assert raw_key_from_locator("KEY::body") == "KEY"


def test_cache_offset_from_locator():
    assert cache_offset_from_locator("KEY @data_3:12345") == 12345
    assert cache_offset_from_locator("KEY @data_1:99::body#some.leaf[0]") == 99
    assert cache_offset_from_locator("KEY") is None  # no location resolved
    assert cache_offset_from_locator("KEY::body") is None  # no location resolved
