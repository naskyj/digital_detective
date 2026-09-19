"""dd_lab/parsers/cache.py's labelling-export reader - one row per physical
cache entry (or per element of a JSON array-of-objects body), not one row
per metadata field. Fixes a real bug: a hand-labelling/content-matching
partner reconciled a prior export against a raw ChromeCacheView dump and
found 41,405 rows for 5,946 real entries - each entry's url/http_status/
content_type/request_time/response_time were each counted as a separate
"artifact" instead of columns describing the same row.

Two kinds of coverage here: `_classify_body_for_labeling` is pure logic
(no real evidence needed, same convention as test_cache_locator.py);
`iter_cache_entries_for_labeling` needs a real Chromium cache, same skip-if-
unavailable pattern as test_cache_files.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dd_core.casedir import CaseLayout
from dd_lab.parsers.base import ParseContext
from dd_lab.parsers.cache import (
    CacheFileEntry,
    CacheLabelRow,
    _binary_placeholder,
    _classify_body_for_labeling,
    _flatten_newlines,
    iter_cache_entries_for_labeling,
    list_cache_files,
)

from tests.sample_case import build_sample_case

# --------------------------------------------------------------------- _classify_body_for_labeling


def test_non_json_text_is_one_row():
    kind, values = _classify_body_for_labeling("function(){ return 1; }")
    assert kind == "text"
    assert values == ["function(){ return 1; }"]


def test_malformed_json_looking_text_falls_back_to_text():
    # starts with "{" but isn't valid JSON - try_parse_json returns None,
    # must degrade to "text", not raise.
    kind, values = _classify_body_for_labeling("{not: valid json,,,")
    assert kind == "text"


def test_single_json_object_is_one_row_whole():
    body = '{"id": 1, "name": "bob"}'
    kind, values = _classify_body_for_labeling(body)
    assert kind == "json_object"
    assert len(values) == 1
    assert json.loads(values[0]) == {"id": 1, "name": "bob"}


def test_nested_json_object_stays_one_row_even_with_an_embedded_array():
    # an object is one artifact, whole, however deep - matches jsonflat.py's
    # own convention (the general ingest rule), applied here to cache bodies.
    body = json.dumps({"user": {"id": 1}, "friends": ["a", "b", "c"]})
    kind, values = _classify_body_for_labeling(body)
    assert kind == "json_object"
    assert len(values) == 1
    assert json.loads(values[0])["friends"] == ["a", "b", "c"]


def test_top_level_array_of_objects_is_exploded_one_row_per_element():
    body = json.dumps([{"id": 1}, {"id": 2}, {"id": 3}])
    kind, values = _classify_body_for_labeling(body)
    assert kind == "json_array_element"
    assert [json.loads(v) for v in values] == [{"id": 1}, {"id": 2}, {"id": 3}]


def test_primitive_array_is_one_row_not_exploded():
    body = json.dumps(["a", "b", "c"])
    kind, values = _classify_body_for_labeling(body)
    assert kind == "json_array"
    assert json.loads(values[0]) == ["a", "b", "c"]


def test_number_array_is_one_row_not_exploded():
    body = json.dumps([1, 2, 3])
    kind, values = _classify_body_for_labeling(body)
    assert kind == "json_array"


def test_mixed_array_is_one_row_not_exploded():
    # not EVERY element is an object, so the array-of-objects rule doesn't
    # apply - stays one row, matching "everything else stays one row".
    body = json.dumps([{"id": 1}, "loose string", 2])
    kind, values = _classify_body_for_labeling(body)
    assert kind == "json_array"
    assert json.loads(values[0]) == [{"id": 1}, "loose string", 2]


def test_empty_array_is_one_row():
    kind, values = _classify_body_for_labeling("[]")
    assert kind == "json_array"
    assert values == ["[]"]


def test_array_of_arrays_is_one_row_not_exploded():
    # elements aren't objects (they're arrays), so array-of-objects doesn't apply.
    body = json.dumps([[1, 2], [3, 4]])
    kind, values = _classify_body_for_labeling(body)
    assert kind == "json_array"


def test_flatten_newlines_escapes_lf_cr_and_crlf():
    assert _flatten_newlines("line1\nline2") == "line1\\nline2"
    assert _flatten_newlines("line1\r\nline2") == "line1\\nline2"
    assert _flatten_newlines("line1\rline2") == "line1\\nline2"
    assert _flatten_newlines("no newlines here") == "no newlines here"


def test_binary_placeholder_names_content_type_and_size():
    assert _binary_placeholder("image/png", 252) == "[binary image/png, 252 bytes]"


def test_binary_placeholder_handles_missing_content_type():
    assert _binary_placeholder("", 10) == "[binary unknown, 10 bytes]"


def test_cache_label_row_column_set_matches_cache_files_csv_plus_value_columns():
    """The partner's own framing: "build it from cache_files.csv... and just
    add the body" - CacheLabelRow's metadata columns must be exactly
    CacheFileEntry's (minus `openable`, a live-browser-only concept), plus
    the columns needed to describe the body itself."""
    file_entry_cols = set(CacheFileEntry.model_fields) - {"openable"}
    label_row_cols = set(CacheLabelRow.model_fields)
    added = label_row_cols - file_entry_cols
    # `filename` (human-readable, guessed from the URL) is a deliberate
    # extra beyond CacheFileEntry's own columns - genuinely useful for a
    # labeller, absent from the live browser inventory only because the UI
    # derives it client-side from `url` instead.
    assert added == {"deleted", "body_kind", "array_index", "value", "filename"}
    assert file_entry_cols <= label_row_cols


def test_cache_label_row_deleted_defaults_false():
    # a placeholder today - cache entries have no real deletion-tracking yet.
    row = CacheLabelRow(
        entry_id="e", app="a", cache_kind="http_cache", cache_dir="d", key="k",
        url="https://x", filename="x", body_kind="empty",
    )
    assert row.deleted is False


# --------------------------------------------------------------------- iter_cache_entries_for_labeling

_CANDIDATES = {
    "vscode": Path.home() / "Library/Application Support/Code",
    "postman": Path.home() / "Library/Application Support/Postman",
}


def _available() -> tuple[str, Path] | None:
    for app, path in _CANDIDATES.items():
        if (path / "Cache" / "Cache_Data").is_dir() or (path / "Cache").is_dir():
            return app, path
    return None


# Applied per-test (not as a module-level `pytestmark`) so the pure
# _classify_body_for_labeling tests above stay real-evidence-independent and
# fast - only the tests below actually read/decode/decompress every real
# cache body, which is genuinely slow against a large real profile cache
# (confirmed: minutes, not seconds, once measured directly - see
# `export_rows` below). `slow` is excluded from the default test run by
# packages/lab/pyproject.toml's `addopts = "-m 'not slow'"`, same convention
# as test_api.py's/test_ingest_real.py's own @pytest.mark.slow tests.
_skip_if_unavailable = pytest.mark.skipif(
    _available() is None, reason="no Electron profile with a cache is available"
)


@pytest.fixture(scope="module")
def cache_ctx(tmp_path_factory):
    app, profile = _available()
    root = tmp_path_factory.mktemp("cases")
    case_id = "case_cache_label_001"
    build_sample_case(root, case_id, app, profile, include=frozenset({"cache"}))
    layout = CaseLayout(root, case_id)
    return ParseContext(case_id=case_id, app=app, evidence_root=layout.evidence)


@pytest.fixture(scope="module")
def export_rows(cache_ctx):
    """Computed ONCE for the whole module, not per test - unlike
    list_cache_files() (metadata only, cheap to recompute), this decodes,
    decompresses and JSON-parses every real cache body, which is expensive
    enough on a real, large profile cache that recomputing it in every test
    function turned this file into most of the suite's real wall-clock
    time (confirmed: the full suite went from ~80s to a 10+ minute hang
    before this fixture existed)."""
    return list(iter_cache_entries_for_labeling(cache_ctx))


@_skip_if_unavailable
@pytest.mark.slow
def test_entry_id_coverage_matches_the_physical_file_inventory(cache_ctx, export_rows):
    """The exact parity check the partner's feedback demands: the number of
    DISTINCT physical entries in the labelling export must equal
    list_cache_files()'s own count (the established, independently-verified
    "one row per physical cache entry" inventory) - even though a JSON
    array-of-objects body means MORE export rows than entries."""
    inventory_ids = {e.entry_id for e in list_cache_files(cache_ctx)}
    export_ids = {r.entry_id for r in export_rows}
    assert export_ids == inventory_ids


@_skip_if_unavailable
@pytest.mark.slow
def test_every_row_has_a_url_and_a_known_body_kind(export_rows):
    assert export_rows
    known = {"empty", "binary", "text", "json_object", "json_array", "json_array_element"}
    for r in export_rows:
        assert r.url
        assert r.body_kind in known


@_skip_if_unavailable
@pytest.mark.slow
def test_json_object_and_array_element_bodies_round_trip_as_valid_json(export_rows):
    json_rows = [r for r in export_rows if r.body_kind in ("json_object", "json_array", "json_array_element")]
    if not json_rows:
        pytest.skip("no JSON-bodied cache entry found in this profile right now")
    for r in json_rows[:200]:
        if r.value.endswith("]") and "…[+" in r.value:
            continue  # truncated - not expected to parse, by design
        json.loads(r.value)  # must not raise


@_skip_if_unavailable
@pytest.mark.slow
def test_binary_rows_get_a_placeholder_not_raw_bytes(export_rows):
    binary_rows = [r for r in export_rows if r.body_kind == "binary"]
    if not binary_rows:
        pytest.skip("no binary-bodied cache entry found in this profile right now")
    for r in binary_rows[:200]:
        # placeholder's byte count is the DECODED body length, which can
        # legitimately differ from `size_bytes` (the on-disk, possibly
        # still-compressed stream size reported without reading the body -
        # see _entry_size) - a compressed SVG is exactly such a case, found
        # while writing this test. Only check the placeholder's own shape.
        assert r.value.startswith("[binary ")
        assert r.value.endswith(" bytes]")


@_skip_if_unavailable
@pytest.mark.slow
def test_text_rows_are_capped_to_a_short_sample(export_rows):
    from dd_lab.parsers.cache import _TEXT_SAMPLE_CHARS

    text_rows = [r for r in export_rows if r.body_kind == "text"]
    if not text_rows:
        pytest.skip("no non-JSON text cache entry found in this profile right now")
    for r in text_rows:
        # newline-escaping (see test_no_row_ever_contains_a_raw_newline)
        # can expand the sample beyond the raw character cap - one \n per
        # embedded newline, so a generous multiple covers even a densely
        # multi-line source file's worth of escaping.
        assert len(r.value) <= _TEXT_SAMPLE_CHARS * 2 + 40


@_skip_if_unavailable
@pytest.mark.slow
def test_no_row_ever_contains_a_raw_newline(export_rows):
    """The actual real bug this guards against: a naive line-count (`wc -l`,
    or a non-RFC4180-compliant viewer) miscounted a correctly-parsed
    6,863-row export as 10,730+ "rows", entirely because of literal
    newlines embedded inside quoted text-body samples. Every value must be
    single-line from here on, regardless of how the file gets opened."""
    for r in export_rows:
        assert "\n" not in r.value
        assert "\r" not in r.value


@_skip_if_unavailable
@pytest.mark.slow
def test_array_index_is_only_set_for_json_array_element_rows(export_rows):
    for r in export_rows:
        if r.body_kind == "json_array_element":
            assert r.array_index is not None
        else:
            assert r.array_index is None


@_skip_if_unavailable
@pytest.mark.slow
def test_a_non_json_array_of_objects_entry_produces_exactly_one_row(export_rows):
    """Coverage parity implies this too: no entry_id appears more than once
    UNLESS its body_kind is json_array_element."""
    from collections import Counter

    counts = Counter(r.entry_id for r in export_rows)
    for entry_id, n in counts.items():
        if n > 1:
            assert all(r.body_kind == "json_array_element" for r in export_rows if r.entry_id == entry_id)
