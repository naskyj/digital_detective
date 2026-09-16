"""ChromeCacheView-style cache-file inventory: `cache_dirs`, `list_cache_files`
and `read_cache_entry` against a real Chromium HTTP cache. Skips cleanly if no
supported profile with cache data is present on the dev machine."""

from __future__ import annotations

from pathlib import Path

import pytest

from dd_core.casedir import CaseLayout
from dd_lab.parsers.base import ParseContext
from dd_lab.parsers.cache import cache_dirs, list_cache_files, read_cache_entry

from tests.sample_case import build_sample_case

_CANDIDATES = {
    "vscode": Path.home() / "Library/Application Support/Code",
    "postman": Path.home() / "Library/Application Support/Postman",
}


def _available() -> tuple[str, Path] | None:
    for app, path in _CANDIDATES.items():
        if (path / "Cache" / "Cache_Data").is_dir() or (path / "Cache").is_dir():
            return app, path
    return None


pytestmark = pytest.mark.skipif(_available() is None, reason="no Electron profile with a cache is available")


@pytest.fixture(scope="module")
def cache_ctx(tmp_path_factory):
    app, profile = _available()
    root = tmp_path_factory.mktemp("cases")
    case_id = "case_cache_001"
    build_sample_case(root, case_id, app, profile, include=frozenset({"cache"}))
    layout = CaseLayout(root, case_id)
    return ParseContext(case_id=case_id, app=app, evidence_root=layout.evidence)


def test_cache_dirs_found(cache_ctx):
    dirs = cache_dirs(cache_ctx)
    assert dirs
    assert any("Cache" in d for d in dirs)


def test_list_cache_files_resolves_real_physical_files(cache_ctx):
    """The whole point of this feature: every listed entry should point at a
    real, existing file on disk under the evidence directory - not just a
    logical cache key - the same fact ChromeCacheView's "Filename" column
    shows, so the UI table can say exactly which file a value came from."""
    entries = list(list_cache_files(cache_ctx))
    assert entries
    with_file = [e for e in entries if e.physical_file]
    assert with_file, "no entry resolved a physical backing file"
    for e in with_file[:100]:
        assert (cache_ctx.evidence_root / e.cache_dir / e.physical_file).exists()

    # entry_id is a stable, unique, URL-safe handle per (app, dir, key)
    ids = [e.entry_id for e in entries]
    assert len(ids) == len(set(ids))


def test_list_cache_files_has_urls_and_sizes(cache_ctx):
    entries = list(list_cache_files(cache_ctx))
    with_url = [e for e in entries if e.url and e.url.startswith("http")]
    assert with_url
    sized = [e for e in entries if e.size_bytes is not None]
    assert sized


def test_read_cache_entry_returns_real_decoded_bytes(cache_ctx):
    entries = list(list_cache_files(cache_ctx))
    # size_bytes > 0, not just openable+content_type - an openable entry can
    # legitimately be empty (see the dedicated test below), which would make
    # this test's "real decoded bytes" assertion fail on a technicality.
    openable = next((e for e in entries if e.openable and e.content_type and (e.size_bytes or 0) > 0), None)
    assert openable, "no non-empty openable cache entry found to test against"
    body, content_type, filename_hint = read_cache_entry(cache_ctx, openable.cache_dir, openable.key)
    assert body
    assert content_type
    assert filename_hint


def test_read_cache_entry_empty_body_is_a_real_empty_response_not_missing(cache_ctx):
    """Found on real evidence: a cache entry can have a genuinely empty body
    (e.g. a 304/204) - `bodies = [b""]` from ccl_chromium_reader. Using
    truthiness (`if b`) to pick the body treats b"" the same as "no body at
    all" and incorrectly 404s an entry the listing itself marked openable.
    `is not None` is the right check - this locks that in."""
    entries = list(list_cache_files(cache_ctx))
    empty = next((e for e in entries if e.openable and (e.size_bytes or 0) == 0), None)
    if empty is None:
        pytest.skip("no empty-body openable cache entry found in this profile right now")
    body, _content_type, _hint = read_cache_entry(cache_ctx, empty.cache_dir, empty.key)
    assert body == b""
    assert body is not None


def test_read_cache_entry_unknown_key_is_none_not_a_crash(cache_ctx):
    dirs = cache_dirs(cache_ctx)
    body, _content_type, _hint = read_cache_entry(cache_ctx, dirs[0], "this-key-does-not-exist")
    assert body is None


def test_cache_parse_source_file_names_the_physical_file(cache_ctx):
    """parse() (which feeds the warehouse / generic artifact browser) should
    record the physical file too, not just the cache directory - the same
    traceability the dedicated cache-file view gives, surfaced in every
    cache-derived NormalizedArtifact."""
    from dd_lab.parsers.cache import parse

    artifacts = list(parse(cache_ctx))
    assert artifacts
    dirs = set(cache_dirs(cache_ctx))
    with_physical_name = [a for a in artifacts if a.source_file not in dirs]
    assert with_physical_name, "no cache artifact's source_file names a physical file"
