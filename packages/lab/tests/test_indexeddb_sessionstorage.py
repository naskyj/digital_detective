"""IndexedDB and Session Storage parsers against real Electron profiles on
the dev machine. Skips cleanly if none is available.

Covers a real bug found on real evidence: indexeddb.py had its own inline
copy of the key-path join that predates the emit_value() fix for
"key.[0]"-style keys - it never went through emit_value() at all, so fixing
emit_value() alone didn't fix IndexedDB. Now both go through the same
shared dd_lab.parsers.base.join_leaf_key()/emit_parsed_value().
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from dd_lab.parsers import indexeddb, sessionstorage
from dd_lab.parsers.base import ParseContext

_IDB_CANDIDATES = [
    Path.home() / "Library/Application Support/Cursor",
    Path.home() / "Library/Application Support/Postman",
]
_SS_CANDIDATES = [
    Path.home() / "Library/Application Support/Code",
    Path.home() / "Library/Application Support/Cursor",
]

_DOT_BRACKET = re.compile(r"\.\[")


def _idb_profile() -> Path | None:
    for p in _IDB_CANDIDATES:
        if (p / "IndexedDB").is_dir():
            return p
    return None


def _ss_profile() -> Path | None:
    for p in _SS_CANDIDATES:
        if (p / "Session Storage").is_dir():
            return p
    return None


def _ctx_for(profile: Path) -> ParseContext:
    ctx = ParseContext(case_id="c1", app="app", evidence_root=profile.parent)
    ctx.app_root = profile
    return ctx


@pytest.mark.skipif(_idb_profile() is None, reason="no real IndexedDB store available")
def test_indexeddb_keys_have_no_stray_dot_before_bracket():
    rows = list(indexeddb.parse(_ctx_for(_idb_profile())))
    assert rows, "no IndexedDB artifacts extracted from a profile known to have an IndexedDB store"
    bad = [r.key for r in rows if _DOT_BRACKET.search(r.key)]
    assert bad == [], f"dot-before-bracket key(s): {bad[:5]}"


@pytest.mark.skipif(_idb_profile() is None, reason="no real IndexedDB store available")
def test_indexeddb_array_index_keys_use_unquoted_brackets():
    rows = list(indexeddb.parse(_ctx_for(_idb_profile())))
    # real IndexedDB records commonly hold arrays (e.g. lists of repos,
    # recently-opened items) - confirm at least one genuinely indexed key
    # made it through in the "[N]" (not quoted, not dotted) shape.
    indexed = [r.key for r in rows if re.search(r"\[\d+\]", r.key)]
    assert indexed, "no array-indexed key found - can't confirm the bracket shape on this profile"
    assert not any(_DOT_BRACKET.search(k) for k in indexed)


@pytest.mark.skipif(_idb_profile() is None, reason="no real IndexedDB store available")
def test_indexeddb_source_file_and_values_are_populated():
    rows = list(indexeddb.parse(_ctx_for(_idb_profile())))
    assert all(r.source_file for r in rows)
    assert any(r.value for r in rows)


@pytest.mark.skipif(_ss_profile() is None, reason="no real Session Storage store available")
def test_sessionstorage_does_not_crash_and_uses_the_resilient_opener():
    """Real evidence on this dev machine happens to have zero live
    session-storage key/value pairs (confirmed at the raw LevelDB record
    level - every namespace was created and torn down without ever writing
    a "map-*" record) - so this can't assert on row content, only that the
    parser runs cleanly against a real store and doesn't regress the
    resilient-open wiring added alongside indexeddb.py/localstorage.py."""
    rows = list(sessionstorage.parse(_ctx_for(_ss_profile())))
    assert isinstance(rows, list)
    bad = [r.key for r in rows if _DOT_BRACKET.search(r.key)]
    assert bad == []
