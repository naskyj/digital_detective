"""End-to-end ingest against a real Electron app profile on the dev machine.

The default test copies only the small, fast sources (Local/Session Storage,
Preferences, Cookies). The full-cache path is a separate `slow` test.
Skips cleanly if no supported profile is present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dd_core.custody import CustodyLog
from dd_lab.etl.warehouse import Warehouse
from dd_lab.ingest import ingest_case

from tests.sample_case import build_sample_case

_CANDIDATES = {
    "vscode": Path.home() / "Library/Application Support/Code",
    "postman": Path.home() / "Library/Application Support/Postman",
}
_FAST_SOURCES = frozenset({"local_storage", "session_storage", "preferences", "cookies", "local_state"})


def _available() -> tuple[str, Path] | None:
    for app, path in _CANDIDATES.items():
        if (path / "Local Storage" / "leveldb").is_dir() or (path / "Preferences").is_file():
            return app, path
    return None


pytestmark = pytest.mark.skipif(_available() is None, reason="no Electron profile available")


@pytest.fixture(scope="module")
def ingested(tmp_path_factory):
    app, profile = _available()
    root = tmp_path_factory.mktemp("cases")
    case_id = "case_sample_001"
    build_sample_case(root, case_id, app, profile, include=_FAST_SOURCES)
    result = ingest_case(root, case_id)
    return root, case_id, app, result


def test_integrity_and_custody(ingested):
    root, case_id, app, result = ingested
    assert result.artifact_count > 0
    assert app in result.per_app

    log = CustodyLog(root / case_id)
    events = [r.event for r in log.entries()]
    assert "integrity_verified" in events
    assert "analysis_complete" in events
    ok, bad = log.verify_chain()
    assert ok, f"custody chain broken at {bad}"


def test_refuses_tampered_evidence(ingested, tmp_path):
    """A modified evidence file must abort ingest and log INTEGRITY_FAILED."""
    from dd_lab.ingest import IntegrityError

    root, case_id, app, _ = ingested
    # tamper with a copied file
    victim = next((root / case_id / "evidence").rglob("*"))
    while victim.is_dir():
        victim = next(victim.iterdir())
    original = victim.read_bytes()
    victim.write_bytes(original + b"\x00TAMPER")
    try:
        with pytest.raises(IntegrityError):
            ingest_case(root, case_id)
        events = [r.event for r in CustodyLog(root / case_id).entries()]
        assert "integrity_failed" in events
    finally:
        victim.write_bytes(original)


def test_warehouse_queryable(ingested):
    root, case_id, _app, result = ingested
    wh = Warehouse(root / case_id / "warehouse" / "case.duckdb")
    try:
        assert wh.count() == result.artifact_count
        assert wh.query("SELECT * FROM v_source_breakdown")
        assert wh.query("SELECT * FROM v_app_summary")
        orphans = wh.query(
            "SELECT count(*) AS n FROM artifacts WHERE source_file IS NULL OR source_file = ''"
        )
        assert orphans[0]["n"] == 0
    finally:
        wh.close()


@pytest.mark.slow
def test_full_cache_ingest(tmp_path):
    app, profile = _available()
    if not (profile / "Cache").exists():
        pytest.skip("no cache in profile")
    build_sample_case(tmp_path, "case_full", app, profile)  # all sources
    result = ingest_case(tmp_path, "case_full")
    assert result.artifact_count > 0
    wh = Warehouse(tmp_path / "case_full" / "warehouse" / "case.duckdb")
    try:
        cache_rows = wh.query("SELECT count(*) AS n FROM artifacts WHERE source_type = 'cache'")
        assert cache_rows[0]["n"] > 0
        # Real evidence found this exact shape broken: a "." immediately
        # before a "[" - whether from a genuine array index (the original
        # bug) or a literal object key that merely looks like one (VS
        # Code's own "[python]"/"[java]" settings keys - the more subtle
        # follow-up bug, found on this same real cache).
        bad = wh.query("SELECT key FROM artifacts WHERE source_type = 'cache' AND key LIKE '%.[%'")
        assert bad == [], f"dot-before-bracket key(s) still present: {bad[:5]}"
    finally:
        wh.close()
