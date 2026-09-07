"""Collector: discovery + acquisition against a real Electron profile.

Runs on any OS with a Chromium-profile app present. Memory capture is not
exercised (needs Windows + WinPmem); its absence must not fail acquisition.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dd_collector.acquire import acquire
from dd_collector.discover import ElectronApp, discover_apps, find_app
from dd_core.custody import CustodyLog
from dd_core.manifest import verify_case

_PROFILES = [
    Path.home() / "Library/Application Support/Code",
    Path.home() / "Library/Application Support/Postman",
    Path.home() / ".config/Code",
]


def _a_profile() -> Path | None:
    for p in _PROFILES:
        if (p / "Local Storage" / "leveldb").is_dir() or (p / "Preferences").is_file():
            return p
    return None


pytestmark = pytest.mark.skipif(_a_profile() is None, reason="no Electron profile present")


def test_discovery_finds_something():
    apps = discover_apps()
    assert apps
    assert all(isinstance(a, ElectronApp) for a in apps)
    assert all(a.profile_dir.is_dir() for a in apps)


def test_find_app_by_path():
    profile = _a_profile()
    app = find_app(str(profile))
    assert app is not None
    assert app.profile_dir == profile


def test_acquisition_produces_verifiable_case(tmp_path):
    profile = _a_profile()
    app = ElectronApp(name="target", profile_dir=profile)
    logs: list[str] = []
    result = acquire([app], tmp_path, operator="Test Operator", progress=logs.append)

    assert result.file_count > 0
    assert (result.case_dir / "manifest.json").is_file()
    assert (result.case_dir / "manifest.json.sig").is_file()

    # the acquired case verifies with the lab's own verifier
    v = verify_case(result.case_dir)
    assert v.ok, v.issues
    assert v.signature_valid

    # custody chain is intact and records the acquisition lifecycle
    coc = CustodyLog(result.case_dir)
    ok, bad = coc.verify_chain()
    assert ok, bad
    events = {r.event for r in coc.entries()}
    assert {"acquisition_start", "artifact_acquired", "acquisition_complete", "manifest_signed"} <= events

    assert any("copying profile" in m for m in logs)


def test_source_is_not_modified(tmp_path):
    profile = _a_profile()
    prefs = profile / "Preferences"
    before = prefs.stat().st_mtime if prefs.is_file() else None
    acquire([ElectronApp(name="t", profile_dir=profile)], tmp_path, operator="op")
    after = prefs.stat().st_mtime if prefs.is_file() else None
    assert before == after


def test_locked_file_is_skipped_not_fatal(tmp_path):
    """A single unreadable file (as a running app's LevelDB WAL would be) must
    not abort the acquisition — it's recorded and the rest proceeds."""
    import os

    from dd_core.evidence import CopySpec, execute_copy

    src = tmp_path / "leveldb"
    src.mkdir()
    (src / "000005.ldb").write_bytes(b"committed")
    locked = src / "000006.log"
    locked.write_bytes(b"wal")
    os.chmod(locked, 0)
    try:
        report = execute_copy(
            [CopySpec(source=src, dest_rel="local_storage/leveldb", is_dir=True)],
            tmp_path / "out",
        )
        assert any("000005" in p.name for p in report.written)
        assert report.skipped and "000006.log" in report.skipped[0][0]
    finally:
        os.chmod(locked, 0o644)
