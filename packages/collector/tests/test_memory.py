"""dd_collector.memory — tool discovery + DumpIt-first / WinPmem-fallback logic.

subprocess.run is mocked throughout: a real winpmem.exe/DumpIt.exe can't
execute on a non-Windows dev machine, and we want deterministic exit codes
without needing the real binaries at all.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dd_collector import memory


@pytest.fixture
def tools_dir(tmp_path, monkeypatch):
    d = tmp_path / "tools"
    d.mkdir()
    monkeypatch.setenv(memory._TOOL_DIRS_ENV, str(d))
    return d


@pytest.fixture
def windows(monkeypatch):
    monkeypatch.setattr(memory.platform, "system", lambda: "Windows")


def _touch(d, name: str) -> None:
    (d / name).write_bytes(b"fake-exe")


def test_full_memory_capture_unsupported_off_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(memory.platform, "system", lambda: "Darwin")
    r = memory.capture_full_memory(tmp_path / "out")
    assert not r.ok
    assert "unsupported" in r.detail


def test_no_tool_found_reports_both_names(windows, tools_dir, tmp_path):
    r = memory.capture_full_memory(tmp_path / "out")
    assert not r.ok
    assert "DumpIt" in r.detail and "winpmem" in r.detail


def test_dumpit_preferred_over_winpmem(windows, tools_dir, tmp_path, monkeypatch):
    _touch(tools_dir, "DumpIt.exe")
    _touch(tools_dir, "winpmem.exe")
    calls: list[list[str]] = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        # simulate DumpIt writing the requested output file
        out = cmd[cmd.index("/O") + 1]
        from pathlib import Path

        Path(out).write_bytes(b"RAM" * 100)
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(memory.subprocess, "run", fake_run)
    r = memory.capture_full_memory(tmp_path / "out")

    assert r.ok
    assert r.kind == "full"
    assert r.path.name == "physmem.raw"
    assert r.path.read_bytes() == b"RAM" * 100
    assert "DumpIt.exe" in r.detail
    assert len(calls) == 1, "winpmem must not be invoked once DumpIt succeeds"
    assert calls[0][0].endswith("DumpIt.exe")
    assert "/T" in calls[0] and "RAW" in calls[0]


def test_falls_back_to_winpmem_when_dumpit_fails(windows, tools_dir, tmp_path, monkeypatch):
    _touch(tools_dir, "DumpIt.exe")
    _touch(tools_dir, "winpmem.exe")

    def fake_run(cmd, **kw):
        exe = cmd[0]
        if "DumpIt" in exe:
            return MagicMock(returncode=1, stdout="", stderr="driver install failed")
        # winpmem: --output <path> --format raw
        from pathlib import Path

        out = cmd[cmd.index("--output") + 1]
        Path(out).write_bytes(b"RAM")
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(memory.subprocess, "run", fake_run)
    r = memory.capture_full_memory(tmp_path / "out")

    assert r.ok
    assert "winpmem.exe" in r.detail


def test_both_tools_fail_reports_combined_detail(windows, tools_dir, tmp_path, monkeypatch):
    _touch(tools_dir, "DumpIt.exe")
    _touch(tools_dir, "winpmem.exe")
    monkeypatch.setattr(
        memory.subprocess, "run",
        lambda cmd, **kw: MagicMock(returncode=1, stdout="", stderr="nope"),
    )
    r = memory.capture_full_memory(tmp_path / "out")
    assert not r.ok
    assert "DumpIt exit 1" in r.detail
    assert "winpmem exit 1" in r.detail


def test_missing_tool_never_aborts_process_capture(monkeypatch, tmp_path):
    monkeypatch.setattr(memory, "_find_tool", lambda *a: None)
    monkeypatch.setattr(memory.shutil, "which", lambda name: None)
    r = memory.capture_process_memory(1234, "SomeApp", tmp_path / "out")
    assert not r.ok
    assert r.pid == 1234


def test_find_tool_prefers_env_dir(tmp_path, monkeypatch):
    d = tmp_path / "custom"
    d.mkdir()
    _touch(d, "winpmem.exe")
    monkeypatch.setenv(memory._TOOL_DIRS_ENV, str(d))
    found = memory._find_tool("winpmem.exe")
    assert found == d / "winpmem.exe"
