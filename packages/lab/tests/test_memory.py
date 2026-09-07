"""Memory string carving + warehouse loading (no Volatility / real dump needed)."""

from __future__ import annotations

import struct

import pytest

from dd_core.casedir import CaseLayout
from dd_core.custody import CustodyLog
from dd_core.enums import CustodyEvent
from dd_lab.memory import find_dumps
from dd_lab.memory.strings import iter_strings

_PLANTED = [
    b"Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.abcDEFghi",
    b"hey are you free to chat after school tomorrow",
    b"wss://gateway.discord.gg/?v=10&encoding=json",
    b"192.168.4.21",
    b"C:\\Users\\suspect\\AppData\\Local\\Discord",
    "displayName Kim Karter email kim@example.com".encode("utf-16-le"),
]


def _fake_dump(path):
    blob = b"\x00" * 4096
    for s in _PLANTED:
        blob += s + b"\x00" * 64
    blob += struct.pack("<1024I", *range(1024))  # some binary noise
    path.write_bytes(blob)


def test_iter_strings_finds_planted(tmp_path):
    d = tmp_path / "dump.raw"
    _fake_dump(d)
    found = list(iter_strings(d, min_len=6))
    assert any("Bearer eyJ" in s for s in found)
    assert any("chat after school" in s for s in found)
    assert any("wss://gateway.discord.gg" in s for s in found)
    assert any("kim@example.com" in s for s in found)  # utf-16


def test_analyze_memory_loads_and_classifies(tmp_path):
    # minimal signed case so a warehouse can exist
    from tests.sample_case import build_sample_case

    profile = _find_profile()
    if profile is None:
        pytest.skip("no Electron profile")
    root = tmp_path / "cases"
    blob = b"\x00" * 4096
    for s in _PLANTED:
        blob += s + b"\x00" * 64
    build_sample_case(root, "c_mem", "app", profile,
                      include=frozenset({"preferences", "local_state"}),
                      memory_dumps={"pid_1234_app.dmp": blob})
    layout = CaseLayout(root, "c_mem")

    assert len(find_dumps(layout)) == 1

    from dd_lab.ingest import ingest_case

    result = ingest_case(root, "c_mem")
    assert result.artifact_count > 0

    from dd_lab.etl.warehouse import Warehouse

    wh = Warehouse(layout.warehouse)
    try:
        mem_rows = wh.query("SELECT category, value FROM artifacts WHERE source_type = 'memory_string'")
        assert mem_rows
        cats = {r["category"] for r in mem_rows}
        assert "CREDENTIAL" in cats or "CHAT_CONTENT" in cats or "NETWORK_DETAIL" in cats
    finally:
        wh.close()

    events = [r.event for r in CustodyLog(layout.dir).entries()]
    assert CustodyEvent.ANALYSIS_STEP in events


def _find_profile():
    from pathlib import Path

    for p in (
        Path.home() / "Library/Application Support/Code",
        Path.home() / "Library/Application Support/Postman",
        Path.home() / ".config/Code",
    ):
        if (p / "Preferences").is_file() or (p / "Local Storage" / "leveldb").is_dir():
            return p
    return None
