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


def test_iter_memory_artifacts_uses_process_scoped_carving_when_matched(tmp_path):
    """When Volatility finds a matching process, the whole flat image must
    NOT be carved directly - only the extracted per-process dump - and the
    resulting rows carry the matched app, not the image's pseudo-app."""
    from unittest.mock import patch

    from dd_core.casedir import CaseLayout
    from dd_lab.memory.analyze import iter_memory_artifacts

    layout = CaseLayout(tmp_path, "case1")
    mem_dir = layout.evidence / "_system" / "memory"
    mem_dir.mkdir(parents=True)
    dump_path = mem_dir / "physmem.raw"
    _fake_dump(dump_path)  # whole-image content - must not surface in this test's output

    proc_dump = tmp_path / "extracted_proc.dmp"
    proc_dump.write_bytes(b"\x00" * 4096 + b"discord process only content marker 1234" + b"\x00" * 64)

    with (
        patch("dd_lab.memory.analyze.vol_available", return_value=True),
        patch("dd_lab.memory.analyze.analyse_image", return_value=iter(())) as analyse,
        patch("dd_lab.memory.analyze.match_target_processes", return_value={4242: "discord"}),
        patch("dd_lab.memory.analyze.dump_process_memory", return_value=proc_dump) as dump_mem,
    ):
        artifacts = list(iter_memory_artifacts("case1", layout, [("_system", dump_path)], ["discord"]))

    analyse.assert_called_once()  # the structural plugins still run on the whole image
    dump_mem.assert_called_once()
    assert artifacts
    assert all(a.app == "discord" for a in artifacts)  # not "_system"
    assert any("discord process only content marker" in a.value for a in artifacts)
    assert not any("Bearer eyJ" in a.value for a in artifacts)  # whole-image content absent
    assert not proc_dump.exists()  # cleaned up right after carving
    # real bug, found while building Phase 4's process_memory agent tool: the
    # API's `q` filter and the UI's process-detail view both search/document
    # source_locator as "pid<PID>:string#n" (see api/main.py's case_artifacts
    # comment, test_api.py::test_artifacts_search_matches_source_locator) to
    # scope carved strings to one process - but _carve_and_yield previously
    # always emitted a bare "string#n" regardless of caller, so that filter
    # could never match a single real process-scoped row. Assert the real
    # per-process code path actually produces the prefixed locator now.
    assert all(a.source_locator.startswith("pid4242:string#") for a in artifacts)


def test_iter_memory_artifacts_falls_back_to_whole_image_when_nothing_matched(tmp_path):
    from unittest.mock import patch

    from dd_core.casedir import CaseLayout
    from dd_lab.memory.analyze import iter_memory_artifacts

    layout = CaseLayout(tmp_path, "case1")
    mem_dir = layout.evidence / "_system" / "memory"
    mem_dir.mkdir(parents=True)
    dump_path = mem_dir / "physmem.raw"
    _fake_dump(dump_path)

    with (
        patch("dd_lab.memory.analyze.vol_available", return_value=True),
        patch("dd_lab.memory.analyze.analyse_image", return_value=iter(())),
        patch("dd_lab.memory.analyze.match_target_processes", return_value={}),
    ):
        artifacts = list(iter_memory_artifacts("case1", layout, [("_system", dump_path)], ["discord"]))

    assert artifacts
    assert all(a.app == "_system" for a in artifacts)
    assert any("Bearer eyJ" in a.value for a in artifacts)


def test_iter_memory_artifacts_skips_matching_when_vol_unavailable(tmp_path):
    from unittest.mock import patch

    from dd_core.casedir import CaseLayout
    from dd_lab.memory.analyze import iter_memory_artifacts

    layout = CaseLayout(tmp_path, "case1")
    mem_dir = layout.evidence / "_system" / "memory"
    mem_dir.mkdir(parents=True)
    dump_path = mem_dir / "physmem.raw"
    _fake_dump(dump_path)

    with (
        patch("dd_lab.memory.analyze.vol_available", return_value=False),
        patch("dd_lab.memory.analyze.match_target_processes") as mtp,
    ):
        artifacts = list(iter_memory_artifacts("case1", layout, [("_system", dump_path)], ["discord"]))

    mtp.assert_not_called()  # no point matching processes vol can't extract anyway
    assert all(a.app == "_system" for a in artifacts)


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


def test_analyze_memory_clears_stale_system_rows_on_rerun(tmp_path):
    """Real bug, found while regenerating a real case's exports: a "_system"
    (the memory-image pseudo-app) row from an EARLIER run - e.g. a
    whole-image string carve from before process-scoped carving existed -
    was never cleared by a later "Run analysis", because ingest_case()'s
    own per-app reset loop explicitly excludes "_system" (it only covers
    real target apps). Confirmed on the real case: 400,000 stale "_system"
    memory_string rows survived a fresh run that itself only added 637,652
    new rows - never re-verified, just silently carried forward forever.

    Uses a minimal signed case (for a real manifest + warehouse) but
    monkeypatches find_dumps/iter_memory_artifacts so this doesn't need a
    real Electron profile, Volatility, or an actual memory image on disk -
    only analyze_memory()'s own reset-then-load behaviour is under test."""
    from datetime import UTC, datetime
    from unittest.mock import patch

    from dd_core.enums import SourceType, ValueType
    from dd_core.schema import NormalizedArtifact
    from dd_lab.etl.warehouse import Warehouse
    from dd_lab.memory.analyze import analyze_memory
    from tests.sample_case import build_sample_case

    profile = _find_profile()
    if profile is None:
        pytest.skip("no Electron profile")
    root = tmp_path / "cases"
    layout = build_sample_case(root, "c_stale", "app", profile, include=frozenset({"preferences"}))

    stale = NormalizedArtifact(
        artifact_id="stale-system-row", case_id="c_stale", app="_system",
        source_type=SourceType.MEMORY_STRING, source_file="_system/memory/physmem.raw",
        source_locator="string#0", key="mem.string", value="leftover from an earlier run",
        value_type=ValueType.STRING, extracted_ts=datetime.now(UTC),
    )
    wh = Warehouse(layout.warehouse)
    try:
        wh.load_artifacts([stale])
    finally:
        wh.close()

    fresh = NormalizedArtifact(
        artifact_id="fresh-system-row", case_id="c_stale", app="_system",
        source_type=SourceType.MEMORY_STRING, source_file="_system/memory/physmem.raw",
        source_locator="string#0", key="mem.string", value="produced by today's run",
        value_type=ValueType.STRING, extracted_ts=datetime.now(UTC),
    )
    with (
        patch("dd_lab.memory.analyze.find_dumps", return_value=[("_system", layout.evidence / "fake.raw")]),
        patch("dd_lab.memory.analyze.iter_memory_artifacts", return_value=iter([fresh])),
    ):
        result = analyze_memory(root, "c_stale", classifier=False)
    assert result.ok

    wh = Warehouse(layout.warehouse)
    try:
        ids = {r["artifact_id"] for r in wh.query("SELECT artifact_id FROM artifacts WHERE app = '_system'")}
    finally:
        wh.close()
    assert ids == {"fresh-system-row"}  # the stale row is gone, not just supplemented


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
