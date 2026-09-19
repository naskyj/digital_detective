"""dd_lab.memory.volatility.match_target_processes / dump_process_memory.

The pslist rows below are shaped exactly like the real case's
(20260915_013847_9be1ee7d) actual windows.pslist.PsList output - Discord's
multi-process fan-out, Roblox's WebView2-hosted children, and the two
orphaned DiscordSystemHelper processes whose parents had already exited -
not invented data."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

from dd_core.enums import ValueType
from dd_lab.memory.volatility import (
    _fuzzy_name_match,
    analyse_image,
    dump_process_memory,
    match_target_processes,
)

# a trimmed, real-shaped subset of the case's actual pslist output
_REAL_SHAPED_ROWS = [
    {"PID": 14380, "PPID": 10980, "ImageFileName": "Discord.exe"},
    {"PID": 15056, "PPID": 14380, "ImageFileName": "Discord.exe"},
    {"PID": 3112, "PPID": 14380, "ImageFileName": "Discord.exe"},
    {"PID": 3096, "PPID": 14380, "ImageFileName": "Discord.exe"},
    {"PID": 15612, "PPID": 14380, "ImageFileName": "Discord.exe"},
    {"PID": 16312, "PPID": 14380, "ImageFileName": "Discord.exe"},
    # orphaned helpers - parent PIDs (16380, 16504) NOT present in this list
    {"PID": 16228, "PPID": 16380, "ImageFileName": "DiscordSystemH"},  # truncated, 15-char EPROCESS field
    {"PID": 9960, "PPID": 16504, "ImageFileName": "DiscordSystemH"},
    # Roblox root + WebView2 fan-out, two generations deep
    {"PID": 2080, "PPID": 8460, "ImageFileName": "RobloxPlayerBe"},
    {"PID": 5276, "PPID": 2080, "ImageFileName": "RobloxCrashHan"},
    {"PID": 17152, "PPID": 2080, "ImageFileName": "msedgewebview2.exe"},
    {"PID": 17132, "PPID": 17152, "ImageFileName": "msedgewebview2.exe"},
    {"PID": 11344, "PPID": 17152, "ImageFileName": "msedgewebview2.exe"},
    # unrelated system noise that must never match
    {"PID": 8460, "PPID": 9144, "ImageFileName": "explorer.exe"},
    {"PID": 4000, "PPID": 4, "ImageFileName": "svchost.exe"},
]


def _mock_rows(rows):
    return patch("dd_lab.memory.volatility._pslist_rows", return_value=rows)


def test_fuzzy_name_match_strips_exe_and_is_bidirectional():
    assert _fuzzy_name_match("discord", "Discord.exe")
    assert _fuzzy_name_match("Discord", "discord.exe")
    assert _fuzzy_name_match("roblox", "RobloxPlayerBe")  # truncated, no .exe at all
    assert not _fuzzy_name_match("slack", "Discord.exe")


def test_fuzzy_name_match_handles_label_prefixed_app_names():
    # source-labelled multi-source cases produce app names like
    # "suspectlaptop_discord" (source_app_prefix() + safe_name()) - the
    # containment check must still find "discord" inside that.
    assert _fuzzy_name_match("suspectlaptop_discord", "Discord.exe")


def test_fuzzy_name_match_handles_real_webview2_qualified_names():
    # the real case's actual manifest target_apps name, verbatim - plain
    # whole-string containment (the pre-fix implementation) does NOT match
    # this against "RobloxPlayerBe": there is no literal "robloxplayerbe"
    # substring anywhere in the label. Token-splitting on "_" is required.
    label = "Roblox__WebView2_via_UniversalApp_WebView2"
    assert _fuzzy_name_match(label, "RobloxPlayerBe")
    # bonus effect of tokenising: "webview2" is itself a token of the label,
    # so the WebView2-hosted children are directly name-matched too, not
    # only reachable via tree-walking from the root.
    assert _fuzzy_name_match(label, "msedgewebview2.exe")


def test_name_matching_finds_the_roots():
    with _mock_rows(_REAL_SHAPED_ROWS):
        result = match_target_processes("fake.raw", ["discord", "roblox"])
    assert result[14380] == "discord"
    assert result[2080] == "roblox"


def test_tree_walk_finds_children_name_matching_alone_would_miss():
    # msedgewebview2.exe never matches "roblox" by name - only reachable by
    # walking the tree from the matched root (2080).
    with _mock_rows(_REAL_SHAPED_ROWS):
        result = match_target_processes("fake.raw", ["discord", "roblox"])
    assert result[17152] == "roblox"  # direct child of the Roblox root
    assert result[17132] == "roblox"  # grandchild, two generations deep
    assert result[11344] == "roblox"
    assert result[5276] == "roblox"  # RobloxCrashHandler, direct child


def test_name_matching_finds_orphaned_helpers_tree_walk_would_miss():
    # 16228/9960's parents (16380/16504) are NOT in the process list at all
    # (already exited) - unreachable by tree-walking from the Discord root,
    # only findable because "DiscordSystemH" itself matches "discord".
    with _mock_rows(_REAL_SHAPED_ROWS):
        result = match_target_processes("fake.raw", ["discord", "roblox"])
    assert result[16228] == "discord"
    assert result[9960] == "discord"


def test_unrelated_system_processes_never_match():
    with _mock_rows(_REAL_SHAPED_ROWS):
        result = match_target_processes("fake.raw", ["discord", "roblox"])
    assert 8460 not in result  # explorer.exe - happens to be Roblox's PPID, must not inherit
    assert 4000 not in result  # svchost.exe


def test_empty_app_names_matches_nothing():
    with _mock_rows(_REAL_SHAPED_ROWS):
        assert match_target_processes("fake.raw", []) == {}


def test_no_rows_returns_empty_not_none():
    with _mock_rows(None):
        assert match_target_processes("fake.raw", ["discord"]) == {}


def test_dump_process_memory_returns_none_on_nonzero_exit(tmp_path):
    with patch("dd_lab.memory.volatility.subprocess.run") as run:
        run.return_value.returncode = 1
        run.return_value.stderr = "no such pid"
        assert dump_process_memory("fake.raw", 99999, tmp_path) is None


def test_dump_process_memory_returns_none_if_no_file_produced(tmp_path):
    with patch("dd_lab.memory.volatility.subprocess.run") as run:
        run.return_value.returncode = 0
        run.return_value.stderr = ""
        assert dump_process_memory("fake.raw", 14380, tmp_path) is None


def test_dump_process_memory_returns_the_dump_path_on_success(tmp_path):
    expected = tmp_path / "pid.14380.dmp"
    expected.write_bytes(b"some memory content")
    with patch("dd_lab.memory.volatility.subprocess.run") as run:
        run.return_value.returncode = 0
        run.return_value.stderr = ""
        result = dump_process_memory("fake.raw", 14380, tmp_path)
    assert result == expected


# ----------------------------------------------------- analyse_image record_ts


def _fake_run_plugin(pslist_rows):
    """Only windows.pslist.PsList returns rows - every other plugin in
    _PLUGINS returns None (its real behaviour when a plugin genuinely
    produces nothing), so analyse_image's per-plugin loop skips them via its
    existing `if not rows: continue`."""

    def _run(image, plugin, timeout=900):
        return pslist_rows if plugin == "windows.pslist.PsList" else None

    return _run


def test_analyse_image_parses_createtime_and_exittime_into_record_ts(tmp_path):
    # real value format, confirmed against the actual case's saved pslist.json
    # this session: ISO8601 with a UTC offset, ExitTime null for a still-
    # running process.
    rows = [
        {"PID": 14380, "ImageFileName": "Discord.exe",
         "CreateTime": "2026-09-13T14:04:02+00:00", "ExitTime": None},
        {"PID": 8460, "ImageFileName": "explorer.exe",
         "CreateTime": "2026-09-13T14:03:10+00:00",
         "ExitTime": "2026-09-13T14:04:30+00:00"},
    ]
    with patch("dd_lab.memory.volatility.run_plugin", side_effect=_fake_run_plugin(rows)):
        artifacts = list(analyse_image("case1", "discord", tmp_path / "mem.raw", tmp_path))

    by_key = {(a.key, a.source_locator): a for a in artifacts}

    create0 = by_key[("mem.process.CreateTime", "windows.pslist.PsList#0")]
    assert create0.record_ts == datetime(2026, 9, 13, 14, 4, 2, tzinfo=UTC)
    assert create0.value_type == ValueType.TIMESTAMP

    exit1 = by_key[("mem.process.ExitTime", "windows.pslist.PsList#1")]
    assert exit1.record_ts == datetime(2026, 9, 13, 14, 4, 30, tzinfo=UTC)
    assert exit1.value_type == ValueType.TIMESTAMP

    # ExitTime=None on row 0 (still running) must not appear as an artifact
    # at all - it's filtered by the existing `val in (None, ...)` skip, same
    # as before this change, unaffected by record_ts parsing.
    assert ("mem.process.ExitTime", "windows.pslist.PsList#0") not in by_key

    # a non-timestamp column on the same rows keeps record_ts=None, unchanged
    pid0 = by_key[("mem.process.PID", "windows.pslist.PsList#0")]
    assert pid0.record_ts is None
    assert pid0.value_type == ValueType.STRING


def test_analyse_image_degrades_unparseable_timestamp_to_record_ts_none(tmp_path):
    # Volatility's own format could vary across environments/versions - must
    # never raise, just fall back to the pre-existing record_ts=None behaviour.
    rows = [{"PID": 1, "ImageFileName": "x.exe", "CreateTime": "not-a-real-timestamp"}]
    with patch("dd_lab.memory.volatility.run_plugin", side_effect=_fake_run_plugin(rows)):
        artifacts = list(analyse_image("case1", "discord", tmp_path / "mem.raw", tmp_path))

    create0 = next(a for a in artifacts if a.key == "mem.process.CreateTime")
    assert create0.record_ts is None
    assert create0.value_type == ValueType.STRING
