"""dd_lab.memory.volatility.match_target_processes / dump_process_memory.

The pslist rows below are shaped exactly like the real case's
(20260915_013847_9be1ee7d) actual windows.pslist.PsList output - Discord's
multi-process fan-out, Roblox's WebView2-hosted children, and the two
orphaned DiscordSystemHelper processes whose parents had already exited -
not invented data."""

from __future__ import annotations

from unittest.mock import patch

from dd_lab.memory.volatility import _fuzzy_name_match, dump_process_memory, match_target_processes

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
