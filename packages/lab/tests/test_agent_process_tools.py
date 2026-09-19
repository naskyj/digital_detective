"""agent/tools.py::list_processes + process_memory — Phase 4 of the agent-
improvement plan (magical-dazzling-dragon.md). A synthetic warehouse, shaped
exactly like real Volatility output: one memory_process row per column per
(plugin, row-index) group, pslist and cmdline sharing source_type
"memory_process" but their own separate source_locator namespaces (matching
dd_lab/memory/volatility.py::analyse_image's real _PLUGINS loop), plus
memory_string rows carved by the per-process path (source_locator
"pid<PID>:string#n", see dd_lab/memory/analyze.py and the real bug fixed
alongside this phase)."""

from __future__ import annotations

from datetime import UTC, datetime

from dd_core.casedir import CaseLayout
from dd_core.enums import ArtifactCategory as C
from dd_core.enums import SourceType as S
from dd_core.schema import ClassificationResult, NormalizedArtifact
from dd_lab.agent.tools import InvestigationDeps, list_processes, process_memory
from dd_lab.etl.warehouse import Warehouse

_NOW = datetime(2026, 9, 1, tzinfo=UTC)


def _mem_process(case_id, app, locator, key, value):
    return NormalizedArtifact(
        artifact_id=f"{app}:{locator}:{key}", case_id=case_id, app=app,
        source_type=S.MEMORY_PROCESS, source_file="_system/memory/physmem.raw",
        source_locator=locator, key=key, value=value, extracted_ts=_NOW,
    )


def _mem_string(case_id, app, locator, value, *, category=C.GENERAL, confidence=0.0):
    return NormalizedArtifact(
        artifact_id=f"{app}:{locator}:mem.string", case_id=case_id, app=app,
        source_type=S.MEMORY_STRING, source_file="_system/memory/physmem.raw#pid14380",
        source_locator=locator, key="mem.string", value=value, extracted_ts=_NOW,
        classification=ClassificationResult(category=category, confidence=confidence),
    )


def _deps(tmp_path, case_id="case1") -> InvestigationDeps:
    layout = CaseLayout(tmp_path, case_id)
    wh = Warehouse(layout.warehouse)
    wh.load_artifacts([
        # pslist row for PID 14380 (Discord.exe), two columns
        _mem_process(case_id, "discord", "windows.pslist.PsList#0", "mem.process.PID", "14380"),
        _mem_process(case_id, "discord", "windows.pslist.PsList#0", "mem.process.ImageFileName", "Discord.exe"),
        _mem_process(case_id, "discord", "windows.pslist.PsList#0", "mem.process.PPID", "10980"),
        _mem_process(case_id, "discord", "windows.pslist.PsList#0", "mem.process.CreateTime",
                     "2026-09-13T14:04:02+00:00"),
        # cmdline row for the SAME pid, different plugin/locator namespace -
        # must still merge into the same pivoted process (real Volatility
        # behaviour: cmdline's own PID column repeats the pid).
        _mem_process(case_id, "discord", "windows.cmdline.CmdLine#0", "mem.cmdline.PID", "14380"),
        _mem_process(case_id, "discord", "windows.cmdline.CmdLine#0", "mem.cmdline.Args",
                     "--type=renderer"),
        # a second, unrelated process for the same app
        _mem_process(case_id, "discord", "windows.pslist.PsList#1", "mem.process.PID", "15056"),
        _mem_process(case_id, "discord", "windows.pslist.PsList#1", "mem.process.ImageFileName", "Discord.exe"),
        # a different app entirely - must not leak into discord's process list
        _mem_process(case_id, "roblox", "windows.pslist.PsList#2", "mem.process.PID", "2080"),
        _mem_process(case_id, "roblox", "windows.pslist.PsList#2", "mem.process.ImageFileName",
                     "RobloxPlayerBe"),
        # memory strings: some carved from pid 14380 specifically, one from a
        # different pid, one from the whole-image fallback (no pid prefix)
        _mem_string(case_id, "discord", "pid14380:string#0", "discord process-scoped marker one",
                    category=C.CHAT_CONTENT, confidence=0.8),
        _mem_string(case_id, "discord", "pid14380:string#1", "discord process-scoped marker two"),
        _mem_string(case_id, "discord", "pid15056:string#0", "belongs to the other process"),
        _mem_string(case_id, "discord", "string#0", "whole-image fallback carve, no pid"),
    ])
    wh.close()
    return InvestigationDeps(cases_root=tmp_path, case_id=case_id)


def test_list_processes_pivots_pslist_and_cmdline_columns_by_pid(tmp_path):
    deps = _deps(tmp_path)
    result = list_processes(deps, "discord")

    assert result["ok"] is True
    assert result["process_count"] == 2
    by_pid = {p["pid"]: p for p in result["processes"]}
    assert set(by_pid) == {"14380", "15056"}

    p = by_pid["14380"]
    assert p["ImageFileName"] == "Discord.exe"
    assert p["PPID"] == "10980"
    assert p["CreateTime"] == "2026-09-13T14:04:02+00:00"
    # cmdline's own row (different source_locator) merged into the same pid
    assert p["Args"] == "--type=renderer"


def test_list_processes_is_scoped_to_the_given_app(tmp_path):
    deps = _deps(tmp_path)
    result = list_processes(deps, "roblox")
    assert result["process_count"] == 1
    assert result["processes"][0]["pid"] == "2080"
    assert result["processes"][0]["ImageFileName"] == "RobloxPlayerBe"


def test_list_processes_empty_for_an_app_with_no_process_rows(tmp_path):
    deps = _deps(tmp_path)
    result = list_processes(deps, "slack")
    assert result == {"ok": True, "app": "slack", "process_count": 0, "processes": []}


def test_process_memory_returns_only_that_pids_carved_strings(tmp_path):
    deps = _deps(tmp_path)
    result = process_memory(deps, 14380)

    assert result["ok"] is True
    assert result["hits"] == 2
    values = {r["value"] for r in result["rows"]}
    assert values == {"discord process-scoped marker one", "discord process-scoped marker two"}
    # the other process's string, and the whole-image fallback carve (no pid
    # prefix at all), must not leak in
    assert not any("other process" in v for v in values)
    assert not any("whole-image fallback" in v for v in values)


def test_process_memory_ranks_classified_hits_first(tmp_path):
    deps = _deps(tmp_path)
    result = process_memory(deps, 14380)
    assert result["rows"][0]["value"] == "discord process-scoped marker one"  # CHAT_CONTENT, not GENERAL


def test_process_memory_empty_for_a_pid_never_carved(tmp_path):
    deps = _deps(tmp_path)
    result = process_memory(deps, 99999)
    assert result == {"ok": True, "pid": 99999, "hits": 0, "rows": []}
