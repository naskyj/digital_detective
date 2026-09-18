"""Run Volatility 3 over a full memory image and normalise its output.

Volatility 3 is invoked as the `vol` CLI with JSON rendering — more stable across
versions than the framework API. It needs symbol tables (downloaded on first use)
and a real OS memory image. If `vol` is not installed, or the image isn't a
recognised OS image, analysis is skipped with a recorded reason rather than
failing the case.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from dd_core.enums import SourceType, ValueType
from dd_core.schema import NormalizedArtifact

log = logging.getLogger("dd_lab.memory.volatility")

# plugin -> (SourceType, key columns to surface, key prefix)
_PLUGINS = {
    "windows.pslist.PsList": (SourceType.MEMORY_PROCESS, "mem.process"),
    "windows.cmdline.CmdLine": (SourceType.MEMORY_PROCESS, "mem.cmdline"),
    "windows.netscan.NetScan": (SourceType.MEMORY_NETWORK, "mem.netscan"),
    "windows.dlllist.DllList": (SourceType.MEMORY_ARTIFACT, "mem.dll"),
    "windows.malfind.Malfind": (SourceType.MEMORY_ARTIFACT, "mem.malfind"),
}


def vol_available() -> bool:
    return shutil.which("vol") is not None


def run_plugin(image: Path, plugin: str, timeout: int = 900) -> list[dict] | None:
    try:
        proc = subprocess.run(
            ["vol", "-q", "-r", "json", "-f", str(image), plugin],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        log.warning("vol %s failed: %s", plugin, e)
        return None
    if proc.returncode != 0:
        log.warning("vol %s exit %d: %s", plugin, proc.returncode, proc.stderr[:300])
        return None
    try:
        return json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        log.warning("vol %s produced non-JSON output", plugin)
        return None


def _pslist_rows(image: Path) -> list[dict] | None:
    return run_plugin(image, "windows.pslist.PsList")


_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


def _fuzzy_name_match(app_name: str, image_file_name: str) -> bool:
    """dd_collector.discover's own _running_electron/discover_apps matching
    (lowercase, strip a trailing .exe, bidirectional substring containment)
    works when app_name is a plain app label like "discord". It does NOT
    work against this project's own SANITISED, possibly source-labelled or
    WebView2-qualified target-app names actually found in real manifests -
    e.g. "Roblox__WebView2_via_UniversalApp_WebView2" contains no literal
    "robloxplayerbe" substring at all, confirmed against the real case
    (20260915_013847_9be1ee7d)'s manifest. Split on non-alphanumerics into
    tokens instead, and match if any token (4+ chars, to skip filler like
    "via") is contained in - or contains - the process name. A useful side
    effect: this also makes "webview2" itself a token of that label, so
    Roblox's WebView2-hosted children (msedgewebview2.exe) are directly
    name-matched too, not just reachable by tree-walking from the root -
    tree-walking remains the general mechanism for children that share no
    token with their parent's label (e.g. a generic Chromium helper name)."""
    b = image_file_name.strip().lower().removesuffix(".exe")
    if not b:
        return False
    tokens = {t for t in _TOKEN_SPLIT.split(app_name.strip().lower()) if len(t) >= 4}
    if not tokens:
        # a short/unsplittable app_name (rare) - fall back to whole-string containment
        a = app_name.strip().lower()
        return bool(a) and (a in b or b in a)
    return any(tok in b or b in tok for tok in tokens)


def match_target_processes(image: Path, app_names: list[str]) -> dict[int, str]:
    """PID -> matched app name, for every process belonging to one of
    `app_names` *or* spawned (directly or transitively) by one such process.

    Both name-matching and tree-walking are needed, neither alone is enough -
    confirmed against a real case: an Electron app's own child renderer/GPU/
    utility processes usually keep the same image name (name-matching alone
    would work there), but a Roblox capture's WebView2-hosted children run as
    `msedgewebview2.exe` - only reachable by walking the tree from the
    matched root - while orphaned helper processes (e.g. a
    "DiscordSystemHelper" whose parent already exited by capture time) are
    only reachable by name-matching, since tree-walking never reaches them.

    Returns {} (not None) if vol fails, there are no rows, or nothing
    matches - callers should treat that as "fall back to whole-image carving",
    not as an error.
    """
    rows = _pslist_rows(image)
    if not rows:
        return {}

    by_pid: dict[int, dict] = {}
    children: dict[int, list[int]] = {}
    for row in rows:
        pid, ppid = row.get("PID"), row.get("PPID")
        if not isinstance(pid, int):
            continue
        by_pid[pid] = row
        if isinstance(ppid, int):
            children.setdefault(ppid, []).append(pid)

    pid_to_app: dict[int, str] = {}

    # 1. name-matching, every row independently (catches orphaned helpers)
    for pid, row in by_pid.items():
        name = str(row.get("ImageFileName") or "")
        for app in app_names:
            if _fuzzy_name_match(app, name):
                pid_to_app.setdefault(pid, app)
                break

    # 2. tree-walk from every matched root (catches non-obviously-named
    # children, e.g. Roblox's WebView2 host + its own renderer fan-out)
    roots = list(pid_to_app.items())
    for root_pid, app in roots:
        stack = list(children.get(root_pid, []))
        while stack:
            pid = stack.pop()
            pid_to_app.setdefault(pid, app)
            stack.extend(children.get(pid, []))

    return pid_to_app


def dump_process_memory(image: Path, pid: int, out_dir: Path, timeout: int = 900) -> Path | None:
    """Extract one process's resident virtual address space to a flat file
    under `out_dir`, via `windows.memmap.Memmap --pid <pid> --dump` (confirmed
    against a real image: produces `<out_dir>/pid.<pid>.dmp`, a file
    `iter_strings()` can run against unmodified). Never raises - same
    swallow-and-log convention as `run_plugin`; returns None on any failure
    so callers can fall back cleanly."""
    try:
        proc = subprocess.run(
            ["vol", "-q", "-o", str(out_dir), "-f", str(image),
             "windows.memmap.Memmap", "--pid", str(pid), "--dump"],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        log.warning("vol memmap --pid %d failed: %s", pid, e)
        return None
    if proc.returncode != 0:
        log.warning("vol memmap --pid %d exit %d: %s", pid, proc.returncode, proc.stderr[:300])
        return None
    dump = out_dir / f"pid.{pid}.dmp"
    if not dump.is_file() or dump.stat().st_size == 0:
        log.warning("vol memmap --pid %d produced no output", pid)
        return None
    return dump


def analyse_image(
    case_id: str, app: str, image: Path, evidence_root: Path
) -> Iterator[NormalizedArtifact]:
    source_file = Path(image).resolve().relative_to(Path(evidence_root).resolve()).as_posix()
    now = datetime.now(UTC)
    for plugin, (stype, prefix) in _PLUGINS.items():
        rows = run_plugin(image, plugin)
        if not rows:
            continue
        for i, row in enumerate(rows):
            for col, val in row.items():
                if val in (None, "", "-", "N/A") or col.startswith("__"):
                    continue
                yield NormalizedArtifact(
                    artifact_id=NormalizedArtifact.make_id(
                        case_id, stype, source_file, f"{plugin}:{i}", col
                    ),
                    case_id=case_id, app=app, source_type=stype,
                    source_file=source_file, source_locator=f"{plugin}#{i}",
                    key=f"{prefix}.{col}", value=str(val)[:8000],
                    value_type=ValueType.STRING, extracted_ts=now,
                )
