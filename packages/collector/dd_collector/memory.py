"""Memory acquisition.

Full physical-RAM capture uses WinPmem or DumpIt (Comae) — whichever is found
next to the collector. A full image captures *all* of physical memory: the
target app does not need to be running for the capture to work, though its
in-memory data will only be present if it was running at (or recently before)
capture time. Per-process capture (procdump) does need the app running.

Everything here degrades gracefully: if a tool or privilege is missing, the
acquisition records that and continues — a missing memory dump must never abort
collection of the disk artifacts.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

_TOOL_DIRS_ENV = "DD_TOOLS_DIR"


@dataclass
class MemoryResult:
    ok: bool
    kind: str  # "full" | "process"
    path: Path | None
    detail: str
    pid: int | None = None


def _find_tool(*names: str) -> Path | None:
    """Look for a bundled acquisition tool next to the collector, in DD_TOOLS_DIR,
    in a sibling `tools/` folder, or on PATH."""
    dirs = [Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent, Path.cwd()]
    if os.environ.get(_TOOL_DIRS_ENV):
        dirs.insert(0, Path(os.environ[_TOOL_DIRS_ENV]))
    dirs += [d / "tools" for d in list(dirs)]
    for name in names:
        for d in dirs:
            cand = d / name
            if cand.is_file():
                return cand
        found = shutil.which(name.removesuffix(".exe"))
        if found:
            return Path(found)
    return None


def capture_full_memory(out_dir: Path) -> MemoryResult:
    if platform.system() != "Windows":
        return MemoryResult(
            False, "full", None, f"full memory capture unsupported on {platform.system()}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "physmem.raw"

    winpmem = _find_tool("winpmem.exe", "winpmem_mini_x64_rc2.exe", "winpmem")
    if winpmem:
        try:
            proc = subprocess.run(
                [str(winpmem), "--output", str(target), "--format", "raw"],
                capture_output=True, text=True, timeout=7200, check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            return MemoryResult(False, "full", None, f"winpmem failed: {e}")
        if proc.returncode == 0 and target.is_file():
            return MemoryResult(True, "full", target,
                                f"{target.stat().st_size:,} bytes via {winpmem.name}")
        # fall through to DumpIt on failure

    dumpit = _find_tool("DumpIt.exe", "dumpit.exe", "DumpIt")
    if dumpit:
        try:
            # Comae DumpIt: /OUTPUT <file> /QUIET ; MoonSols DumpIt: interactive Y/N
            proc = subprocess.run(
                [str(dumpit), "/OUTPUT", str(target), "/QUIET"],
                capture_output=True, text=True, timeout=7200, check=False, input="y\n",
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            return MemoryResult(False, "full", None, f"DumpIt failed: {e}")
        produced = target if target.is_file() else next(
            iter(sorted(out_dir.glob("*.raw")) + sorted(out_dir.glob("*.dmp"))), None
        )
        if produced:
            if produced != target:
                produced.rename(target)
            return MemoryResult(True, "full", target,
                                f"{target.stat().st_size:,} bytes via {dumpit.name}")
        return MemoryResult(False, "full", None,
                            f"DumpIt exit {proc.returncode}: {(proc.stderr or proc.stdout)[:400]}")

    return MemoryResult(
        False, "full", None,
        "no memory-capture tool found — bundle winpmem.exe or DumpIt.exe with the collector "
        "(or set DD_TOOLS_DIR)",
    )


def capture_process_memory(pid: int, name: str, out_dir: Path) -> MemoryResult:
    system = platform.system()
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"pid_{pid}_{_safe(name)}.dmp"

    if system == "Windows":
        procdump = _find_tool("procdump64.exe", "procdump.exe", "procdump")
        if procdump:
            proc = subprocess.run(
                [str(procdump), "-accepteula", "-ma", str(pid), str(target)],
                capture_output=True, text=True, timeout=600, check=False,
            )
            if target.is_file():
                return MemoryResult(True, "process", target, "procdump -ma", pid)
            return MemoryResult(False, "process", None,
                                f"procdump failed: {(proc.stderr or proc.stdout)[:300]}", pid)
        return MemoryResult(False, "process", None,
                            "procdump not found (bundle it for per-process dumps)", pid)

    # macOS / Linux: best-effort for development only.
    tool = shutil.which("gcore")
    if tool:
        proc = subprocess.run([tool, "-o", str(target), str(pid)], capture_output=True, text=True,
                              timeout=600, check=False)
        produced = next(iter(sorted(out_dir.glob(f"{target.name}*"))), None)
        if produced:
            produced.rename(target)
            return MemoryResult(True, "process", target, f"gcore ({proc.returncode})", pid)
    return MemoryResult(False, "process", None, f"no process-dump tool available on {system}", pid)


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:40]
