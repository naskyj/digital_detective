"""Memory acquisition.

Full-system capture uses WinPmem (open-source, https://github.com/Velocidex/WinPmem)
which must be present next to the collector as `winpmem.exe`. Per-process capture
uses the OS's own facilities. Everything here degrades gracefully: if a tool or
privilege is missing, the acquisition records that and continues — a missing
memory dump must never abort collection of the disk artifacts.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MemoryResult:
    ok: bool
    kind: str  # "full" | "process"
    path: Path | None
    detail: str
    pid: int | None = None


def _winpmem_path() -> Path | None:
    here = Path(__file__).resolve().parent
    for cand in (here / "winpmem.exe", here.parent / "winpmem.exe", Path("winpmem.exe")):
        if cand.is_file():
            return cand
    found = shutil.which("winpmem")
    return Path(found) if found else None


def capture_full_memory(out_dir: Path) -> MemoryResult:
    if platform.system() != "Windows":
        return MemoryResult(False, "full", None, f"full memory capture unsupported on {platform.system()}")
    winpmem = _winpmem_path()
    if winpmem is None:
        return MemoryResult(False, "full", None, "winpmem.exe not found next to the collector")
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "physmem.raw"
    try:
        proc = subprocess.run(
            [str(winpmem), "--output", str(target), "--format", "raw"],
            capture_output=True, text=True, timeout=3600, check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        return MemoryResult(False, "full", None, f"winpmem failed: {e}")
    if proc.returncode != 0 or not target.is_file():
        return MemoryResult(False, "full", None, f"winpmem exit {proc.returncode}: {proc.stderr[:400]}")
    return MemoryResult(True, "full", target, f"{target.stat().st_size:,} bytes via {winpmem.name}")


def capture_process_memory(pid: int, name: str, out_dir: Path) -> MemoryResult:
    system = platform.system()
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"pid_{pid}_{_safe(name)}.dmp"

    if system == "Windows":
        # Prefer a full user-mode minidump via a bundled procdump if present.
        procdump = shutil.which("procdump")
        if procdump:
            proc = subprocess.run(
                [procdump, "-accepteula", "-ma", str(pid), str(target)],
                capture_output=True, text=True, timeout=600, check=False,
            )
            if target.is_file():
                return MemoryResult(True, "process", target, "procdump -ma", pid)
            return MemoryResult(False, "process", None, f"procdump failed: {proc.stderr[:300]}", pid)
        return MemoryResult(False, "process", None, "procdump not found (bundle it for process dumps)", pid)

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
