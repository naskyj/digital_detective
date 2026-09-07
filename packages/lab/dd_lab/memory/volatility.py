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
