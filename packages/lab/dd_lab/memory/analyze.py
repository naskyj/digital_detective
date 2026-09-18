"""Analyse the memory dumps in a case: Volatility (full images) + string carving.

Dumps live under `<case>/evidence/<app>/memory/` (per-process `.dmp`) and
`<case>/evidence/_system/memory/` (full `physmem.raw`). Output rows go into the
same warehouse and through the same classifier as disk artifacts.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from dd_core.casedir import CaseLayout
from dd_core.enums import SourceType, ValueType
from dd_core.evidence import MEMORY
from dd_core.manifest import Manifest
from dd_core.schema import NormalizedArtifact
from dd_lab.memory.strings import iter_strings
from dd_lab.memory.volatility import analyse_image, dump_process_memory, match_target_processes, vol_available

log = logging.getLogger("dd_lab.memory")

# a full OS image: try Volatility. a process .dmp: string-carve only.
_FULL_IMAGE_NAMES = {"physmem.raw", "physmem.dmp", "memory.raw", "memdump.raw"}


@dataclass
class MemoryResult:
    ok: bool
    skipped_reason: str | None = None
    dumps: list[str] = field(default_factory=list)
    artifact_count: int = 0
    volatility_used: bool = False
    by_category: dict[str, int] = field(default_factory=dict)


def find_dumps(layout: CaseLayout) -> list[tuple[str, Path]]:
    """Return (app, dump_path) for every memory dump in the case."""
    out: list[tuple[str, Path]] = []
    if not layout.evidence.is_dir():
        return out
    for app_dir in sorted(layout.evidence.iterdir()):
        mem_dir = app_dir / MEMORY
        if mem_dir.is_dir():
            for f in sorted(mem_dir.rglob("*")):
                if f.is_file() and f.suffix.lower() in (".raw", ".dmp", ".mem", ".bin"):
                    out.append((app_dir.name, f))
    return out


def _carve_and_yield(
    case_id: str, app: str, dump_path: Path, source_file: str, now: datetime,
) -> Iterator[NormalizedArtifact]:
    """Run iter_strings() over `dump_path` and yield memory_string artifacts
    tagged with `app`/`source_file` - the one string-carving step shared by
    both the whole-image path and the per-process path below."""
    for n, s in enumerate(iter_strings(dump_path)):
        yield NormalizedArtifact(
            artifact_id=NormalizedArtifact.make_id(
                case_id, SourceType.MEMORY_STRING, source_file, str(n), "mem.string"
            ),
            case_id=case_id, app=app, source_type=SourceType.MEMORY_STRING,
            source_file=source_file, source_locator=f"string#{n}",
            key="mem.string", value=s, value_type=ValueType.STRING, extracted_ts=now,
        )


def iter_memory_artifacts(
    case_id: str, layout: CaseLayout, dumps: list[tuple[str, Path]],
    app_names: list[str] | None = None,
) -> Iterator[NormalizedArtifact]:
    now = datetime.now(UTC)
    for app, dump in dumps:
        source_file = dump.resolve().relative_to(layout.evidence.resolve()).as_posix()
        is_full = dump.name.lower() in _FULL_IMAGE_NAMES

        if not (is_full and vol_available()):
            log.info("carving strings from %s", source_file)
            yield from _carve_and_yield(case_id, app, dump, source_file, now)
            continue

        log.info("running Volatility on %s", source_file)
        yield from analyse_image(case_id, app, dump, layout.evidence)

        pid_to_app = match_target_processes(dump, app_names or [])
        if not pid_to_app:
            # no target-app process found on this image (or vol failed) -
            # fall back to carving the whole flat image, exactly as before.
            log.info("carving strings from %s (whole image - no matched processes)", source_file)
            yield from _carve_and_yield(case_id, app, dump, source_file, now)
            continue

        log.info("carving %d matched process(es) instead of the whole image: %s",
                  len(pid_to_app), sorted(set(pid_to_app.values())))
        with tempfile.TemporaryDirectory(prefix="dd_memmap_") as tmp:
            tmp_dir = Path(tmp)
            for pid, matched_app in pid_to_app.items():
                proc_dump = dump_process_memory(dump, pid, tmp_dir)
                if proc_dump is None:
                    log.warning("could not extract pid %d memory, skipping", pid)
                    continue
                proc_source_file = f"{source_file}#pid{pid}"
                try:
                    yield from _carve_and_yield(case_id, matched_app, proc_dump, proc_source_file, now)
                finally:
                    proc_dump.unlink(missing_ok=True)  # don't hold every matched process's dump at once


def analyze_memory(
    cases_root: str | Path, case_id: str, *, classifier=None, load: bool = True
) -> MemoryResult:
    from dd_lab.classify import classify_stream
    from dd_lab.classify.pipeline import ClassifyResult
    from dd_lab.etl.warehouse import Warehouse

    layout = CaseLayout(cases_root, case_id)
    dumps = find_dumps(layout)
    if not dumps:
        return MemoryResult(ok=False, skipped_reason="no memory dumps in evidence")

    result = MemoryResult(ok=True, dumps=[str(d) for _, d in dumps],
                          volatility_used=any(d.name.lower() in _FULL_IMAGE_NAMES for _, d in dumps)
                          and vol_available())

    # the case's real target apps (not "_system", the memory-image pseudo-app
    # itself) - used to scope full-image string carving to just their
    # processes (+ spawned descendants) instead of the whole flat image.
    app_names = [t.name for t in Manifest.read(layout.dir).target_apps if t.name != "_system"]

    tally = ClassifyResult()
    stream = iter_memory_artifacts(case_id, layout, dumps, app_names)
    if classifier is not False:
        stream = classify_stream(stream, classifier or None, tally)

    if load:
        wh = Warehouse(layout.warehouse)
        try:
            result.artifact_count = wh.load_artifacts(stream)
            wh.build_views()
        finally:
            wh.close()
    else:
        result.artifact_count = sum(1 for _ in stream)
    result.by_category = tally.by_category
    return result
