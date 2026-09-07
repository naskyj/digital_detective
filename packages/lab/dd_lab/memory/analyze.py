"""Analyse the memory dumps in a case: Volatility (full images) + string carving.

Dumps live under `<case>/evidence/<app>/memory/` (per-process `.dmp`) and
`<case>/evidence/_system/memory/` (full `physmem.raw`). Output rows go into the
same warehouse and through the same classifier as disk artifacts.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from dd_core.casedir import CaseLayout
from dd_core.enums import SourceType, ValueType
from dd_core.evidence import MEMORY
from dd_core.schema import NormalizedArtifact
from dd_lab.memory.strings import iter_strings
from dd_lab.memory.volatility import analyse_image, vol_available

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


def iter_memory_artifacts(
    case_id: str, layout: CaseLayout, dumps: list[tuple[str, Path]]
) -> Iterator[NormalizedArtifact]:
    now = datetime.now(UTC)
    for app, dump in dumps:
        source_file = dump.resolve().relative_to(layout.evidence.resolve()).as_posix()
        is_full = dump.name.lower() in _FULL_IMAGE_NAMES

        if is_full and vol_available():
            log.info("running Volatility on %s", source_file)
            yield from analyse_image(case_id, app, dump, layout.evidence)

        log.info("carving strings from %s", source_file)
        for n, s in enumerate(iter_strings(dump)):
            yield NormalizedArtifact(
                artifact_id=NormalizedArtifact.make_id(
                    case_id, SourceType.MEMORY_STRING, source_file, str(n), "mem.string"
                ),
                case_id=case_id, app=app, source_type=SourceType.MEMORY_STRING,
                source_file=source_file, source_locator=f"string#{n}",
                key="mem.string", value=s, value_type=ValueType.STRING, extracted_ts=now,
            )


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

    tally = ClassifyResult()
    stream = iter_memory_artifacts(case_id, layout, dumps)
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
