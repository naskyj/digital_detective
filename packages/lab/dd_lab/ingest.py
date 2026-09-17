"""Import a collected case into the lab: verify integrity, parse, load, view.

This is the deterministic backbone the agent drives. It refuses to touch a case
whose evidence does not match its signed manifest.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from dd_core.casedir import CaseLayout
from dd_core.custody import CustodyLog
from dd_core.enums import CustodyEvent
from dd_core.manifest import Manifest
from dd_lab.classify import Classifier, classify_stream
from dd_lab.classify.pipeline import ClassifyResult
from dd_lab.etl.warehouse import Warehouse
from dd_lab.parsers import discover_apps, parse_app

log = logging.getLogger("dd_lab.ingest")


class IntegrityError(RuntimeError):
    """Raised when evidence does not match its signed manifest."""


@dataclass
class IngestResult:
    case_id: str
    run_id: str
    apps: list[str]
    artifact_count: int
    per_app: dict[str, int] = field(default_factory=dict)
    by_category: dict[str, int] = field(default_factory=dict)
    started_ts: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_ts: datetime | None = None


def ingest_case(
    case_root: str | Path,
    case_id: str,
    *,
    actor: str = "dd-lab",
    require_integrity: bool = True,
    classify: bool = True,
    classifier: Classifier | None = None,
    analyze_memory: bool = True,
) -> IngestResult:
    layout = CaseLayout(case_root, case_id)
    custody = CustodyLog(layout.dir)

    # 1. integrity ----------------------------------------------------------
    from dd_lab.verification import verify_cached

    verification, _ = verify_cached(case_root, case_id, force=True)
    if not verification.ok:
        custody.append(
            CustodyEvent.INTEGRITY_FAILED, actor,
            f"verification failed: sig_valid={verification.signature_valid}, "
            f"{len(verification.issues)} issue(s)",
            data={"issues": [i.model_dump() for i in verification.issues[:50]]},
        )
        if require_integrity:
            raise IntegrityError(
                f"evidence integrity check failed for {case_id}: "
                f"signature_valid={verification.signature_valid}, issues={verification.issues[:5]}"
            )
        log.warning("continuing despite integrity failure (require_integrity=False)")
    else:
        custody.append(
            CustodyEvent.INTEGRITY_VERIFIED, actor,
            f"{verification.checked} files verified against signed manifest",
        )

    # 2. import -----------------------------------------------------------
    layout.create_lab_skeleton()
    manifest = Manifest.read(layout.dir)
    apps = [a.name for a in manifest.target_apps] or discover_apps(layout.evidence)
    run_id = uuid.uuid4().hex
    result = IngestResult(case_id=case_id, run_id=run_id, apps=apps, artifact_count=0)
    custody.append(
        CustodyEvent.IMPORT, actor, f"import started (run {run_id}) for apps: {', '.join(apps)}",
    )
    custody.append(CustodyEvent.ANALYSIS_START, actor, "parse + load into warehouse")

    # 3. parse (+ classify in-stream) + load -----------------------------
    wh = Warehouse(layout.warehouse)
    tally = ClassifyResult()
    if classify:
        custody.append(CustodyEvent.ANALYSIS_STEP, actor, "classifying artifacts in-stream (ensemble)")
    clf = classifier or (Classifier() if classify else None)
    try:
        for app in apps:
            # load_artifacts() is ON CONFLICT (artifact_id) DO NOTHING - an
            # artifact's id is stable (hashed from case_id/source_type/
            # source_file/locator/key, not its computed fields), so without
            # clearing first, "Re-run analysis" on an already-processed case
            # would silently keep every artifact's ORIGINAL fields forever,
            # even after a parser or classifier fix that changes what gets
            # computed for that same id (the `deleted` flag being exactly
            # such a case). reset_app() makes a re-run genuinely re-derive
            # this app's artifacts from the evidence, not just re-assert
            # unchanged ones.
            wh.reset_app(app)
            counter = _Counter()
            stream = _counting(parse_app(case_id, layout.evidence, app), counter)
            if classify:
                stream = classify_stream(stream, clf, tally)
            wh.load_artifacts(stream)
            result.per_app[app] = counter.n
            result.artifact_count += counter.n
            custody.append(
                CustodyEvent.ANALYSIS_STEP, actor, f"parsed {app}: {counter.n} artifacts",
            )
        wh.build_views()
        if classify:
            result.by_category = tally.by_category
            custody.append(
                CustodyEvent.ANALYSIS_STEP, actor,
                f"classified {tally.classified} artifacts ({tally.classifier_version})",
                data={"by_category": tally.by_category},
            )
    finally:
        wh.close()

    # memory analysis uses its own warehouse connection, so run it after close
    if analyze_memory:
        from dd_lab.memory import analyze_memory as _analyze_memory
        from dd_lab.memory import find_dumps

        if find_dumps(layout):
            custody.append(CustodyEvent.ANALYSIS_STEP, actor, "analysing memory dumps")
            mem = _analyze_memory(case_root, case_id, classifier=clf)
            result.artifact_count += mem.artifact_count
            for cat, n in mem.by_category.items():
                result.by_category[cat] = result.by_category.get(cat, 0) + n
            custody.append(
                CustodyEvent.ANALYSIS_STEP, actor,
                f"memory: +{mem.artifact_count} artifacts (volatility={mem.volatility_used})",
            )

    wh = Warehouse(layout.warehouse)
    try:
        result.finished_ts = datetime.now(UTC)
        wh.build_views()
        wh.record_run(
            run_id, result.started_ts, result.finished_ts, apps, result.artifact_count,
            notes="ingest_case",
        )
    finally:
        wh.close()

    custody.append(
        CustodyEvent.ANALYSIS_COMPLETE, actor,
        f"ingest complete: {result.artifact_count} artifacts across {len(apps)} app(s)",
        data={"per_app": result.per_app, "run_id": run_id},
    )
    return result


class _Counter:
    __slots__ = ("n",)

    def __init__(self) -> None:
        self.n = 0


def _counting(it, counter: _Counter):
    for x in it:
        counter.n += 1
        yield x
