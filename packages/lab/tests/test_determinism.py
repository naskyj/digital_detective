"""R1: the same evidence must produce the same conclusions, every run.

The requirement is stated in the paper (§1.2) and relied on by every claim
about reproducibility, but nothing tested it. A reviewer asked for exactly
this — run it five times and hash the outputs.

What this pins is that the two deterministic investigation outputs,
correlation and the computed narrative, are byte-identical across repeated
runs over the same warehouse. That is not automatic: both aggregate over
dict and set iteration, both sort by scores that can tie, and a tie broken by
hash order would be stable within a process and vary across processes. A
report that changed between runs on identical evidence would undermine every
other guarantee in the system.

Verified on the real 675,234-artifact case before being written down here:
five runs, one distinct hash for each output.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta

import pytest

from dd_core.enums import ArtifactCategory as C
from dd_core.enums import SourceType as S
from dd_core.schema import ClassificationResult, NormalizedArtifact
from dd_lab.etl.warehouse import Warehouse
from dd_lab.investigate.correlate import build_correlation_report
from dd_lab.investigate.timeline import build_case_narrative

_NOW = datetime(2026, 9, 13, 18, 0, tzinfo=UTC)
_RUNS = 5


def _canon(obj) -> bytes:
    """Content hash, not memory-layout hash: sorting keys means a difference
    in dict ordering alone cannot show up as a difference here, so anything
    this does catch is a real change in what the system concluded."""
    return json.dumps(obj, sort_keys=True, default=str).encode()


def _art(aid: str, app: str, key: str, value: str, minutes: int,
         category: C = C.USER):
    return NormalizedArtifact(
        artifact_id=aid, case_id="c1", app=app, source_type=S.CACHE,
        source_file=f"{app}/Cache/data_1", source_locator=f"loc-{aid}",
        key=key, value=value,
        record_ts=_NOW + timedelta(minutes=minutes), extracted_ts=_NOW,
        classification=ClassificationResult(
            category=category, confidence=0.9, classifier_version="test",
            relevant=True, relevance_score=1.0, relevance_model_score=0.99),
    )


@pytest.fixture
def wh(tmp_path):
    """A case with cross-app shared identifiers and tied scores — the shape
    that makes ordering non-trivial. Several artifacts deliberately share a
    relevance score and a timestamp so any tie-break that fell back on hash
    or insertion order would show up as a differing digest."""
    w = Warehouse(tmp_path / "case.duckdb")
    shared = "tyler_mia_2026"
    w.load_artifacts([
        _art("a1", "discord", "idb.user.name", f"{shared} joined", 0),
        _art("a2", "Slack", "cache.body", f"hello {shared}", 0),      # tie on time
        _art("a3", "roblox", "local_storage.u", f"{shared}", 0),      # tie on time
        _art("a4", "discord", "cache.body.content", "see you at five", 5),
        _art("a5", "Slack", "cache.body.content", "see you at five", 5),
        _art("a6", "roblox", "idb.session", "9182736455", 40),
        _art("a7", "discord", "idb.session", "9182736455", 41),
        # a second episode, after the default session gap
        _art("a8", "discord", "cache.url", f"https://x.test/{shared}", 400),
        _art("a9", "Slack", "cache.url", f"https://x.test/{shared}", 402),
    ])
    w.build_views()
    yield w
    w.close()


def _digests(path) -> dict[str, str]:
    """One full open/compute/close cycle, as a fresh run would do."""
    w = Warehouse(path)
    try:
        return {
            "correlation": hashlib.sha256(_canon(build_correlation_report(w))).hexdigest(),
            "narrative": hashlib.sha256(_canon(build_case_narrative(w))).hexdigest(),
        }
    finally:
        w.close()


def test_correlation_and_narrative_are_byte_identical_across_runs(wh):
    """R1. Five runs over the same warehouse, one digest each."""
    runs = [_digests(wh.path) for _ in range(_RUNS)]
    for output in ("correlation", "narrative"):
        digests = {r[output] for r in runs}
        assert len(digests) == 1, (
            f"{output} differed across {_RUNS} runs on identical evidence: {digests}")


def test_the_fixture_actually_exercises_ties(wh):
    """Guard on the guard: if the fixture stopped containing tied scores and
    timestamps, the test above would still pass while testing nothing."""
    rows = wh.query(
        "SELECT count(*) AS n FROM ("
        "  SELECT record_ts FROM artifacts WHERE relevant "
        "  GROUP BY record_ts HAVING count(*) > 1)")
    assert rows[0]["n"] >= 2, "fixture no longer has tied timestamps"
    scores = wh.query("SELECT count(DISTINCT relevance_score) AS n FROM artifacts")
    assert scores[0]["n"] == 1, "fixture no longer has tied relevance scores"


def test_a_changed_warehouse_changes_the_digest(wh):
    """The digest has to be sensitive to what the system concluded, or the
    determinism test above would pass trivially."""
    before = _digests(wh.path)
    wh.load_artifacts([_art("a10", "Slack", "idb.user.name", "tyler_mia_2026 again", 7)])
    after = _digests(wh.path)
    assert after != before, "the digest is insensitive to the evidence"
