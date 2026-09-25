"""Phase 2 — investigation reasoning.

Everything here is deterministic Python (same principle as classify/ and
etl/): it never invents a fact, only finds facts already sitting in the
warehouse and links them. An LLM agent may use these tools to *narrate* a
hypothesis, but the linking, ordering, and verification are code, and every
result carries the citation (artifact_id, app, source_file, source_locator)
needed to trace it back to the original evidence byte-for-byte.
"""

from __future__ import annotations

from dd_lab.investigate.correlate import CorrelatedEntity, build_correlation_report
from dd_lab.investigate.entities import EntityHit, extract_entities
from dd_lab.investigate.timeline import (
    Coverage,
    Episode,
    TimeKind,
    TimelineEvent,
    build_case_narrative,
    build_episodes,
    load_events,
    narrate,
    time_kind,
)
from dd_lab.investigate.verify import ClaimVerification, verify_claim

__all__ = [
    "ClaimVerification",
    "CorrelatedEntity",
    "Coverage",
    "EntityHit",
    "Episode",
    "TimeKind",
    "TimelineEvent",
    "build_case_narrative",
    "build_correlation_report",
    "build_episodes",
    "extract_entities",
    "load_events",
    "narrate",
    "time_kind",
    "verify_claim",
]
