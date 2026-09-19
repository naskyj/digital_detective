"""Cross-source correlation — the thing a per-row classifier structurally can't do.

The classifier (dd_lab/classify) labels one artifact at a time. Correlation
looks *across* every artifact for the same identifier (a Discord user id, a
Roblox asset id, an invite code, an email) showing up in more than one app or
more than one storage mechanism — e.g. the same id in a Discord Cache JSON
blob, a raw memory-string carve, and an IndexedDB message row. That link is
the actual investigative signal; no single row states it.

Matching is exact-value for short identifiers and substring (ILIKE) for long
ones, since a longer id/invite code embedded inside an unflattened JSON blob
(a Cache body under the flatten-size cap, see dd_lab/parsers/jsonflat.py)
won't equal the isolated value elsewhere byte-for-byte. Every result carries
the full citation columns (artifact_id, app, source_file, source_locator) so a
reader can go straight to the original bytes — this module does not classify
or invent, only look up and group what's already in the warehouse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from dd_lab.etl.warehouse import Warehouse
from dd_lab.investigate.entities import extract_entities

_CITE_COLS = (
    "artifact_id, app, source_type, source_file, source_locator, "
    "key, value, category, record_ts"
)
# below this length an exact value match is required; substring search on a
# very short value would match almost everything and swamp the result with noise.
_SUBSTRING_MIN_LEN = 6


@dataclass
class Citation:
    artifact_id: str
    app: str
    source_type: str
    source_file: str
    source_locator: str
    key: str
    value: str
    category: str | None
    record_ts: datetime | None

    @classmethod
    def from_row(cls, row: dict) -> Citation:
        return cls(
            artifact_id=row["artifact_id"], app=row["app"], source_type=row["source_type"],
            source_file=row["source_file"], source_locator=row["source_locator"],
            key=row["key"], value=row["value"], category=row.get("category"),
            record_ts=row.get("record_ts"),
        )


@dataclass
class CorrelatedEntity:
    entity_type: str
    value: str
    artifacts: list[Citation] = field(default_factory=list)

    @property
    def apps(self) -> list[str]:
        return sorted({c.app for c in self.artifacts})

    @property
    def source_types(self) -> list[str]:
        return sorted({c.source_type for c in self.artifacts})

    def timeline(self) -> list[Citation]:
        """Artifacts ordered by when the record itself claims to have happened
        (falls back to the end for anything with no timestamp)."""
        return sorted(self.artifacts, key=lambda c: c.record_ts or datetime.max.replace(tzinfo=None))


def _related_rows(wh: Warehouse, value: str, limit: int) -> list[dict]:
    if len(value) >= _SUBSTRING_MIN_LEN:
        return wh.query(
            f"SELECT {_CITE_COLS} FROM artifacts WHERE value ILIKE ? "
            f"ORDER BY record_ts NULLS LAST LIMIT ?",
            [f"%{value}%", limit],
        )
    return wh.query(
        f"SELECT {_CITE_COLS} FROM artifacts WHERE value = ? "
        f"ORDER BY record_ts NULLS LAST LIMIT ?",
        [value, limit],
    )


def build_correlation_report(
    wh: Warehouse,
    *,
    candidate_row_limit: int = 5000,
    max_entities: int = 100,
    max_artifacts_per_entity: int = 50,
    min_distinct_sources: int = 2,
) -> list[CorrelatedEntity]:
    """Scan the higher-signal (`relevant`) artifacts for entities, then for
    each distinct one look across the *whole* warehouse (every category,
    every source type) for where else it appears. Only entities that actually
    turn up in more than one app or more than one storage mechanism are
    returned — that cross-source spread is the point; a value that only ever
    appears once has nothing to correlate.
    """
    candidates = wh.query(
        f"SELECT {_CITE_COLS} FROM artifacts WHERE relevant "
        f"ORDER BY confidence DESC NULLS LAST LIMIT ?",
        [candidate_row_limit],
    )

    seen: dict[tuple[str, str], CorrelatedEntity] = {}
    for row in candidates:
        if len(seen) >= max_entities:
            break
        for hit in extract_entities(row["app"], row["key"], row["value"] or ""):
            key = (hit.entity_type, hit.normalized)
            if key in seen:
                continue
            related = _related_rows(wh, hit.normalized, max_artifacts_per_entity)
            citations = [Citation.from_row(r) for r in related]
            entity = CorrelatedEntity(entity_type=hit.entity_type, value=hit.normalized, artifacts=citations)
            if len(entity.apps) >= min_distinct_sources or len(entity.source_types) >= min_distinct_sources:
                seen[key] = entity

    # most cross-referenced (widest spread) first
    return sorted(seen.values(), key=lambda e: (-len(e.artifacts), e.entity_type, e.value))
