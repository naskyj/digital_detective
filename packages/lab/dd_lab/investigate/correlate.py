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

WHY AN ENTITY IS RANKED WHERE IT IS
-----------------------------------
Ranking used to be `-len(e.artifacts)` — raw citation count. That was not a
weak heuristic, it was no heuristic at all: `_related_rows` caps at
`max_artifacts_per_entity` (50), so every entity occurring 50+ times tied at
exactly 50 and the sort silently degenerated to alphabetical by value. On a
real case that put `growtopia`, `leagueoflegends` and `minecraft` — vanity
names cached from Discord's own server-discovery directory — above the two
genuine invite codes in the evidence.

Measured on that case, raw frequency does not separate signal from noise,
but two other things do:

    value             rows  apps  src types  relevant
    leagueoflegends     13     1      2       1  ( 8%)
    growtopia           11     1      2       1  ( 9%)
    RxPTgtyY            10     1      2       8  (80%)   <- a real invite
    T0C1J4SDLG2         67     3      5       3          <- Slack workspace

`RxPTgtyY` and `leagueoflegends` have near-identical occurrence counts. What
separates them is the share of those occurrences the classifier independently
flagged relevant — 80% against 8%. And what makes `T0C1J4SDLG2` matter is
spread: three applications and five storage mechanisms.

So significance is built from spread, independent corroboration, recovered
deleted records and entity kind — never from raw count, which is the one
signal shown not to discriminate. Every entity carries `reasons()`, the
plain-English version of its own score, so a reader is never asked to trust
an unexplained number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

from dd_lab.etl.warehouse import Warehouse
from dd_lab.investigate.entities import extract_entities

_CITE_COLS = (
    "artifact_id, app, source_type, source_file, source_locator, "
    "key, value, category, record_ts, relevant, relevance_score, deleted"
)
# below this length an exact value match is required; substring search on a
# very short value would match almost everything and swamp the result with noise.
_SUBSTRING_MIN_LEN = 6

#: How much an entity kind is worth before any evidence about it. A
#: credential-bearing token is a finding on sight; an asset id is a breadcrumb.
#: Unlisted types get the default. These are priors, deliberately small
#: relative to the spread and corroboration terms below - what an entity IS
#: matters less than where it turns up and whether anything corroborates it.
_TYPE_WEIGHT = {
    "slack_token": 3.0,
    "email": 2.0,
    "discord_user_id": 1.5,
    "roblox_user_id": 1.5,
    "slack_user_id": 1.5,
    "slack_team_id": 1.0,
    "discord_invite": 1.0,
    "roblox_asset_id": 0.5,
}
_DEFAULT_TYPE_WEIGHT = 1.0


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
    # Relevance is a PER-CITATION signal, deliberately kept alongside
    # category rather than used to exclude a citation outright - an entity
    # is seeded only from relevant artifacts (see build_correlation_report),
    # but once seeded, _related_rows() deliberately looks across the WHOLE
    # warehouse for corroborating occurrences, relevant or not (recall
    # matters more than precision for forensic evidence - this project's
    # consistent choice elsewhere too: deleted records kept not deleted,
    # superseded values kept not dropped). These fields let a reader (or
    # the agent) tell a genuinely relevant corroborating hit apart from an
    # incidental match on the same value, instead of losing that
    # distinction entirely.
    relevant: bool | None
    relevance_score: float | None
    #: Whether this occurrence sits in a RECOVERED DELETED record. Present
    #: in the schema since the deleted-record work but carried by no
    #: citation, no timeline and no report section until now - despite
    #: being the strongest single flag available: someone tried to remove
    #: it and it survived.
    deleted: bool = False

    @classmethod
    def from_row(cls, row: dict) -> Citation:
        return cls(
            artifact_id=row["artifact_id"], app=row["app"], source_type=row["source_type"],
            source_file=row["source_file"], source_locator=row["source_locator"],
            key=row["key"], value=row["value"], category=row.get("category"),
            record_ts=row.get("record_ts"),
            relevant=row.get("relevant"), relevance_score=row.get("relevance_score"),
            deleted=bool(row.get("deleted")),
        )


@dataclass
class CoOccurrence:
    """Another entity found in the SAME artifact as this one.

    This is the "how are things connected" layer, and it did not exist:
    `extract_entities` returns a list per row, so the fact that one artifact
    yielded both an email and a Discord user id was discarded on the next
    loop iteration. Two identifiers sharing an artifact is the raw material
    for identity resolution - the same person's handle in two systems -
    and nothing was keeping it.
    """
    entity_type: str
    value: str
    shared_artifacts: int


@dataclass
class CorrelatedEntity:
    entity_type: str
    value: str
    artifacts: list[Citation] = field(default_factory=list)
    #: TRUE totals across the warehouse, not len(artifacts) - that is capped
    #: by max_artifacts_per_entity and so saturates. Obtained from the same
    #: query via a window function, which is evaluated before LIMIT.
    occurrence_count: int = 0
    relevant_count: int = 0
    deleted_count: int = 0
    co_occurring: list[CoOccurrence] = field(default_factory=list)

    @property
    def apps(self) -> list[str]:
        return sorted({c.app for c in self.artifacts})

    @property
    def source_types(self) -> list[str]:
        return sorted({c.source_type for c in self.artifacts})

    @property
    def cross_app(self) -> bool:
        """The genuinely interesting case, and previously indistinguishable
        from the mundane one: `min_distinct_sources` was tested as
        `apps >= n OR source_types >= n`, so "the same id in discord/cache
        and discord/indexeddb" qualified identically to "the same id in
        Discord AND Slack". One is a storage detail; the other places one
        identifier in two different applications."""
        return len(self.apps) >= 2

    @property
    def relevant_ratio(self) -> float:
        return (self.relevant_count / self.occurrence_count) if self.occurrence_count else 0.0

    @property
    def memory_only(self) -> bool:
        """Present in memory and never written to disk. Worth flagging on
        its own: it was live in a process at acquisition but left no
        persisted trace, so a disk-only examination would have missed it
        entirely."""
        st = self.source_types
        return bool(st) and all(s.startswith("memory") for s in st)

    @property
    def significance(self) -> float:
        """Never raw count - see the module docstring for the measurement
        showing count does not discriminate."""
        score = _TYPE_WEIGHT.get(self.entity_type, _DEFAULT_TYPE_WEIGHT)
        score += 3.0 * max(0, len(self.apps) - 1)          # spread across applications
        score += 1.0 * max(0, len(self.source_types) - 1)  # spread across storage
        score += 4.0 * self.relevant_ratio                 # independent corroboration
        if self.deleted_count:
            score += 3.0                                    # survived an attempted deletion
        if self.memory_only:
            score += 1.0
        score += min(2.0, 0.5 * len(self.co_occurring))     # connected to other identifiers
        # A long tail of occurrences adds a little, with heavy diminishing
        # returns, so it can break ties without ever dominating.
        score += min(1.0, math.log10(1 + self.occurrence_count) / 2)
        return round(score, 2)

    def reasons(self) -> list[str]:
        """Why this entity is ranked where it is, in plain English.

        The point of the report is to tell an investigator something they
        would not otherwise have noticed. A bare score does the opposite, so
        every contribution above gets a sentence.
        """
        out: list[str] = []
        if self.cross_app:
            out.append(
                f"Appears in {len(self.apps)} different applications "
                f"({', '.join(self.apps)}) — the same identifier links them."
            )
        if len(self.source_types) >= 2:
            out.append(
                f"Found in {len(self.source_types)} storage mechanisms "
                f"({', '.join(self.source_types)}), so it survived in more than one place."
            )
        if self.deleted_count:
            out.append(
                f"{self.deleted_count} occurrence(s) are in RECOVERED DELETED records — "
                f"present despite an attempt to remove them."
            )
        if self.memory_only:
            out.append(
                "Only ever seen in memory, never written to disk — a disk-only "
                "examination would not have found this."
            )
        if self.occurrence_count and self.relevant_ratio >= 0.5:
            out.append(
                f"{self.relevant_count} of {self.occurrence_count} occurrences "
                f"({round(100 * self.relevant_ratio)}%) were independently assessed relevant."
            )
        elif self.occurrence_count and self.relevant_ratio <= 0.15:
            out.append(
                f"Only {self.relevant_count} of {self.occurrence_count} occurrences "
                f"({round(100 * self.relevant_ratio)}%) were independently assessed relevant — "
                f"consistent with incidental or cached content rather than user activity."
            )
        if self.co_occurring:
            named = ", ".join(
                f"{c.value} ({c.entity_type.replace('_', ' ')}, {c.shared_artifacts} shared)"
                for c in self.co_occurring[:3]
            )
            out.append(f"Occurs in the same artifact as {named}.")
        ts = [c.record_ts for c in self.artifacts if c.record_ts]
        if len(ts) >= 2:
            lo, hi = min(ts), max(ts)
            if lo != hi:
                out.append(
                    f"Occurrences span {lo:%d %b %Y %H:%M} to {hi:%d %b %Y %H:%M}."
                )
        return out

    def timeline(self) -> list[Citation]:
        """Artifacts ordered by when the record itself claims to have happened
        (falls back to the end for anything with no timestamp)."""
        return sorted(self.artifacts, key=lambda c: c.record_ts or datetime.max.replace(tzinfo=None))


def _related_rows(wh: Warehouse, value: str, limit: int) -> list[dict]:
    """Citations plus the TRUE totals.

    The three `count(*) OVER ()` window terms are evaluated before LIMIT, so
    they report the real number of occurrences even though only `limit` rows
    come back - which is what makes an uncapped occurrence_count free rather
    than a second full scan per entity.
    """
    totals = (
        "count(*) OVER () AS _total, "
        "count(*) FILTER (WHERE relevant) OVER () AS _relevant, "
        "count(*) FILTER (WHERE deleted) OVER () AS _deleted"
    )
    if len(value) >= _SUBSTRING_MIN_LEN:
        return wh.query(
            f"SELECT {_CITE_COLS}, {totals} FROM artifacts WHERE value ILIKE ? "
            f"ORDER BY record_ts NULLS LAST LIMIT ?",
            [f"%{value}%", limit],
        )
    return wh.query(
        f"SELECT {_CITE_COLS}, {totals} FROM artifacts WHERE value = ? "
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

    Results are ordered by `significance` (see the module docstring), not by
    citation count.
    """
    candidates = wh.query(
        f"SELECT {_CITE_COLS} FROM artifacts WHERE relevant "
        f"ORDER BY confidence DESC NULLS LAST LIMIT ?",
        [candidate_row_limit],
    )

    # Co-occurrence is collected during the same scan: two entities pulled
    # out of ONE artifact are, by construction, connected in the evidence.
    pair_counts: dict[tuple[tuple[str, str], tuple[str, str]], int] = {}
    seen: dict[tuple[str, str], CorrelatedEntity] = {}
    rejected: set[tuple[str, str]] = set()

    for row in candidates:
        hits = extract_entities(row["app"], row["key"], row["value"] or "")
        keys = sorted({(h.entity_type, h.normalized) for h in hits})
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

        if len(seen) >= max_entities:
            continue  # keep counting co-occurrence, just stop adding entities
        for key in keys:
            # `rejected` matters for cost, not correctness: without it a
            # common non-qualifying value re-runs its full-table ILIKE scan
            # once per candidate row that mentions it.
            if key in seen or key in rejected:
                continue
            entity_type, value = key
            related = _related_rows(wh, value, max_artifacts_per_entity)
            citations = [Citation.from_row(r) for r in related]
            entity = CorrelatedEntity(
                entity_type=entity_type, value=value, artifacts=citations,
                occurrence_count=int(related[0]["_total"]) if related else 0,
                relevant_count=int(related[0]["_relevant"]) if related else 0,
                deleted_count=int(related[0]["_deleted"]) if related else 0,
            )
            if (len(entity.apps) >= min_distinct_sources
                    or len(entity.source_types) >= min_distinct_sources):
                seen[key] = entity
            else:
                rejected.add(key)

    # Both endpoints must themselves be correlated entities. A link to
    # something that appears in only one place is a weaker claim and would
    # also mean the report references entities it never lists - so pairs are
    # counted for everything and kept only where both ends survived.
    for (a, b), n in pair_counts.items():
        if a in seen and b in seen:
            seen[a].co_occurring.append(CoOccurrence(b[0], b[1], n))
            seen[b].co_occurring.append(CoOccurrence(a[0], a[1], n))
    for e in seen.values():
        e.co_occurring.sort(key=lambda c: (-c.shared_artifacts, c.entity_type, c.value))

    return sorted(
        seen.values(), key=lambda e: (-e.significance, e.entity_type, e.value)
    )
