"""Turning timestamped artifacts into a sequence of episodes, and episodes
into prose.

The problem this solves, stated plainly: `v_timeline` is
`SELECT ... WHERE record_ts IS NOT NULL ORDER BY record_ts`, and every
layer above it renders that as a flat table. A flat table of 600 rows is
not a reconstruction of events - it is the raw material for one. Nothing
in this project computed sessions, gaps, app hand-offs or coverage, so
nobody could say "what happened" without reading every row by hand.

Everything here is deterministic Python over the warehouse. No model, no
network, no API key. A tool that can only explain itself when a language
model is reachable is not a forensic tool, so the narrative floor has to
stand on its own; `dd_lab.agent` may write richer prose ON TOP of these
same computed facts, but never instead of them.

THREE KINDS OF TIME
-------------------
The single biggest correctness problem here is that `record_ts` does not
mean one thing. Measured on a real 675k-artifact case:

  EVENT   something happened at this moment. A cache entry's request_time,
          a cookie's creation or last-access time. These are the only rows
          a timeline may treat as events.

  EXPIRY  a validity boundary in the FUTURE - `cookie.<host>.<name>.
          expires_utc`. Nothing happened then. 14 of that case's 622
          timestamped relevant artifacts were expiry rows dated up to
          2027, and because a timeline sorts ascending they landed at the
          very END - exactly where a reader looks for "most recent
          activity". Worse than noise: actively misleading.

  STATE   no intrinsic time at all. The artifact was PRESENT when the
          evidence was acquired, and that is all that can be said. Memory
          strings, LevelDB/localStorage values, most config. On that same
          case this was 89% of all relevant artifacts - including almost
          all of the payment and credential material, which is the most
          probative evidence in the case.

That last number is why this module reports coverage instead of hiding it.
A narrative built from 11% of the evidence that silently implies it covers
all of it is an untrue statement about a case.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum

from dd_lab.etl.warehouse import Warehouse

#: Keys whose timestamp is a future validity boundary, not an event.
#: Anchored at the end because these are always the final `.`-segment the
#: cookie parser appends (`parsers/cookies.py:72`).
_EXPIRY_KEY_RE = re.compile(r"\.(expires?_utc|expiry|expires_at|expiration)$", re.I)

#: Gap at which one episode ends and the next begins. Not a round number
#: picked for looking reasonable - measured on the real case's own
#: inter-event gaps (607 gaps between relevant, non-expiry events):
#:
#:     0 min      537      |  31-60 min     2
#:     1-5 min     44      |  61-180 min    4
#:     6-30 min    19      |  >180 min      1   (longest: 1,184 min)
#:
#: 600 of 607 gaps fall at or below 30 minutes and then the distribution
#: falls off a cliff. The boundary sits in the empty space, so episodes
#: break where the user actually stopped, not where a constant says they
#: did. Re-measure on a case whose rhythm looks different - it is a
#: parameter everywhere it is used.
DEFAULT_SESSION_GAP = timedelta(minutes=30)

#: Categories worth naming in prose when they appear in an episode. Order
#: is rules.py's own CATEGORY_PRIORITY - most sensitive first - so the
#: sentence leads with the most serious thing present.
_NOTABLE_CATEGORIES = (
    "CREDENTIAL", "PAYMENT_INFO", "LOCATION_INFO", "CHAT_CONTENT",
    "CONTACT_PII", "USER", "MEDIA", "NAVIGATION_ACTIVITY",
)


class TimeKind(StrEnum):
    EVENT = "event"
    EXPIRY = "expiry"
    STATE = "state"


def time_kind(key: str, record_ts: datetime | None) -> TimeKind:
    """Which of the three kinds of time this artifact's timestamp is."""
    if record_ts is None:
        return TimeKind.STATE
    if _EXPIRY_KEY_RE.search(key or ""):
        return TimeKind.EXPIRY
    return TimeKind.EVENT


@dataclass(frozen=True)
class TimelineEvent:
    artifact_id: str
    ts: datetime
    app: str
    source_type: str
    category: str
    key: str
    value: str
    relevant: bool
    relevance_score: float
    deleted: bool
    source_file: str
    source_locator: str

    @classmethod
    def from_row(cls, r: dict) -> TimelineEvent:
        return cls(
            artifact_id=r["artifact_id"], ts=r["record_ts"], app=r["app"] or "",
            source_type=r["source_type"] or "", category=r["category"] or "",
            key=r["key"] or "", value=r["value"] or "",
            relevant=bool(r["relevant"]), relevance_score=float(r["relevance_score"] or 0.0),
            deleted=bool(r["deleted"]), source_file=r["source_file"] or "",
            source_locator=r["source_locator"] or "",
        )


@dataclass
class Coverage:
    """What fraction of the evidence the narrative can actually speak to.

    Reported in the output rather than computed and discarded, because the
    honest version of "here is what happened" includes how much of the case
    it is silent about.
    """
    total: int = 0
    events: int = 0
    expiry: int = 0
    state: int = 0

    @property
    def event_pct(self) -> float:
        return round(100.0 * self.events / self.total, 1) if self.total else 0.0

    def summary(self) -> str:
        if not self.total:
            return "No artifacts to report on."
        return (
            f"{self.events:,} of {self.total:,} artifacts ({self.event_pct}%) carry an event "
            f"timestamp and appear below. {self.state:,} have no intrinsic time and are present "
            f"as state at acquisition, not as events"
            + (f"; {self.expiry:,} are future validity boundaries (cookie expiry) and are "
               f"excluded from the sequence entirely" if self.expiry else "")
            + "."
        )


@dataclass
class Episode:
    """One continuous stretch of activity, bounded by DEFAULT_SESSION_GAP."""
    events: list[TimelineEvent] = field(default_factory=list)
    gap_before: timedelta | None = None

    @property
    def start(self) -> datetime:
        return self.events[0].ts

    @property
    def end(self) -> datetime:
        return self.events[-1].ts

    @property
    def duration(self) -> timedelta:
        return self.end - self.start

    @property
    def apps(self) -> list[str]:
        """Apps in order of FIRST APPEARANCE, not alphabetical - the order
        is the finding. "Roblox, then Discord joins 40 minutes in" is a
        different event from "Discord, then Roblox"."""
        out: list[str] = []
        for e in self.events:
            if e.app and e.app not in out:
                out.append(e.app)
        return out

    @property
    def categories(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self.events:
            if e.category:
                counts[e.category] = counts.get(e.category, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    @property
    def deleted_count(self) -> int:
        return sum(1 for e in self.events if e.deleted)

    def app_entry_times(self) -> dict[str, datetime]:
        first: dict[str, datetime] = {}
        for e in self.events:
            if e.app and e.app not in first:
                first[e.app] = e.ts
        return first

    def notable(self, limit: int = 5) -> list[TimelineEvent]:
        """The events a reader should look at first: most sensitive
        category, then highest relevance, then earliest. Deleted records
        sort to the very front - a recovered record is the strongest
        single signal in the schema and nothing else in this project
        surfaces it."""
        prio = {c: i for i, c in enumerate(_NOTABLE_CATEGORIES)}
        return sorted(
            self.events,
            key=lambda e: (not e.deleted, prio.get(e.category, 99), -e.relevance_score, e.ts),
        )[:limit]


def load_events(
    wh: Warehouse, *, relevant_only: bool = True, limit: int = 5000
) -> tuple[list[TimelineEvent], Coverage]:
    """Every artifact that is genuinely an EVENT, oldest first, plus the
    coverage figures for everything that is not.

    Coverage is counted over the same population the events are drawn from
    (relevant-only or not), so the percentage means what it says.
    """
    where = "WHERE relevant" if relevant_only else ""
    counts = wh.query(
        f"SELECT count(*) AS total, "  # noqa: S608 - `where` is a literal, not user input
        f"count(*) FILTER (WHERE record_ts IS NULL) AS state, "
        f"count(*) FILTER (WHERE record_ts IS NOT NULL "
        f"                 AND regexp_matches(key, '\\.(expires?_utc|expiry|expires_at|expiration)$')"
        f"                 ) AS expiry "
        f"FROM artifacts {where}"
    )[0]
    rows = wh.query(
        f"SELECT artifact_id, record_ts, app, source_type, category, key, "  # noqa: S608
        f"substr(value, 1, 400) AS value, relevant, relevance_score, deleted, "
        f"source_file, source_locator FROM artifacts {where} "
        f"{'AND' if where else 'WHERE'} record_ts IS NOT NULL "
        f"AND NOT regexp_matches(key, '\\.(expires?_utc|expiry|expires_at|expiration)$') "
        f"ORDER BY record_ts LIMIT {int(limit)}"
    )
    events = [TimelineEvent.from_row(r) for r in rows]
    cov = Coverage(
        total=int(counts["total"]), events=len(events),
        expiry=int(counts["expiry"]), state=int(counts["state"]),
    )
    return events, cov


def build_episodes(
    events: list[TimelineEvent], gap: timedelta = DEFAULT_SESSION_GAP
) -> list[Episode]:
    """Split a time-ordered event list wherever activity stops for `gap`."""
    episodes: list[Episode] = []
    current: Episode | None = None
    prev_ts: datetime | None = None
    for e in events:
        delta = (e.ts - prev_ts) if prev_ts is not None else None
        if current is None or (delta is not None and delta > gap):
            current = Episode(events=[], gap_before=delta)
            episodes.append(current)
        current.events.append(e)
        prev_ts = e.ts
    return episodes


# ----------------------------------------------------------------- prose


def _fmt_duration(d: timedelta) -> str:
    mins = int(d.total_seconds() // 60)
    if mins < 1:
        return "under a minute"
    if mins < 60:
        return f"{mins} minute{'s' if mins != 1 else ''}"
    h, m = divmod(mins, 60)
    if h < 24:
        return f"{h}h {m:02d}m"
    d_, h = divmod(h, 24)
    return f"{d_}d {h}h"


def _fmt_gap(d: timedelta) -> str:
    return _fmt_duration(d)


def _join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def narrate_episode(ep: Episode, index: int) -> str:
    """One paragraph describing a single episode.

    Deliberately states only what was computed - counts, times, ordering,
    categories present. No inference about intent or wrongdoing: that is
    the investigator's call, and the same rule the agent prompt already
    imposes on findings applies with more force to prose that reads like a
    story.
    """
    day = ep.start.strftime("%a %d %b %Y")
    span = (
        f"{ep.start.strftime('%H:%M')}-{ep.end.strftime('%H:%M')}"
        if ep.duration.total_seconds() >= 60 else ep.start.strftime("%H:%M")
    )
    bits = [f"**Episode {index}** - {day}, {span} ({_fmt_duration(ep.duration)}), "
            f"{len(ep.events):,} event{'s' if len(ep.events) != 1 else ''}."]

    if ep.gap_before is not None:
        bits.append(f"Follows {_fmt_gap(ep.gap_before)} of no recorded activity.")

    apps = ep.apps
    if len(apps) == 1:
        bits.append(f"All activity is in {apps[0]}.")
    elif apps:
        entry = ep.app_entry_times()
        first = apps[0]
        joins = [
            f"{a} from {entry[a].strftime('%H:%M')}"
            for a in apps[1:]
        ]
        bits.append(
            f"Begins in {first}, with {_join(joins)} - {len(apps)} applications active "
            f"in the same period."
        )

    cats = ep.categories
    named = [f"{c.replace('_', ' ').lower()} ({cats[c]})" for c in _NOTABLE_CATEGORIES if c in cats]
    if named:
        bits.append(f"Categories present: {_join(named[:5])}.")

    if ep.deleted_count:
        bits.append(
            f"{ep.deleted_count} of these are RECOVERED DELETED records."
        )
    return " ".join(bits)


def narrate(
    episodes: list[Episode], coverage: Coverage, *, max_episodes: int = 12
) -> str:
    """The case-level narrative: what happened, in order, with the limits
    of the evidence stated up front rather than buried."""
    if not episodes:
        return (
            "No timestamped events to reconstruct a sequence from. "
            + coverage.summary()
        )

    all_events = [e for ep in episodes for e in ep.events]
    start, end = episodes[0].start, episodes[-1].end
    apps: list[str] = []
    for e in all_events:
        if e.app and e.app not in apps:
            apps.append(e.app)

    by_day: dict[str, int] = {}
    for e in all_events:
        d = e.ts.strftime("%Y-%m-%d")
        by_day[d] = by_day.get(d, 0) + 1
    busiest_day, busiest_n = max(by_day.items(), key=lambda kv: kv[1])
    busiest_pct = round(100.0 * busiest_n / len(all_events))

    longest = max(episodes, key=lambda ep: ep.duration)
    multi_app = [ep for ep in episodes if len(ep.apps) > 1]

    out = ["## What happened", ""]
    out.append(
        f"Recorded activity runs from {start.strftime('%a %d %b %Y %H:%M')} to "
        f"{end.strftime('%a %d %b %Y %H:%M')}, in {len(episodes)} "
        f"episode{'s' if len(episodes) != 1 else ''} of continuous use separated by gaps of "
        f"{_fmt_duration(DEFAULT_SESSION_GAP)} or more, across {_join(apps)}."
    )
    out.append(
        f"Activity is concentrated on {busiest_day} ({busiest_n:,} of {len(all_events):,} events, "
        f"{busiest_pct}%). The longest continuous episode runs "
        f"{_fmt_duration(longest.duration)} from {longest.start.strftime('%d %b %H:%M')}."
    )
    if multi_app:
        out.append(
            f"{len(multi_app)} of {len(episodes)} episodes involve more than one application "
            f"in the same sitting - the periods where activity in one app can be placed "
            f"alongside activity in another."
        )
    deleted_total = sum(ep.deleted_count for ep in episodes)
    if deleted_total:
        out.append(f"{deleted_total} recovered deleted record(s) fall inside these episodes.")

    out += ["", f"*Coverage: {coverage.summary()}*", ""]

    for i, ep in enumerate(episodes[:max_episodes], start=1):
        out += ["", narrate_episode(ep, i)]
    if len(episodes) > max_episodes:
        out.append(f"\n*({len(episodes) - max_episodes} further episode(s) not shown.)*")
    return "\n".join(out)


def build_case_narrative(
    wh: Warehouse, *, relevant_only: bool = True, gap: timedelta = DEFAULT_SESSION_GAP,
    limit: int = 5000, max_episodes: int = 12,
) -> dict:
    """One call for callers that just want the story and its structure."""
    events, coverage = load_events(wh, relevant_only=relevant_only, limit=limit)
    episodes = build_episodes(events, gap)
    return {
        "narrative": narrate(episodes, coverage, max_episodes=max_episodes),
        "coverage": {
            "total": coverage.total, "events": coverage.events,
            "expiry": coverage.expiry, "state": coverage.state,
            "event_pct": coverage.event_pct,
        },
        "episode_count": len(episodes),
        "episodes": [
            {
                "index": i,
                "start": ep.start, "end": ep.end,
                "duration_minutes": round(ep.duration.total_seconds() / 60, 1),
                "gap_before_minutes": (
                    round(ep.gap_before.total_seconds() / 60, 1) if ep.gap_before else None
                ),
                "event_count": len(ep.events),
                "apps": ep.apps,
                "categories": ep.categories,
                "deleted_count": ep.deleted_count,
                "summary": narrate_episode(ep, i),
                "notable": [
                    {
                        "artifact_id": e.artifact_id, "ts": e.ts, "app": e.app,
                        "category": e.category, "key": e.key, "value": e.value[:200],
                        "relevance_score": e.relevance_score, "deleted": e.deleted,
                        "source_file": e.source_file, "source_locator": e.source_locator,
                    }
                    for e in ep.notable()
                ],
            }
            for i, ep in enumerate(episodes, start=1)
        ],
    }
