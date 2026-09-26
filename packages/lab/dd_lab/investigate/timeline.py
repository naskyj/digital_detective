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
from datetime import UTC, datetime, timedelta
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
    #: Time read out of the artifact's OWN CONTENT rather than recorded by
    #: the application - see derive_time(). Kept as a separate kind, never
    #: merged into EVENT, because the two support different claims: an
    #: EVENT time is what the app itself wrote down, an INFERRED one is what
    #: an identifier embedded in the data happens to encode.
    INFERRED = "inferred"


#: Discord ids are snowflakes: the top 42 bits are milliseconds since
#: 2015-01-01. Decoding one gives the exact moment that OBJECT was created -
#: for a message id that is when the message was sent.
DISCORD_EPOCH_MS = 1_420_070_400_000
_SNOWFLAKE_RE = re.compile(r"\b(1[0-9]{17,18})\b")
_ISO_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}(?::\d{2})?)")
_EPOCH_MS_RE = re.compile(r"\b(\d{13})\b")

#: Nothing Discord-shaped can predate Discord itself, so anything decoding
#: before this is a number that merely looked like a snowflake.
_MIN_PLAUSIBLE = datetime(2015, 1, 1)


def derive_time(value: str, *, not_after: datetime) -> tuple[datetime, str] | None:
    """A timestamp recoverable from the artifact's own content, and what it
    was recovered from - or None.

    Measured on a real case's 4,948 undated relevant memory strings: 505
    (10.2%) carry a Discord snowflake, 97 an epoch-ms value, 24 an ISO
    datetime. The rest contain no time at all - they are JS fragments,
    tokens, URLs and config, so this is a real ceiling and not a parser
    shortcoming.

    `not_after` is the plausibility bound and is derived from the case's own
    data by the caller, not from the clock - a forensic result must not
    depend on when it was run. It exists because 18-digit numbers that are
    not snowflakes decode to nonsense: on that same case 505 snowflakes
    yielded two dates in 2027-2028, alongside 55 genuine-but-old 2023 ids.
    Old ids are real and are kept; impossible ones are not.
    """
    if not value:
        return None
    if m := _ISO_RE.search(value):
        try:
            iso = f"{m.group(1)}T{m.group(2)}"
            dt = datetime.fromisoformat(iso if len(m.group(2)) > 5 else iso + ":00")
            if _MIN_PLAUSIBLE <= dt <= not_after:
                return dt, "iso8601"
        except ValueError:
            pass
    if m := _SNOWFLAKE_RE.search(value):
        dt = datetime.fromtimestamp(
            ((int(m.group(1)) >> 22) + DISCORD_EPOCH_MS) / 1000, tz=UTC
        ).replace(tzinfo=None)
        if _MIN_PLAUSIBLE <= dt <= not_after:
            return dt, "discord_snowflake"
    if m := _EPOCH_MS_RE.search(value):
        dt = datetime.fromtimestamp(int(m.group(1)) / 1000, tz=UTC).replace(tzinfo=None)
        if _MIN_PLAUSIBLE <= dt <= not_after:
            return dt, "epoch_ms"
    return None


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
    #: How this event's time was obtained. EVENT means the application
    #: recorded it; INFERRED means it was decoded from the artifact's own
    #: content, and `time_source` says from what.
    kind: TimeKind = TimeKind.EVENT
    time_source: str = "recorded"

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
    #: State artifacts whose time was recovered from their own content.
    #: Counted apart from `events` throughout - they are a weaker claim.
    inferred: int = 0

    @property
    def placed(self) -> int:
        """Everything the narrative can position in time at all."""
        return self.events + self.inferred

    @property
    def event_pct(self) -> float:
        """Deliberately the RECORDED fraction, not `placed` - the headline
        number stays the strongest claim, and inferred times are reported
        separately rather than quietly inflating it."""
        return round(100.0 * self.events / self.total, 1) if self.total else 0.0

    def summary(self) -> str:
        if not self.total:
            return "No artifacts to report on."
        return (
            f"{self.events:,} of {self.total:,} artifacts ({self.event_pct}%) carry an event "
            f"timestamp and appear below"
            + (f", a further {self.inferred:,} placed in time by decoding an identifier in "
               f"their own content (marked as inferred)" if self.inferred else "")
            + f". {self.state:,} have no intrinsic time and are present "
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

    @property
    def inferred_count(self) -> int:
        """Events positioned by decoding their own content rather than by a
        time the application recorded. Surfaced per episode so a reader can
        see how much of an episode rests on the weaker claim."""
        return sum(1 for e in self.events if e.kind is TimeKind.INFERRED)

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
    wh: Warehouse, *, relevant_only: bool = True, limit: int = 5000,
    include_inferred: bool = True,
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

    if include_inferred and events:
        # The plausibility bound comes from the case's own recorded events,
        # never from the clock: a forensic result must not depend on when it
        # was run. A month of slack past the last recorded event is enough to
        # admit genuine activity the parsers did not date, while rejecting
        # the 18-digit numbers that merely look like snowflakes (two of the
        # real case's 505 decoded to 2027 and 2028).
        not_after = max(e.ts for e in events) + timedelta(days=30)
        undated = wh.query(
            f"SELECT artifact_id, app, source_type, category, key, "  # noqa: S608
            f"substr(value, 1, 400) AS value, relevant, relevance_score, deleted, "
            f"source_file, source_locator FROM artifacts {where} "
            f"{'AND' if where else 'WHERE'} record_ts IS NULL LIMIT {int(limit)}"
        )
        for r in undated:
            got = derive_time(r["value"] or "", not_after=not_after)
            if got is None:
                continue
            ts, src = got
            events.append(TimelineEvent(
                artifact_id=r["artifact_id"], ts=ts, app=r["app"] or "",
                source_type=r["source_type"] or "", category=r["category"] or "",
                key=r["key"] or "", value=r["value"] or "",
                relevant=bool(r["relevant"]),
                relevance_score=float(r["relevance_score"] or 0.0),
                deleted=bool(r["deleted"]), source_file=r["source_file"] or "",
                source_locator=r["source_locator"] or "",
                kind=TimeKind.INFERRED, time_source=src,
            ))
        cov.inferred = sum(1 for e in events if e.kind is TimeKind.INFERRED)
        events.sort(key=lambda e: e.ts)

    return events, cov


def attach_inferred(
    episodes: list[Episode], inferred: list[TimelineEvent],
    gap: timedelta = DEFAULT_SESSION_GAP,
) -> list[TimelineEvent]:
    """Fold inferred events into the recorded episodes they fall inside, and
    return the ones that fall outside every episode.

    Episodes are built from RECORDED times only, on purpose. An episode
    assembled out of inferred times alone is not a period of activity - it is
    a cluster of moments when objects that were later referenced happened to
    be created. On the real case that produced an "Episode 1 - Wed 13 Sep
    2023, 64 events" three years before the evidence, which reads as activity
    in 2023 and is not. Those are returned separately instead, so nothing is
    discarded and nothing is overstated: a 2023 message id sitting in memory
    is a genuine finding, just a different one.
    """
    outside: list[TimelineEvent] = []
    for e in inferred:
        for ep in episodes:
            if ep.start - gap <= e.ts <= ep.end + gap:
                ep.events.append(e)
                break
        else:
            outside.append(e)
    for ep in episodes:
        ep.events.sort(key=lambda x: x.ts)
    return sorted(outside, key=lambda x: x.ts)


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
    if ep.inferred_count:
        bits.append(
            f"{ep.inferred_count} of these {len(ep.events)} events were positioned by "
            f"decoding an identifier in the artifact itself, not by a time the application "
            f"recorded."
        )
    return " ".join(bits)


def narrate(
    episodes: list[Episode], coverage: Coverage, *, max_episodes: int = 12,
    outside: list[TimelineEvent] | None = None,
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

    inferred_total = sum(ep.inferred_count for ep in episodes)
    if inferred_total:
        out.append(
            f"{inferred_total:,} of the events below carry no time the application recorded; "
            f"they are positioned by decoding an identifier in the artifact itself (a Discord "
            f"snowflake, an embedded timestamp). That is a weaker claim than a recorded time, "
            f"and those events are marked as inferred throughout."
        )
    if outside:
        out.append(
            f"A further {len(outside):,} artifact(s) reference objects created between "
            f"{outside[0].ts:%d %b %Y} and {outside[-1].ts:%d %b %Y} - outside every period of "
            f"recorded activity above. These are NOT shown as activity: an object created in "
            f"the past being referenced now is a different fact from something happening then. "
            f"They are evidence that older material was present on the device."
        )

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
    recorded = [e for e in events if e.kind is TimeKind.EVENT]
    inferred = [e for e in events if e.kind is TimeKind.INFERRED]
    episodes = build_episodes(recorded, gap)
    outside = attach_inferred(episodes, inferred, gap)
    return {
        # Inferred times that land outside every recorded episode. Kept and
        # counted, never plotted as activity - see attach_inferred.
        "referenced_outside_window": {
            "count": len(outside),
            "earliest": outside[0].ts if outside else None,
            "latest": outside[-1].ts if outside else None,
            "examples": [
                {"artifact_id": e.artifact_id, "ts": e.ts, "app": e.app,
                 "time_source": e.time_source, "key": e.key, "value": e.value[:160]}
                for e in outside[:10]
            ],
        },
        "narrative": narrate(
            episodes, coverage, max_episodes=max_episodes, outside=outside),
        "coverage": {
            "total": coverage.total, "events": coverage.events,
            "expiry": coverage.expiry, "state": coverage.state,
            "inferred": coverage.inferred, "placed": coverage.placed,
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
                "inferred_count": ep.inferred_count,
                "summary": narrate_episode(ep, i),
                "notable": [
                    {
                        "artifact_id": e.artifact_id, "ts": e.ts, "app": e.app,
                        "category": e.category, "key": e.key, "value": e.value[:200],
                        "relevance_score": e.relevance_score, "deleted": e.deleted,
                        "time_kind": str(e.kind), "time_source": e.time_source,
                        "source_file": e.source_file, "source_locator": e.source_locator,
                    }
                    for e in ep.notable()
                ],
            }
            for i, ep in enumerate(episodes, start=1)
        ],
    }
