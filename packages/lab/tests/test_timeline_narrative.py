"""Episode reconstruction and the deterministic narrative.

Synthetic warehouse throughout - the point of these is that the narrative
is computed, reproducible and says only what the rows support, so they
must not depend on any real case, any model, or any network.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from dd_core.enums import ArtifactCategory as C
from dd_core.enums import SourceType as S
from dd_core.schema import ClassificationResult, NormalizedArtifact
from dd_lab.etl.warehouse import Warehouse
from dd_lab.investigate.timeline import (
    DEFAULT_SESSION_GAP,
    TimeKind,
    build_case_narrative,
    build_episodes,
    load_events,
    time_kind,
)

_DAY = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)


def _art(app, key, value, *, minutes=0, category=C.CHAT_CONTENT, ts=True, deleted=False,
         source_type=S.CACHE, relevant=True, score=0.9):
    when = _DAY + timedelta(minutes=minutes)
    return NormalizedArtifact(
        artifact_id=f"{app}:{key}:{minutes}:{value[:8]}",
        case_id="c1", app=app, source_type=source_type,
        source_file=f"{app}/Cache/f_{minutes}", source_locator=f"loc{minutes}",
        key=key, value=value,
        record_ts=when if ts else None,
        extracted_ts=_DAY, deleted=deleted,
        classification=ClassificationResult(
            category=category, confidence=0.9, classifier_version="test",
            relevant=relevant, relevance_score=score,
        ),
    )


@pytest.fixture
def wh(tmp_path):
    w = Warehouse(tmp_path / "case.duckdb")
    yield w
    w.close()


# ------------------------------------------------------------- time_kind


@pytest.mark.parametrize(
    ("key", "has_ts", "expected"),
    [
        ("cache.url", True, TimeKind.EVENT),
        ("cache.request_time", True, TimeKind.EVENT),
        ("cookie.discordapp.com.__dcfduid.last_access_utc", True, TimeKind.EVENT),
        ("cookie.discordapp.com.__dcfduid.creation_utc", True, TimeKind.EVENT),
        # future validity boundaries - nothing happened at these moments
        ("cookie.discordapp.com.__dcfduid.expires_utc", True, TimeKind.EXPIRY),
        ("cookie..roblox.com.ARID.expires_utc", True, TimeKind.EXPIRY),
        # no intrinsic time at all
        ("mem.string", False, TimeKind.STATE),
        ("local_storage.token", False, TimeKind.STATE),
    ],
)
def test_time_kind(key, has_ts, expected):
    assert time_kind(key, _DAY if has_ts else None) is expected


def test_expiry_rows_are_excluded_from_the_event_sequence(wh):
    """On the real case 14 of 622 timestamped relevant artifacts were cookie
    expiry dates running to 2027. A timeline sorts ascending, so they landed
    at the very END - where a reader looks for the most recent activity."""
    wh.load_artifacts([
        _art("discord", "cache.url", "https://discord.com/api/x", minutes=0),
        _art("discord", "cookie.discordapp.com.__dcfduid.expires_utc",
             "2027-10-18T12:21:51", minutes=60 * 24 * 400, category=C.TIMESTAMP),
        _art("discord", "cache.url", "https://discord.com/api/y", minutes=5),
    ])
    events, cov = load_events(wh)

    assert [e.key for e in events] == ["cache.url", "cache.url"]
    assert cov.expiry == 1
    assert cov.events == 2
    # and the narrative must not end in 2027
    assert "2027" not in build_case_narrative(wh)["narrative"]


def test_state_artifacts_are_counted_not_silently_dropped(wh):
    """89% of the real case's relevant artifacts have no timestamp, and they
    are the payment/credential material. A narrative covering 11% of the
    evidence must say so."""
    wh.load_artifacts(
        [_art("discord", "cache.url", "https://x", minutes=0)]
        + [_art("discord", "mem.string", f"secret{i}", minutes=i, ts=False,
                category=C.CREDENTIAL) for i in range(9)]
    )
    out = build_case_narrative(wh)

    assert out["coverage"] == {"total": 10, "events": 1, "expiry": 0, "state": 9,
                               "event_pct": 10.0}
    assert "9 have no intrinsic time" in out["narrative"]
    assert "10.0%" in out["narrative"]


# ------------------------------------------------------------- episodes


def test_episodes_split_on_the_gap_and_not_within_it(wh):
    gap_min = int(DEFAULT_SESSION_GAP.total_seconds() // 60)
    wh.load_artifacts([
        _art("discord", "cache.url", "a", minutes=0),
        _art("discord", "cache.url", "b", minutes=gap_min - 1),      # same episode
        _art("discord", "cache.url", "c", minutes=gap_min * 3),      # new episode
        _art("discord", "cache.url", "d", minutes=gap_min * 3 + 2),  # same as c
    ])
    events, _ = load_events(wh)
    eps = build_episodes(events)

    assert [len(e.events) for e in eps] == [2, 2]
    assert eps[0].gap_before is None
    assert eps[1].gap_before == timedelta(minutes=gap_min * 2 + 1)


def test_apps_are_ordered_by_first_appearance_not_alphabetically(wh):
    """"Slack first, then Discord joins 20 minutes in" is a different event
    from "Discord, then Slack". Alphabetical ordering destroys that."""
    wh.load_artifacts([
        _art("Slack", "cache.url", "a", minutes=0),
        _art("discord", "cache.url", "b", minutes=5),
        _art("Aardvark", "cache.url", "c", minutes=10),
    ])
    events, _ = load_events(wh)
    ep = build_episodes(events)[0]

    assert ep.apps == ["Slack", "discord", "Aardvark"]
    # DuckDB hands timestamps back naive; compare on the wall clock.
    assert ep.app_entry_times()["discord"].hour == 12
    assert ep.app_entry_times()["discord"].minute == 5


def test_narrative_names_the_app_handoff(wh):
    wh.load_artifacts([
        _art("Slack", "cache.url", "a", minutes=0),
        _art("discord", "cache.url", "b", minutes=20),
    ])
    text = build_case_narrative(wh)["narrative"]
    assert "Begins in Slack" in text
    assert "discord from" in text


def test_deleted_records_are_surfaced(wh):
    """`deleted` is arguably the highest-signal column in the schema and no
    timeline, citation or report section carried it before."""
    wh.load_artifacts([
        _art("discord", "cache.url", "ordinary", minutes=0, score=0.99),
        _art("discord", "idb.messages.content", "recovered message", minutes=1,
             deleted=True, score=0.30),
    ])
    out = build_case_narrative(wh)
    ep = out["episodes"][0]

    assert ep["deleted_count"] == 1
    assert "RECOVERED DELETED" in ep["summary"]
    # deleted sorts to the front of `notable` even with a far lower score
    assert ep["notable"][0]["deleted"] is True


def test_single_app_episode_says_so(wh):
    wh.load_artifacts([_art("discord", "cache.url", f"u{i}", minutes=i) for i in range(3)])
    ep = build_case_narrative(wh)["episodes"][0]
    assert "All activity is in discord." in ep["summary"]


def test_empty_case_produces_an_honest_statement_not_a_crash(wh):
    out = build_case_narrative(wh)
    assert out["episode_count"] == 0
    assert out["episodes"] == []
    assert "No timestamped events" in out["narrative"]


def test_state_only_case_does_not_claim_a_sequence(wh):
    """A memory-only case has no event times at all. It must not produce a
    narrative implying a reconstructed sequence."""
    wh.load_artifacts([
        _art("discord", "mem.string", f"s{i}", minutes=i, ts=False, category=C.CREDENTIAL)
        for i in range(5)
    ])
    out = build_case_narrative(wh)

    assert out["episode_count"] == 0
    assert "No timestamped events" in out["narrative"]
    assert out["coverage"]["state"] == 5


def test_relevant_only_is_honoured_in_both_events_and_coverage(wh):
    """Coverage is counted over the same population the events come from,
    so the percentage means what it says."""
    wh.load_artifacts([
        _art("discord", "cache.url", "keep", minutes=0, relevant=True),
        _art("discord", "cache.url", "drop", minutes=1, relevant=False, score=0.0),
    ])
    out_rel = build_case_narrative(wh, relevant_only=True)
    out_all = build_case_narrative(wh, relevant_only=False)

    assert out_rel["coverage"]["total"] == 1 and out_rel["coverage"]["events"] == 1
    assert out_all["coverage"]["total"] == 2 and out_all["coverage"]["events"] == 2


def test_notable_leads_with_the_most_sensitive_category(wh):
    wh.load_artifacts([
        _art("discord", "cache.url", "https://x", minutes=0, category=C.MEDIA, score=0.99),
        _art("discord", "local_storage.token", "hunter2", minutes=1,
             category=C.CREDENTIAL, score=0.51),
    ])
    notable = build_case_narrative(wh)["episodes"][0]["notable"]
    assert notable[0]["category"] == "CREDENTIAL", [n["category"] for n in notable]
