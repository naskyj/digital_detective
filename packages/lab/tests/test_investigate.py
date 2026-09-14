"""Phase 2 — cross-source correlation, entity extraction, self-verification.

Uses a synthetic warehouse (hand-built NormalizedArtifact rows) so the exact
scenario from the design discussion is reproducible without any real evidence:
the same Roblox asset id shows up in a Discord Cache JSON blob, a raw
memory-string carve, and an IndexedDB message row.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from dd_core.enums import ArtifactCategory as C
from dd_core.enums import SourceType as S
from dd_core.schema import ClassificationResult, NormalizedArtifact
from dd_lab.etl.warehouse import Warehouse
from dd_lab.investigate import build_correlation_report, extract_entities, verify_claim

_NOW = datetime(2026, 9, 1, tzinfo=UTC)


def _artifact(
    app, source_type, source_file, locator, key, value, category, *, ts_offset_min=0, case_id="c1"
):
    return NormalizedArtifact(
        artifact_id=f"{app}:{source_file}:{locator}:{key}",
        case_id=case_id,
        app=app,
        source_type=source_type,
        source_file=source_file,
        source_locator=locator,
        key=key,
        value=value,
        record_ts=_NOW + timedelta(minutes=ts_offset_min),
        extracted_ts=_NOW,
        classification=ClassificationResult(category=category, confidence=0.9, classifier_version="test"),
    )


@pytest.fixture
def wh(tmp_path):
    w = Warehouse(tmp_path / "case.duckdb")
    yield w
    w.close()


# --------------------------------------------------------------- extract_entities


def test_extract_email():
    hits = extract_entities("Slack", "user.email", "reach me at kim@example.com please")
    assert any(h.entity_type == "email" and h.normalized == "kim@example.com" for h in hits)


def test_extract_discord_invite():
    hits = extract_entities("Discord", "cache.body.invite_url", "join us: discord.gg/abc123XY")
    assert any(h.entity_type == "discord_invite" and h.normalized == "abc123XY" for h in hits)


def test_extract_discord_mention():
    # found via real evidence: a captured referenced_message with no URL/key
    # context at all, just Discord's own "<@id>" mention syntax in the text.
    hits = extract_entities(
        "Discord", "cache.body.referenced_message.content",
        "Also, <@1473705902501531761>\nAccount sharing is against ToS.",
    )
    assert any(
        h.entity_type == "discord_user_id" and h.normalized == "1473705902501531761" for h in hits
    )


def test_discord_role_mention_is_not_a_user():
    hits = extract_entities("Discord", "cache.body.content", "ping <@&1473705902501531761> everyone")
    assert not any(h.entity_type == "discord_user_id" for h in hits)


def test_extract_roblox_asset_id():
    hits = extract_entities("Discord", "cache.body.embed.thumbnail", "rbxassetid://9876543210")
    assert any(h.entity_type == "roblox_asset_id" and h.normalized == "9876543210" for h in hits)


def test_bare_snowflake_needs_context():
    # a long digit string with no id-ish key and no discord/roblox app context
    # must NOT be tagged - avoids flooding correlation with plain numbers.
    hits = extract_entities("SomeApp", "counter.value", "123456789012345678")
    assert hits == []


def test_bare_snowflake_tagged_with_discord_context():
    hits = extract_entities("Discord", "message.authorId", "123456789012345678")
    assert any(
        h.entity_type == "discord_user_id" and h.normalized == "123456789012345678" for h in hits
    )


# --------------------------------------------------------------- build_correlation_report


def test_correlates_asset_id_across_app_cache_memory_and_leveldb(wh):
    """The scenario from the design discussion: the same Roblox asset id
    appears in a Discord Cache JSON blob, a memory-string carve, and an
    IndexedDB message row - three different apps/source types."""
    asset_id = "9876543210"
    artifacts = [
        _artifact(
            "Discord", S.CACHE, "discord/Cache/f_000123", "http-body",
            "cache.body.embeds[0].thumbnail.url", f"https://t3.rbxcdn.com/rbxassetid/{asset_id}",
            C.CHAT_CONTENT, ts_offset_min=0,
        ),
        _artifact(
            "_system", S.MEMORY_STRING, "_system/memory/physmem.raw", "offset:48213",
            "memory.string", f"rbxassetid://{asset_id} shared in dm", C.CHAT_CONTENT, ts_offset_min=5,
        ),
        _artifact(
            "Discord", S.INDEXEDDB, "discord/IndexedDB/messages.leveldb", "store/messages/998",
            "content", f"check this out rbxassetid://{asset_id}", C.CHAT_CONTENT, ts_offset_min=10,
        ),
        # unrelated noise row - must not show up as correlated (appears once, one app)
        _artifact(
            "Discord", S.CACHE, "discord/Cache/f_000456", "http-body",
            "cache.body.unrelated", "nothing interesting here", C.GENERAL,
        ),
    ]
    wh.load_artifacts(artifacts)

    report = build_correlation_report(wh, min_distinct_sources=2)

    hit = next((e for e in report if asset_id in e.value), None)
    assert hit is not None, [e.value for e in report]
    assert hit.entity_type == "roblox_asset_id"
    assert set(hit.source_types) >= {"cache", "memory_string", "indexeddb"}
    assert len(hit.artifacts) >= 3

    # timeline() must return them oldest-first with full citations
    ordered = hit.timeline()
    assert [c.record_ts for c in ordered] == sorted(c.record_ts for c in ordered)
    assert all(c.artifact_id and c.source_file and c.source_locator for c in ordered)


def test_single_occurrence_is_not_correlated(wh):
    wh.load_artifacts([
        _artifact(
            "Discord", S.CACHE, "discord/Cache/f_1", "http-body",
            "cache.body.embed", "rbxassetid://1112223334", C.CHAT_CONTENT,
        ),
    ])
    report = build_correlation_report(wh, min_distinct_sources=2)
    assert report == []


# --------------------------------------------------------------- verify_claim


def test_verify_claim_confirms_real_artifact(wh):
    a = _artifact(
        "Discord", S.INDEXEDDB, "discord/IndexedDB/x", "loc1", "content", "hello world",
        C.CHAT_CONTENT,
    )
    wh.load_artifacts([a])

    v = verify_claim(wh, a.artifact_id, "hello world")
    assert v.ok
    assert v.source_file == "discord/IndexedDB/x"


def test_verify_claim_rejects_misquote(wh):
    a = _artifact(
        "Discord", S.INDEXEDDB, "discord/IndexedDB/x", "loc1", "content", "hello world",
        C.CHAT_CONTENT,
    )
    wh.load_artifacts([a])

    v = verify_claim(wh, a.artifact_id, "this text is not in the artifact")
    assert not v.ok
    assert "not found" in v.reason


def test_verify_claim_rejects_unknown_artifact_id(wh):
    v = verify_claim(wh, "does-not-exist")
    assert not v.ok
    assert "not found in warehouse" in v.reason
