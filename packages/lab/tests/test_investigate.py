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


def test_extract_slack_token():
    # a real xoxc- client token, found in real evidence this session -
    # distinctive enough to scan unscoped, no app context needed.
    hits = extract_entities(
        "SomeApp", "local_storage.value",
        "session token: xoxc-12052162462546-12047601193813-abc123",
    )
    assert any(h.entity_type == "slack_token" for h in hits)
    assert any(h.normalized.startswith("xoxc-") for h in hits)


def test_extract_slack_team_and_user_id_with_slack_context():
    # real workspace/user ids found in real evidence this session (a Slack
    # workspace "TP Gaming Clan"), embedded in a compound key string as
    # commonly as in a value - both are scanned.
    hits = extract_entities(
        "Slack", "https://app.slack.com :: endpoints-v1-T0C1J4SDLG2",
        '{"user_id": "U0C1DHP5PPX"}',
    )
    assert any(h.entity_type == "slack_team_id" and h.normalized == "T0C1J4SDLG2" for h in hits)
    assert any(h.entity_type == "slack_user_id" and h.normalized == "U0C1DHP5PPX" for h in hits)


def test_slack_team_id_shape_needs_slack_app_context():
    # a "T" + alnum shape is generic enough that a hex/hash fragment could
    # share it by chance - must not be tagged outside Slack's own evidence.
    hits = extract_entities("SomeOtherApp", "some.key", "T0C1J4SDLG2")
    assert not any(h.entity_type in ("slack_team_id", "slack_user_id") for h in hits)


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


def test_correlation_keeps_a_non_relevant_matching_citation_but_flags_it(wh):
    """Citation.relevant/relevance_score exist specifically so a citation
    pulled in by _related_rows() (which deliberately does NOT filter by
    relevance - see correlate.py's own comment on why: recall over
    precision for forensic evidence) can still be told apart from a
    genuinely relevant one. This is the actual behavior, not just the
    absence of filtering - a plumbing/telemetry-style match sharing the
    same value as a real finding must still show up as a citation (kept),
    but with relevant=False (flagged), not silently equal-weighted."""
    asset_id = "5551112223"
    seed = _artifact(
        "Discord", S.CACHE, "discord/Cache/f_seed", "http-body",
        "cache.body.embed", f"rbxassetid://{asset_id} shared", C.CHAT_CONTENT,
    )
    # Same value, second source type (satisfies min_distinct_sources), but
    # explicitly marked not relevant - e.g. an infra/plumbing hit that just
    # happens to share the id, distinct from category (which alone would
    # default relevant=True here per ClassificationResult's own fallback).
    noise = NormalizedArtifact(
        artifact_id="Discord:memory:offset99:noise",
        case_id="c1",
        app="Discord",
        source_type=S.MEMORY_STRING,
        source_file="_system/memory/physmem.raw",
        source_locator="offset:99000",
        key="telemetry.beacon",
        value=f"beacon?id={asset_id}",
        record_ts=_NOW + timedelta(minutes=1),
        extracted_ts=_NOW,
        classification=ClassificationResult(
            category=C.NETWORK_DETAIL, confidence=0.4, classifier_version="test",
            relevant=False, relevance_score=0.0,
        ),
    )
    wh.load_artifacts([seed, noise])

    report = build_correlation_report(wh, min_distinct_sources=2)
    hit = next((e for e in report if asset_id in e.value), None)
    assert hit is not None, [e.value for e in report]

    by_id = {c.artifact_id: c for c in hit.artifacts}
    assert seed.artifact_id in by_id and noise.artifact_id in by_id  # both KEPT (recall preserved)
    assert by_id[seed.artifact_id].relevant is True
    assert by_id[noise.artifact_id].relevant is False  # but correctly FLAGGED, not equal-weighted
    assert by_id[noise.artifact_id].relevance_score == 0.0


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


# --------------------------------------------------------------- shared email pattern

def test_correlation_does_not_treat_image_filenames_as_emails():
    """`mask-84x84@2x.png` is syntactically a valid email. On a real case ALL
    of the top 100 cross-source "correlated identifiers" were bogus emails of
    this shape, which made the correlations view useless."""
    for v in (
        "mask-84x84@2x.png", "tray-unread@5x.png", "alert@2x.png", "icon@3x.svg",
        # Carved binary that merely has an email SHAPE. `.pn` is a genuine
        # ccTLD, so no blocklist rejects `edge@2x.pn` - only requiring a
        # plausible TLD does. All real examples from one case.
        "4@p.amp", "9@o.ad", "h@fb.vuw", "n@2.nc", "edge@2x.pn",
        "ap5@ip0akp4fpp3cop3dpp.alp",
    ):
        hits = [h for h in extract_entities("discord", "mem.string", v) if h.entity_type == "email"]
        assert not hits, f"{v!r} was extracted as an email: {[h.normalized for h in hits]}"


def test_classification_and_correlation_share_one_email_pattern():
    """These two subsystems each had their own copy. Fixing rules.py cleared
    24,519 of 24,663 bogus classifications on a real case while correlations
    stayed full of the identical false positives, because entities.py was
    never touched. Pin them to the same object so that cannot recur."""
    from dd_core.enums import ArtifactCategory as C
    from dd_core.patterns import EMAIL_RE, EMAIL_STRICT_RE
    from dd_lab.classify.rules import _VALUE_PATTERNS
    from dd_lab.investigate.entities import _EMAIL

    # Both come from the shared module - neither subsystem defines its own.
    assert _VALUE_PATTERNS[C.USER][0][0] is EMAIL_RE
    assert _EMAIL is EMAIL_STRICT_RE


# ------------------------------------------------- significance and connections
#
# Ranking used to be `-len(e.artifacts)`, and `_related_rows` caps that list -
# so every entity at or above the cap tied exactly and the sort degenerated to
# alphabetical by value. On the real case that put Discord's cached
# server-discovery names (growtopia, leagueoflegends, minecraft) above the
# genuine invite codes in the evidence.


def _ent(app, source_type, key, value, *, n=1, relevant=True, deleted=False, ts_offset=0):
    return [
        NormalizedArtifact(
            # value fingerprint included: artifact_id is the primary key, so
            # two different values at the same app/source/key would otherwise
            # overwrite each other and the fixture would silently lose rows.
            artifact_id=f"{app}:{source_type}:{key}:{abs(hash(value)) % 10**8}:{i}",
            case_id="c1", app=app, source_type=source_type,
            source_file=f"{app}/f{i}", source_locator=f"loc{i}",
            key=key, value=value,
            record_ts=_NOW + timedelta(minutes=ts_offset + i),
            extracted_ts=_NOW, deleted=deleted,
            classification=ClassificationResult(
                category=C.CHAT_CONTENT, confidence=0.9, classifier_version="test",
                relevant=relevant, relevance_score=0.9 if relevant else 0.0,
            ),
        )
        for i in range(n)
    ]


def test_ranking_is_not_citation_count(wh):
    """A widely-cached but uncorroborated value must rank BELOW a rarer one
    that spans two applications - the exact inversion seen on the real case."""
    noisy = "https://discord.gg/growtopia"
    real = "https://discord.gg/RxPTgtyY"
    arts = (
        # 30 occurrences, one app, almost none independently relevant
        _ent("discord", S.CACHE, "cache.body", noisy, n=1)
        + _ent("discord", S.MEMORY_STRING, "mem.string", noisy, n=29, relevant=False)
        # 4 occurrences, two apps, all relevant
        + _ent("discord", S.CACHE, "cache.body", real, n=2)
        + _ent("Slack", S.INDEXEDDB, "idb.messages.content", real, n=2)
    )
    wh.load_artifacts(arts)

    report = build_correlation_report(wh, max_artifacts_per_entity=50)
    order = [e.value for e in report]

    assert order.index("RxPTgtyY") < order.index("growtopia"), order
    by_value = {e.value: e for e in report}
    assert by_value["growtopia"].occurrence_count == 30
    assert by_value["RxPTgtyY"].occurrence_count == 4


def test_occurrence_count_is_the_true_total_not_the_citation_cap(wh):
    """The saturation bug itself: with a cap of 5, two entities at 20 and 40
    occurrences both yield 5 citations. If the count comes from len(artifacts)
    they are indistinguishable."""
    wh.load_artifacts(
        _ent("discord", S.CACHE, "cache.body", "https://discord.gg/AAAAAAAA", n=20)
        + _ent("discord", S.MEMORY_STRING, "mem.string", "https://discord.gg/AAAAAAAA", n=20)
        + _ent("discord", S.CACHE, "cache.body", "https://discord.gg/BBBBBBBB", n=5)
        + _ent("discord", S.MEMORY_STRING, "mem.string", "https://discord.gg/BBBBBBBB", n=5)
    )
    report = build_correlation_report(wh, max_artifacts_per_entity=5)
    by_value = {e.value: e for e in report}

    assert len(by_value["AAAAAAAA"].artifacts) == 5   # citations capped
    assert len(by_value["BBBBBBBB"].artifacts) == 5
    assert by_value["AAAAAAAA"].occurrence_count == 40  # true totals survive
    assert by_value["BBBBBBBB"].occurrence_count == 10


def test_cross_app_is_distinguished_from_cross_storage(wh):
    """`min_distinct_sources` is an OR, so "same id in discord/cache and
    discord/indexeddb" qualified identically to "same id in Discord AND
    Slack". One is a storage detail; the other links two applications."""
    wh.load_artifacts(
        _ent("discord", S.CACHE, "cache.body", "https://discord.gg/STORAGEON", n=1)
        + _ent("discord", S.INDEXEDDB, "idb.content", "https://discord.gg/STORAGEON", n=1)
        + _ent("discord", S.CACHE, "cache.body", "https://discord.gg/TWOAPPSXX", n=1)
        + _ent("Slack", S.CACHE, "cache.body", "https://discord.gg/TWOAPPSXX", n=1)
    )
    by_value = {e.value: e for e in build_correlation_report(wh)}

    assert by_value["STORAGEON"].cross_app is False
    assert by_value["TWOAPPSXX"].cross_app is True
    assert by_value["TWOAPPSXX"].significance > by_value["STORAGEON"].significance
    assert any("2 different applications" in r for r in by_value["TWOAPPSXX"].reasons())


def test_recovered_deleted_occurrences_are_surfaced_and_raise_significance(wh):
    """`deleted` has been in the schema throughout and was carried by no
    citation, timeline or report section - despite being the strongest single
    flag available: someone tried to remove it and it survived."""
    wh.load_artifacts(
        _ent("discord", S.CACHE, "cache.body", "https://discord.gg/DELETEDAA", n=1)
        + _ent("discord", S.INDEXEDDB, "idb.content", "https://discord.gg/DELETEDAA",
               n=1, deleted=True)
        + _ent("discord", S.CACHE, "cache.body", "https://discord.gg/ORDINARYB", n=1)
        + _ent("discord", S.INDEXEDDB, "idb.content", "https://discord.gg/ORDINARYB", n=1)
    )
    by_value = {e.value: e for e in build_correlation_report(wh)}

    assert by_value["DELETEDAA"].deleted_count == 1
    assert by_value["ORDINARYB"].deleted_count == 0
    assert by_value["DELETEDAA"].significance > by_value["ORDINARYB"].significance
    assert any("RECOVERED DELETED" in r for r in by_value["DELETEDAA"].reasons())
    assert any(c.deleted for c in by_value["DELETEDAA"].artifacts)


def test_memory_only_entity_is_flagged(wh):
    """Live in a process at acquisition but never persisted - a disk-only
    examination would have missed it entirely."""
    wh.load_artifacts(
        _ent("discord", S.MEMORY_STRING, "mem.string", "https://discord.gg/MEMONLYAA", n=1)
        + _ent("discord", S.MEMORY_PROCESS, "mem.proc", "https://discord.gg/MEMONLYAA", n=1)
    )
    e = build_correlation_report(wh)[0]

    assert e.memory_only is True
    assert any("never written to disk" in r for r in e.reasons())


def test_two_entities_in_one_artifact_are_linked(wh):
    """The "how are things connected" layer, which did not exist:
    extract_entities returns a list per row, so an artifact yielding both an
    email and a Discord id had that pairing discarded on the next iteration."""
    both = "contact kim@example.com or join https://discord.gg/PAIREDXYZ"
    wh.load_artifacts(
        _ent("discord", S.CACHE, "cache.body", both, n=1)
        + _ent("discord", S.MEMORY_STRING, "mem.string", both, n=1)
    )
    report = build_correlation_report(wh)
    by_type = {e.entity_type: e for e in report}

    assert "email" in by_type and "discord_invite" in by_type
    linked = by_type["email"].co_occurring
    assert [c.value for c in linked] == ["PAIREDXYZ"]
    assert linked[0].shared_artifacts == 2
    assert any("same artifact as" in r for r in by_type["email"].reasons())


def test_reasons_are_empty_for_nothing_and_never_invent(wh):
    """reasons() is the plain-English version of the score. It must describe
    only what was computed - an entity with one storage type, no deletions and
    no partners gets no claims made about it."""
    wh.load_artifacts(
        _ent("discord", S.CACHE, "cache.body", "https://discord.gg/PLAINAAAA", n=1)
        + _ent("Slack", S.CACHE, "cache.body", "https://discord.gg/PLAINAAAA", n=1)
    )
    e = build_correlation_report(wh)[0]
    joined = " ".join(e.reasons())

    assert "RECOVERED DELETED" not in joined
    assert "never written to disk" not in joined
    assert "same artifact as" not in joined
