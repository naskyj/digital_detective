"""The report's own structure: numbering, the narrative section, and the
provenance columns.

Separate from test_report_seal.py (which is about tamper-evidence) and from
test_eval_regression.py (which is about the analysis chain end to end). This
file is specifically about whether the document a reader opens is correct as
a document.
"""

from __future__ import annotations

import html
import re
from datetime import UTC, datetime, timedelta

import pytest

from dd_core.casedir import CaseLayout
from dd_core.enums import ArtifactCategory as C
from dd_core.enums import SourceType as S
from dd_core.manifest import AcquisitionHost, Manifest
from dd_core.schema import ClassificationResult, NormalizedArtifact
from dd_core.signing import generate_keypair
from dd_lab.case import CaseMeta
from dd_lab.etl.warehouse import Warehouse
from dd_lab.report import build_report

_NOW = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)


def _seed_manifest(layout: CaseLayout, case_id: str) -> None:
    layout.create_evidence_skeleton()
    priv, pub = generate_keypair()
    Manifest(
        case_id=case_id, collector_version="test-0.0", operator="test",
        host=AcquisitionHost(hostname="h", os="t", os_version="0", username="u", timezone="UTC"),
        started_ts=_NOW, completed_ts=_NOW, target_apps=[], entries=[],
        public_key_pem=pub.decode(),
    ).write(layout.dir, priv)


def _art(app, key, value, *, minutes=0, ts=True, category=C.CHAT_CONTENT,
         source_type=S.CACHE, deleted=False, case_id="case"):
    return NormalizedArtifact(
        artifact_id=f"{app}:{key}:{minutes}:{abs(hash(value)) % 10**8}",
        case_id=case_id, app=app, source_type=source_type,
        source_file=f"{app}/Cache/f_{minutes}", source_locator=f"offset:{minutes}",
        key=key, value=value,
        record_ts=_NOW + timedelta(minutes=minutes) if ts else None,
        extracted_ts=_NOW, deleted=deleted,
        classification=ClassificationResult(
            category=category, confidence=0.9, classifier_version="test",
            relevant=True, relevance_score=0.9,
        ),
    )


@pytest.fixture
def case(tmp_path):
    """A case with no keywords - the configuration that exposed the
    section-numbering bug."""
    case_id = "case"
    root = tmp_path / "cases"
    layout = CaseLayout(root, case_id)
    layout.dir.mkdir(parents=True)
    _seed_manifest(layout, case_id)
    CaseMeta(case_id=case_id, name="Structure Case", keywords=[]).save(layout)

    wh = Warehouse(layout.warehouse)
    wh.load_artifacts([
        _art("discord", "cache.url", "https://discord.gg/PAIREDXYZ", minutes=0),
        _art("Slack", "cache.body", "join https://discord.gg/PAIREDXYZ", minutes=5),
        _art("discord", "idb.messages.content", "see you at 5", minutes=7),
        # a future cookie expiry - must not appear as an event
        _art("discord", "cookie.discordapp.com.__dcfduid.expires_utc",
             "2027-10-18T12:21:51", minutes=60 * 24 * 400, category=C.TIMESTAMP),
        # a state-only artifact - no timestamp at all
        _art("discord", "mem.string", "hunter2", ts=False, category=C.CREDENTIAL,
             source_type=S.MEMORY_STRING),
    ])
    wh.build_views()
    wh.close()
    return root, case_id


def _text(path) -> str:
    return html.unescape(re.sub("<.*?>", " ", path.read_text("utf-8")))


def test_section_numbers_are_contiguous(case):
    """They were hardcoded strings, so a case with no keywords skipped the
    keyword section and produced a report that jumped from 5 to 7."""
    root, case_id = case
    raw = build_report(root, case_id, actor="test").read_text("utf-8")

    numbers = [int(m) for m in re.findall(r"<h2>(\d+)\.", raw)]
    assert numbers == list(range(1, len(numbers) + 1)), numbers


def test_what_happened_section_is_present_and_early(case):
    root, case_id = case
    raw = build_report(root, case_id, actor="test").read_text("utf-8")
    heads = re.findall(r"<h2>(.*?)</h2>", raw)

    assert any("What Happened" in h for h in heads), heads
    # second, straight after provenance - a reader should learn what the
    # evidence shows before being handed rows of it
    assert "What Happened" in heads[1], heads


def test_narrative_states_its_own_coverage(case):
    """The case is 3 events and 1 state-only artifact. A narrative that
    implied it described everything would be an untrue statement."""
    root, case_id = case
    text = _text(build_report(root, case_id, actor="test"))

    assert "have no intrinsic time" in text
    assert "cookie expiry dates" in text
    assert "2027" not in text.split("Artifact Categories")[0]


def test_timeline_rows_carry_file_and_locator(case):
    """The timeline was the ONLY report section whose rows could not be
    traced back to evidence: v_timeline selects neither column."""
    root, case_id = case
    raw = build_report(root, case_id, actor="test").read_text("utf-8")
    timeline = raw.split("Evidence Timeline")[1]

    assert "Locator" in timeline
    assert "offset:0" in timeline
    assert "discord/Cache/f_0" in timeline


def test_correlated_entity_explains_itself(case):
    """A bare rank tells an investigator nothing about what they might have
    missed, so every entity carries the plain-English version of its score."""
    root, case_id = case
    text = _text(build_report(root, case_id, actor="test"))
    section = text.split("Cross-Source Correlations")[1]

    assert "significance" in section
    assert "different applications" in section
    assert "independently relevant" in section


def test_expiry_rows_never_appear_as_timeline_events(case):
    root, case_id = case
    raw = build_report(root, case_id, actor="test").read_text("utf-8")
    timeline = raw.split("Evidence Timeline")[1]

    assert "expires_utc" not in timeline
    assert "2027" not in timeline
