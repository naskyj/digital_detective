"""dd_lab.case - CaseMeta/create_case/list_cases, in particular that a case
can now exist with metadata and no evidence yet (case creation used to
require a signed manifest to already exist - see create_case's docstring)."""

from __future__ import annotations

from dd_core.casedir import CaseLayout
from dd_lab.case import CaseMeta, CaseStatus, create_case, list_cases, new_case_id


def test_create_case_needs_no_evidence(tmp_path):
    meta = create_case(tmp_path, "case1", "My Case", "a description", ["gift card"])
    assert meta.case_id == "case1"
    assert meta.status == CaseStatus.NEW
    assert meta.sources == []
    layout = CaseLayout(tmp_path, "case1")
    assert layout.case_meta.is_file()
    assert not layout.manifest.is_file()  # still no evidence - that's fine


def test_create_case_registers_pre_existing_dd_collect_evidence(tmp_path):
    """The OTHER supported flow, still needed: dd-collect (the live
    collector) already produced a fully-signed case elsewhere - this just
    attaches a name/keywords to it, and must never touch its manifest. A
    real regression caught while building the unified New Case page: it
    initially had no way to name an EXISTING case_id at all, only to
    generate a fresh one for acquiring new evidence."""
    layout = CaseLayout(tmp_path, "collector_case_123")
    layout.dir.mkdir(parents=True)
    layout.manifest.write_text('{"fake": "already-signed manifest"}')

    meta = create_case(tmp_path, "collector_case_123", "Registered dd-collect case", "", ["token"])
    assert meta.case_id == "collector_case_123"
    assert layout.manifest.read_text() == '{"fake": "already-signed manifest"}'  # untouched


def test_create_case_generates_an_id_when_none_given(tmp_path):
    meta = create_case(tmp_path, None, "Auto ID case")
    assert meta.case_id  # non-empty, auto-generated
    assert CaseLayout(tmp_path, meta.case_id).case_meta.is_file()


def test_new_case_id_is_unique_and_sortable():
    a, b = new_case_id(), new_case_id()
    assert a != b
    assert a[:8].isdigit()  # YYYYMMDD prefix - sorts chronologically


def test_list_cases_shows_an_evidence_free_case(tmp_path):
    create_case(tmp_path, "case1", "My Case")
    cases = list_cases(tmp_path)
    assert len(cases) == 1
    assert cases[0].case_id == "case1"
    assert cases[0].status == CaseStatus.NEW


def test_case_meta_round_trips_sources(tmp_path):
    from dd_lab.case import EvidenceSourceMeta

    layout = CaseLayout(tmp_path, "case1")
    meta = CaseMeta(case_id="case1", name="My Case", sources=[
        EvidenceSourceMeta(label="Suspect Laptop", kind="image", path="/x.E01", apps=["discord"]),
    ])
    meta.save(layout)
    loaded = CaseMeta.load(layout)
    assert len(loaded.sources) == 1
    assert loaded.sources[0].label == "Suspect Laptop"
    assert loaded.sources[0].kind == "image"
    assert loaded.sources[0].apps == ["discord"]
