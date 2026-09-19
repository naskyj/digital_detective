"""export_for_labeling.py and score_labels.py - built as standalone scripts,
loaded dynamically here (same pattern as test_fetch_collector_tools.py)
rather than as an importable package."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for pkg in ("core", "lab"):
    p = str(_REPO_ROOT / "packages" / pkg)
    if p not in sys.path:
        sys.path.insert(0, p)


def _load(name: str):
    script = _REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


efl = _load("export_for_labeling")
sl = _load("score_labels")


@pytest.fixture()
def case(tmp_path):
    """A tiny real (signed, verified) case with a handful of artifacts
    across two source types, built the same way the rest of the lab suite
    builds sample cases - but inlined here rather than depending on
    packages/lab/tests/sample_case.py, since scripts/ has no test
    dependency on the lab package's own test helpers.

    cache.csv now reads fresh from the evidence directory rather than the
    warehouse (see _load_cache_rows/iter_cache_entries_for_labeling) - an
    "app" directory under evidence is enough for main() to know which apps
    to scan; the tests below monkeypatch the actual cache reader itself, so
    no real Chromium cache needs to exist on disk here."""
    from datetime import UTC, datetime

    from dd_core.casedir import CaseLayout
    from dd_core.enums import SourceType, ValueType
    from dd_core.schema import NormalizedArtifact
    from dd_lab.etl.warehouse import Warehouse

    layout = CaseLayout(tmp_path, "case1")
    layout.create_lab_skeleton()
    (layout.evidence / "app").mkdir(parents=True, exist_ok=True)
    wh = Warehouse(layout.warehouse)
    try:
        rows = [
            NormalizedArtifact(
                artifact_id=f"a{i}", case_id="case1", app="app",
                source_type=SourceType.CACHE if i % 2 == 0 else SourceType.LOCAL_STORAGE,
                source_file="app/f", source_locator=f"loc{i}", key=f"k{i}", value=f"v{i}",
                value_type=ValueType.STRING, extracted_ts=datetime.now(UTC),
            )
            for i in range(6)
        ]
        wh.load_artifacts(rows)
    finally:
        wh.close()
    return tmp_path, "case1"


def _canned_cache_rows():
    """Two physical entries: one plain JSON-object body (one row), one
    JSON array-of-objects body (exploded into two rows sharing one
    entry_id) - the exact shape the real partner feedback this fixes is
    about: entry count, not metadata-field count."""
    from datetime import UTC, datetime

    from dd_lab.parsers.cache import CacheLabelRow

    return [
        CacheLabelRow(
            entry_id="e1", app="app", cache_kind="http_cache", cache_dir="app/Cache/Cache_Data",
            key="1/0/https://example.com/a.json",
            url="https://example.com/a.json", filename="a.json", http_status="HTTP/1.1 200",
            content_type="application/json", content_encoding=None, cache_control="no-cache",
            server=None, size_bytes=42,
            physical_file="data_1", offset=100,
            request_time=datetime(2026, 1, 1, tzinfo=UTC), response_time=datetime(2026, 1, 1, tzinfo=UTC),
            body_kind="json_object", array_index=None, value='{"id": 1}',
        ),
        CacheLabelRow(
            entry_id="e2", app="app", cache_kind="http_cache", cache_dir="app/Cache/Cache_Data",
            key="1/0/https://example.com/list.json",
            url="https://example.com/list.json", filename="list.json", http_status="HTTP/1.1 200",
            content_type="application/json", content_encoding=None, cache_control=None,
            server=None, size_bytes=20,
            physical_file="data_1", offset=200,
            request_time=None, response_time=None,
            body_kind="json_array_element", array_index=0, value='{"n": 1}',
        ),
        CacheLabelRow(
            entry_id="e2", app="app", cache_kind="http_cache", cache_dir="app/Cache/Cache_Data",
            key="1/0/https://example.com/list.json",
            url="https://example.com/list.json", filename="list.json", http_status="HTTP/1.1 200",
            content_type="application/json", content_encoding=None, cache_control=None,
            server=None, size_bytes=20,
            physical_file="data_1", offset=200,
            request_time=None, response_time=None,
            body_kind="json_array_element", array_index=1, value='{"n": 2}',
        ),
    ]


def test_export_writes_one_csv_per_source_type(case, tmp_path, monkeypatch):
    cases_root, case_id = case

    import dd_lab.parsers.cache as cache_mod

    canned = _canned_cache_rows()
    monkeypatch.setattr(cache_mod, "iter_cache_entries_for_labeling", lambda ctx: iter(canned))

    out_dir = tmp_path / "out"
    rc = efl.main(["--cases-root", str(cases_root), "--case-id", case_id, "--out-dir", str(out_dir)])
    assert rc == 0
    assert (out_dir / "cache.csv").is_file()
    assert (out_dir / "local_storage.csv").is_file()
    with (out_dir / "cache.csv").open() as f:
        rows = list(csv.DictReader(f))
    # one row for the whole-object entry (e1) + one row per array element of
    # the second entry (e2) = 3 rows for 2 physical entries - NOT one row per
    # metadata field (the real bug this fixes: 41,405 rows for 5,946 entries).
    assert len(rows) == 3
    assert {r["entry_id"] for r in rows} == {"e1", "e2"}
    for col in (
        "entry_id", "app", "source_type", "cache_dir", "physical_file", "offset", "url",
        "cache_control", "server", "deleted", "body_kind", "array_index", "value", "label",
    ):
        assert col in rows[0]
    assert all(r["source_type"] == "cache" for r in rows)
    assert all(r["label"] == "" for r in rows)
    # category/confidence/text were being assigned PER METADATA FIELD (e.g.
    # a bare content-type string classified CHAT_CONTENT) - meaningless once
    # a row is a whole entry, so dropped entirely rather than carried forward
    # incorrectly at the entry level.
    assert "category" not in rows[0]
    assert "confidence" not in rows[0]
    assert "text" not in rows[0]


def test_cache_csv_is_never_downsampled(case, tmp_path, monkeypatch):
    """cache has a different unit (one row per physical entry/array element)
    from every other source_type (one row per classified field) - stratifying
    it by --sample's category-based logic would be meaningless (cache rows
    carry no category) and would break the coverage parity a content-matching
    partner needs against the raw cache dump. --sample must leave it alone."""
    cases_root, case_id = case
    import dd_lab.parsers.cache as cache_mod

    canned = _canned_cache_rows()
    monkeypatch.setattr(cache_mod, "iter_cache_entries_for_labeling", lambda ctx: iter(canned))

    out_dir = tmp_path / "out"
    rc = efl.main([
        "--cases-root", str(cases_root), "--case-id", case_id, "--out-dir", str(out_dir), "--sample", "1",
    ])
    assert rc == 0
    with (out_dir / "cache.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3  # untouched by --sample 1, unlike local_storage.csv






def test_load_rows_marks_only_the_highest_sequence_number_as_latest(tmp_path):
    from datetime import UTC, datetime

    from dd_core.casedir import CaseLayout
    from dd_core.enums import SourceType, ValueType
    from dd_core.schema import NormalizedArtifact
    from dd_lab.etl.warehouse import Warehouse

    layout = CaseLayout(tmp_path, "case2")
    layout.create_lab_skeleton()
    wh = Warehouse(layout.warehouse)
    try:
        # Three historical writes of the SAME key (increasing seq, one with
        # a "#$" leaf-path suffix like a real flattened JSON value would
        # have) plus one unrelated key, mirroring real localstorage.py output.
        rows = [
            NormalizedArtifact(
                artifact_id="a1", case_id="case2", app="app", source_type=SourceType.LOCAL_STORAGE,
                source_file="app/f", source_locator="origin\x1fk\x1fseq=1#$", key="origin :: k",
                value="old", value_type=ValueType.STRING, extracted_ts=datetime.now(UTC),
            ),
            NormalizedArtifact(
                artifact_id="a2", case_id="case2", app="app", source_type=SourceType.LOCAL_STORAGE,
                source_file="app/f", source_locator="origin\x1fk\x1fseq=9#$", key="origin :: k",
                value="new", value_type=ValueType.STRING, extracted_ts=datetime.now(UTC),
            ),
            NormalizedArtifact(
                artifact_id="a3", case_id="case2", app="app", source_type=SourceType.LOCAL_STORAGE,
                source_file="app/f", source_locator="origin\x1fk\x1fseq=4#$", key="origin :: k",
                value="mid", value_type=ValueType.STRING, extracted_ts=datetime.now(UTC),
            ),
            NormalizedArtifact(
                artifact_id="a4", case_id="case2", app="app", source_type=SourceType.LOCAL_STORAGE,
                source_file="app/f", source_locator="origin\x1fother\x1fseq=1#$", key="origin :: other",
                value="v", value_type=ValueType.STRING, extracted_ts=datetime.now(UTC),
            ),
        ]
        wh.load_artifacts(rows)
        by_id = {r["artifact_id"]: r for r in efl._load_rows(wh, "local_storage")}
    finally:
        wh.close()
    assert by_id["a1"]["latest"] is False
    assert by_id["a2"]["latest"] is True  # highest seq (9) for "origin :: k"
    assert by_id["a3"]["latest"] is False
    assert by_id["a4"]["latest"] is True  # only version of "origin :: other"


def test_export_missing_warehouse_fails_cleanly(tmp_path):
    rc = efl.main([
        "--cases-root", str(tmp_path), "--case-id", "no-such-case", "--out-dir", str(tmp_path / "out"),
    ])
    assert rc == 1


def test_stratified_sample_respects_a_floor_per_category():
    import random

    rows = [{"category": "GENERAL", "n": i} for i in range(1000)] + [
        {"category": "CREDENTIAL", "n": i} for i in range(3)
    ]
    sample = efl._stratified_sample(rows, target=50, rng=random.Random(1))
    cats = {r["category"] for r in sample}
    assert "CREDENTIAL" in cats  # the rare category must not be sampled out entirely
    assert len(sample) <= 50


def test_score_labels_computes_accuracy_and_confusion(tmp_path, capsys):
    d = tmp_path / "labeled"
    d.mkdir()
    with (d / "cache.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=efl._FIELDS)
        w.writeheader()
        w.writerow({"artifact_id": "a1", "app": "app", "source_type": "cache", "category": "CHAT_CONTENT",
                    "confidence": "0.9", "key": "k1", "value": "v1", "text": "t1", "record_ts": "",
                    "source_file": "f", "label": "CHAT_CONTENT"})
        w.writerow({"artifact_id": "a2", "app": "app", "source_type": "cache", "category": "CHAT_CONTENT",
                    "confidence": "0.9", "key": "k2", "value": "v2", "text": "t2", "record_ts": "",
                    "source_file": "f", "label": "GENERAL"})
        w.writerow({"artifact_id": "a3", "app": "app", "source_type": "cache", "category": "PAYMENT_INFO",
                    "confidence": "0.5", "key": "k3", "value": "v3", "text": "t3", "record_ts": "",
                    "source_file": "f", "label": ""})  # not yet labelled - excluded

    rc = sl.main(["--dir", str(d)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "2 labelled row(s) scored" in out
    assert "50.0%" in out  # 1 of 2 correct


def test_score_labels_no_labelled_rows_fails_cleanly(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    with (d / "cache.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=efl._FIELDS)
        w.writeheader()
    rc = sl.main(["--dir", str(d)])
    assert rc == 1
