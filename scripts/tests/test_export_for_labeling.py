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
    dependency on the lab package's own test helpers."""
    from datetime import UTC, datetime

    from dd_core.casedir import CaseLayout
    from dd_core.enums import SourceType, ValueType
    from dd_core.schema import NormalizedArtifact
    from dd_lab.etl.warehouse import Warehouse

    layout = CaseLayout(tmp_path, "case1")
    layout.create_lab_skeleton()
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


def test_export_writes_one_csv_per_source_type(case, tmp_path):
    cases_root, case_id = case
    out_dir = tmp_path / "out"
    rc = efl.main(["--cases-root", str(cases_root), "--case-id", case_id, "--out-dir", str(out_dir)])
    assert rc == 0
    assert (out_dir / "cache.csv").is_file()
    assert (out_dir / "local_storage.csv").is_file()
    with (out_dir / "cache.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3  # i = 0, 2, 4
    row = rows[0]
    for col in ("artifact_id", "app", "source_type", "category", "key", "value", "text", "label"):
        assert col in row
    assert row["text"] == f"[cache] {row['key']} = {row['value']}"
    assert row["label"] == ""


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
