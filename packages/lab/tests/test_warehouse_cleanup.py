"""Warehouse cleans up orphaned _load_*.jsonl temp files from a previous
interrupted run (the app closed/restarted, or crashed, mid-load) - found on
real evidence: an interrupted memory-analysis run left hundreds of MB of
temp files sitting in a case's warehouse/ directory indefinitely."""

from __future__ import annotations

from dd_lab.etl.warehouse import Warehouse


def test_orphaned_load_file_removed_on_open(tmp_path):
    warehouse_dir = tmp_path / "warehouse"
    warehouse_dir.mkdir()
    orphan = warehouse_dir / "_load_deadbeefdeadbeefdeadbeefdeadbeef.jsonl"
    orphan.write_text('{"fake": "leftover from a crashed load"}\n')

    Warehouse(warehouse_dir / "case.duckdb").close()

    assert not orphan.exists()


def test_unrelated_files_in_the_same_directory_are_untouched(tmp_path):
    warehouse_dir = tmp_path / "warehouse"
    warehouse_dir.mkdir()
    keep = warehouse_dir / "notes.txt"
    keep.write_text("not a load temp file")

    Warehouse(warehouse_dir / "case.duckdb").close()

    assert keep.exists()


def test_opening_with_no_orphans_does_not_error(tmp_path):
    wh = Warehouse(tmp_path / "warehouse" / "case.duckdb")
    wh.close()
