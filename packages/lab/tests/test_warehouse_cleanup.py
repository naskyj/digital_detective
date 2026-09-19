"""Warehouse cleans up orphaned _load_*.jsonl temp files from a previous
interrupted run (the app closed/restarted, or crashed, mid-load) - found on
real evidence: an interrupted memory-analysis run left hundreds of MB of
temp files sitting in a case's warehouse/ directory indefinitely.

A second, real failure this file also guards against: existence alone used
to be enough to call a _load_*.jsonl file orphaned - "any file found when a
Warehouse opens predates this instantiation" was true for a single-threaded
caller, but false the moment a SECOND Warehouse for the same case opens
while a FIRST one is still actively streaming into its own temp file (the
ingest job runs in a background thread - JobManager.submit - while the
server keeps handling other requests, so a concurrent request opening its
own Warehouse for the same case used to delete the active job's in-progress
file out from under it). Confirmed as the cause of a genuine real-case
failure: `read_json()` then errored "No files found that match the pattern
.../_load_<uuid>.jsonl" because the file it had just written was gone."""

from __future__ import annotations

import os
import time

from dd_lab.etl.warehouse import Warehouse, _clean_orphaned_load_files


def _backdate(path, seconds_ago: float) -> None:
    ts = time.time() - seconds_ago
    os.utime(path, (ts, ts))


def test_orphaned_load_file_removed_on_open(tmp_path):
    warehouse_dir = tmp_path / "warehouse"
    warehouse_dir.mkdir()
    orphan = warehouse_dir / "_load_deadbeefdeadbeefdeadbeefdeadbeef.jsonl"
    orphan.write_text('{"fake": "leftover from a crashed load"}\n')
    _backdate(orphan, seconds_ago=400)  # older than the default 300s min age

    Warehouse(warehouse_dir / "case.duckdb").close()

    assert not orphan.exists()


def test_a_freshly_written_load_file_is_left_alone(tmp_path):
    """The actual regression test for the real bug: a file written moments
    ago (indistinguishable, by existence alone, from a genuine orphan) must
    survive a second Warehouse being opened for the same case - it might be
    a concurrent, still-running load's own temp file, not an orphan at all."""
    warehouse_dir = tmp_path / "warehouse"
    warehouse_dir.mkdir()
    active = warehouse_dir / "_load_cafecafecafecafecafecafecafecafe.jsonl"
    active.write_text('{"still": "being written"}\n')  # mtime is "now"

    Warehouse(warehouse_dir / "case.duckdb").close()

    assert active.exists()


def test_clean_orphaned_load_files_respects_a_custom_min_age(tmp_path):
    """Direct unit coverage of the threshold boundary itself, independent of
    Warehouse's own default (300s) - a file just under the given min_age is
    kept, a file at/over it is removed."""
    warehouse_dir = tmp_path / "warehouse"
    warehouse_dir.mkdir()
    young = warehouse_dir / "_load_1111111111111111111111111111.jsonl"
    old = warehouse_dir / "_load_2222222222222222222222222222.jsonl"
    young.write_text("{}\n")
    old.write_text("{}\n")
    _backdate(young, seconds_ago=1)
    _backdate(old, seconds_ago=10)

    _clean_orphaned_load_files(warehouse_dir, min_age_seconds=5)

    assert young.exists()
    assert not old.exists()


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
