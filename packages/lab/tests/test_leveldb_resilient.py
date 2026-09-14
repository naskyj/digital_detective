"""dd_lab.parsers.base.open_leveldb_resilient — the retry-without-.log logic.

Found via a real 48GB E01 test image: a live-acquired `.log` (write-ahead log)
commonly ends with zero-byte block padding that `ccl_chromium_reader`'s reader
treats as fatal, aborting the WHOLE store — not just the log — and silently
dropping everything already compacted into `.ldb`/`.sst` files too (9,848 real
Local Storage records recovered once this was fixed, previously 0). These
tests exercise the retry/copy logic itself with a fake `opener`, since
reproducing the exact corrupt-.log byte pattern isn't worth the trouble when
the logic under test is ours, not ccl_chromium_reader's parsing.
"""

from __future__ import annotations

from pathlib import Path

from dd_lab.parsers.base import open_leveldb_resilient


def test_succeeds_first_try_no_copy_needed(tmp_path):
    calls = []

    def opener(d):
        calls.append(d)
        return "db-handle"

    obj, cleanup = open_leveldb_resilient(tmp_path, opener)

    assert obj == "db-handle"
    assert cleanup is None
    assert calls == [tmp_path]  # opened directly, no retry copy made


def test_retries_without_log_file_on_failure(tmp_path):
    (tmp_path / "000029.log").write_bytes(b"corrupt-wal-data")
    (tmp_path / "000027.ldb").write_bytes(b"compacted-sst-data")
    (tmp_path / "MANIFEST-000001").write_bytes(b"manifest")

    attempts: list[Path] = []

    def opener(d: Path):
        attempts.append(d)
        if (d / "000029.log").exists():
            raise ValueError("simulated: corrupt .log block type")
        return f"opened:{d}"

    obj, cleanup = open_leveldb_resilient(tmp_path, opener)

    assert obj is not None and obj.startswith("opened:")
    assert len(attempts) == 2
    assert attempts[0] == tmp_path  # first try: the real dir, with .log
    assert cleanup is not None and cleanup != tmp_path  # second try: a filtered copy
    assert not (cleanup / "000029.log").exists()  # .log excluded
    assert (cleanup / "000027.ldb").read_bytes() == b"compacted-sst-data"  # .ldb kept
    assert (cleanup / "MANIFEST-000001").is_file()  # manifest kept


def test_gives_up_cleanly_when_still_failing_without_log(tmp_path):
    (tmp_path / "000029.log").write_bytes(b"corrupt")
    (tmp_path / "000027.ldb").write_bytes(b"also-corrupt-somehow")

    def opener(d):
        raise ValueError("still broken")

    obj, cleanup = open_leveldb_resilient(tmp_path, opener)

    assert obj is None
    assert cleanup is None


def test_nothing_left_to_open_without_log_files(tmp_path):
    """A dir containing ONLY a .log file - after filtering there's nothing to
    retry with, so it gives up without a pointless second attempt."""
    (tmp_path / "000029.log").write_bytes(b"corrupt")
    calls = 0

    def opener(d):
        nonlocal calls
        calls += 1
        raise ValueError("broken")

    obj, cleanup = open_leveldb_resilient(tmp_path, opener)

    assert obj is None
    assert cleanup is None
    assert calls == 1  # only the first (failing) attempt against the real dir
