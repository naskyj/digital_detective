"""dd_core.evidence — copy resilience against a hostile/uncooperative source."""

from __future__ import annotations

import os
import sys

import pytest

from dd_core.evidence import _MAX_COPY_DEPTH, CopySpec, execute_copy


def test_symlinked_dir_is_skipped_not_followed(tmp_path):
    """A symlink must never be followed — it could point outside the profile
    (another user's files, a device node) or loop back on itself."""
    if sys.platform == "win32":
        pytest.skip("symlinks need elevated privileges on Windows")

    src = tmp_path / "leveldb"
    src.mkdir()
    (src / "000005.ldb").write_bytes(b"committed")
    loop = src / "loop"
    loop.symlink_to(src)  # a symlink back to its own parent - would recurse forever

    report = execute_copy(
        [CopySpec(source=src, dest_rel="local_storage/leveldb", is_dir=True)],
        tmp_path / "out",
    )
    assert any("000005" in str(p) for p in report.written)
    assert any("symlink" in reason for _, reason in report.skipped)
    # the symlink itself must not appear as a copied file/dir
    assert not (tmp_path / "out" / "local_storage" / "leveldb" / "loop" / "000005.ldb").exists()


def test_symlinked_file_is_skipped_not_followed(tmp_path):
    if sys.platform == "win32":
        pytest.skip("symlinks need elevated privileges on Windows")

    real = tmp_path / "real_prefs.json"
    real.write_text("{}")
    link = tmp_path / "Preferences"
    link.symlink_to(real)

    report = execute_copy([CopySpec(source=link, dest_rel="Preferences", is_dir=False)], tmp_path / "out")
    assert not report.written
    assert report.skipped and "symlink" in report.skipped[0][1]
    assert not (tmp_path / "out" / "Preferences").exists()


def test_excessively_deep_tree_is_bounded(tmp_path):
    """A pathologically deep directory tree is cut off rather than recursed
    into indefinitely, independent of the symlink guard."""
    src = tmp_path / "deep"
    cur = src
    cur.mkdir()
    for i in range(_MAX_COPY_DEPTH + 5):
        cur = cur / f"d{i}"
        cur.mkdir()
    (cur / "leaf.txt").write_bytes(b"x")

    report = execute_copy([CopySpec(source=src, dest_rel="deep", is_dir=True)], tmp_path / "out")
    assert any("max copy depth" in reason for _, reason in report.skipped)
    assert not any("leaf.txt" in str(p) for p in report.written)


def test_locked_file_still_copies_the_rest(tmp_path):
    src = tmp_path / "leveldb"
    src.mkdir()
    (src / "000005.ldb").write_bytes(b"committed")
    locked = src / "000006.log"
    locked.write_bytes(b"wal")
    os.chmod(locked, 0)
    try:
        report = execute_copy(
            [CopySpec(source=src, dest_rel="local_storage/leveldb", is_dir=True)],
            tmp_path / "out",
        )
        assert any("000005" in p.name for p in report.written)
        assert report.skipped and "000006.log" in report.skipped[0][0]
    finally:
        os.chmod(locked, 0o644)
