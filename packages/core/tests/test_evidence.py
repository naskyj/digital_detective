"""dd_core.evidence — copy resilience against a hostile/uncooperative source."""

from __future__ import annotations

import os
import sys

import pytest

from dd_core.evidence import (
    _MAX_COPY_DEPTH,
    CopySpec,
    execute_copy,
    find_webview2_profiles,
    iter_profiles_for_entry,
)


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


def _make_profile(d):
    d.mkdir(parents=True)
    (d / "Local State").write_text("{}")


def test_find_webview2_profiles_finds_a_nested_ebwebview_default(tmp_path):
    """The real case this exists for: a native (non-Electron) app that
    embeds a Microsoft Edge WebView2 control keeps ITS OWN Chromium
    profile at <somewhere>/EBWebView/<profile-name>/..., often nested much
    deeper than an Electron app's would be."""
    app_root = tmp_path / "SomeApp"
    _make_profile(app_root / "UniversalApp" / "WebView2" / "EBWebView" / "Default")

    found = find_webview2_profiles(app_root)
    assert len(found) == 1
    assert found[0] == app_root / "UniversalApp" / "WebView2" / "EBWebView" / "Default"


def test_find_webview2_profiles_finds_multiple_versioned_copies(tmp_path):
    """Confirmed on real evidence (Roblox): an app can keep more than one
    WebView2-embedded profile around, one per installed version."""
    app_root = tmp_path / "SomeApp"
    _make_profile(app_root / "Versions" / "v1" / "app.exe.WebView2" / "EBWebView" / "Default")
    _make_profile(app_root / "Versions" / "v2" / "app.exe.WebView2" / "EBWebView" / "Default")

    found = find_webview2_profiles(app_root)
    assert len(found) == 2


def test_find_webview2_profiles_ignores_an_ebwebview_dir_with_no_real_profile_inside(tmp_path):
    app_root = tmp_path / "SomeApp"
    empty = app_root / "WebView2" / "EBWebView" / "Default"
    empty.mkdir(parents=True)  # a profile-name dir with none of the Chromium marker files
    assert find_webview2_profiles(app_root) == []


def test_find_webview2_profiles_returns_empty_for_a_plain_non_electron_app(tmp_path):
    app_root = tmp_path / "PlainApp"
    app_root.mkdir()
    (app_root / "settings.ini").write_text("[x]\n")
    assert find_webview2_profiles(app_root) == []


def test_find_webview2_profiles_respects_max_depth(tmp_path):
    app_root = tmp_path / "DeepApp"
    nested = app_root
    for i in range(10):
        nested = nested / f"d{i}"
    _make_profile(nested / "EBWebView" / "Default")

    assert find_webview2_profiles(app_root, max_depth=3) == []
    assert find_webview2_profiles(app_root, max_depth=12) != []


def test_iter_profiles_for_entry_direct_match(tmp_path):
    app_root = tmp_path / "SomeApp"
    _make_profile(app_root)
    assert iter_profiles_for_entry(app_root) == [("SomeApp", app_root)]


def test_iter_profiles_for_entry_one_level_nested(tmp_path):
    app_root = tmp_path / "SomeApp"
    nested = app_root / "SomeApp"
    _make_profile(nested)
    assert iter_profiles_for_entry(app_root) == [("SomeApp", nested)]


def test_iter_profiles_for_entry_msix_redirect(tmp_path):
    """A Microsoft Store (MSIX/UWP) package folder, copied on its own
    (without the surrounding AppData/Local/Packages/ prefix a real image or
    live machine would have) - confirmed on real evidence: a Slack MSIX
    package folder named "<PackageFamilyName>" holds its real profile at
    "<that folder>/LocalCache/Roaming/Slack/...", invisible to a direct or
    one-level-nested check."""
    app_root = tmp_path / "SomeVendor.Slack_abc123"
    profile = app_root / "LocalCache" / "Roaming" / "Slack"
    _make_profile(profile)
    assert iter_profiles_for_entry(app_root) == [("Slack", profile)]


def test_iter_profiles_for_entry_msix_redirect_local_variant(tmp_path):
    app_root = tmp_path / "SomeVendor.App_abc123"
    profile = app_root / "LocalCache" / "Local" / "App"
    _make_profile(profile)
    assert iter_profiles_for_entry(app_root) == [("App", profile)]


def test_iter_profiles_for_entry_falls_through_to_webview2(tmp_path):
    """MSIX redirect and WebView2 are both checked, in that order, but only
    one should ever fire for a given entry - confirm the WebView2 path
    still works when there's no LocalCache directory at all."""
    app_root = tmp_path / "Roblox"
    profile = app_root / "UniversalApp" / "WebView2" / "EBWebView" / "Default"
    _make_profile(profile)
    found = iter_profiles_for_entry(app_root)
    assert len(found) == 1
    assert found[0][1] == profile


def test_iter_profiles_for_entry_no_match_returns_empty(tmp_path):
    app_root = tmp_path / "PlainApp"
    app_root.mkdir()
    assert iter_profiles_for_entry(app_root) == []
