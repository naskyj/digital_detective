"""dd_core.evidence — copy resilience against a hostile/uncooperative source."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from dd_core.evidence import (
    _MAX_COPY_DEPTH,
    CopySpec,
    execute_copy,
    find_webview2_profiles,
    iter_profiles_for_entry,
    plan_native_root_copy,
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
    # a normal Electron app: entry IS the profile, so native_root is None -
    # plan_profile_copy()'s whole-profile copy already gets everything.
    app_root = tmp_path / "SomeApp"
    _make_profile(app_root)
    assert iter_profiles_for_entry(app_root) == [("SomeApp", app_root, None)]


def test_iter_profiles_for_entry_one_level_nested(tmp_path):
    app_root = tmp_path / "SomeApp"
    nested = app_root / "SomeApp"
    _make_profile(nested)
    assert iter_profiles_for_entry(app_root) == [("SomeApp", nested, None)]


def test_iter_profiles_for_entry_msix_redirect(tmp_path):
    """A Microsoft Store (MSIX/UWP) package folder, copied on its own
    (without the surrounding AppData/Local/Packages/ prefix a real image or
    live machine would have) - confirmed on real evidence: a Slack MSIX
    package folder named "<PackageFamilyName>" holds its real profile at
    "<that folder>/LocalCache/Roaming/Slack/...", invisible to a direct or
    one-level-nested check. native_root is the outer package folder - real
    native app data can sit there too, well outside the Chromium profile."""
    app_root = tmp_path / "SomeVendor.Slack_abc123"
    profile = app_root / "LocalCache" / "Roaming" / "Slack"
    _make_profile(profile)
    assert iter_profiles_for_entry(app_root) == [("Slack", profile, app_root)]


def test_iter_profiles_for_entry_msix_redirect_local_variant(tmp_path):
    app_root = tmp_path / "SomeVendor.App_abc123"
    profile = app_root / "LocalCache" / "Local" / "App"
    _make_profile(profile)
    assert iter_profiles_for_entry(app_root) == [("App", profile, app_root)]


def test_iter_profiles_for_entry_falls_through_to_webview2(tmp_path):
    """MSIX redirect and WebView2 are both checked, in that order, but only
    one should ever fire for a given entry - confirm the WebView2 path
    still works when there's no LocalCache directory at all. native_root is
    the app's own root here too - e.g. Roblox's own logs/XML config, sitting
    well outside the WebView2-embedded Chromium profile found underneath it."""
    app_root = tmp_path / "Roblox"
    profile = app_root / "UniversalApp" / "WebView2" / "EBWebView" / "Default"
    _make_profile(profile)
    found = iter_profiles_for_entry(app_root)
    assert len(found) == 1
    assert found[0][1] == profile
    assert found[0][2] == app_root


def test_iter_profiles_for_entry_no_match_returns_empty(tmp_path):
    app_root = tmp_path / "PlainApp"
    app_root.mkdir()
    assert iter_profiles_for_entry(app_root) == []


def _rels(specs):
    return {s.dest_rel for s in specs}


def test_plan_native_root_copy_picks_up_allowlisted_extensions(tmp_path):
    root = tmp_path / "Roblox"
    (root / "logs" / "1.0.123").mkdir(parents=True)
    (root / "logs" / "1.0.123" / "20260101_Player_1_last.log").write_text("hello")
    (root / "AnalysticsSettings.xml").write_text("<root/>")
    (root / "LocalStorage").mkdir()
    (root / "LocalStorage" / "appStorage.json").write_text("{}")

    specs = plan_native_root_copy(root)
    rels = _rels(specs)
    assert "native/logs/1.0.123/20260101_Player_1_last.log" in rels
    assert "native/AnalysticsSettings.xml" in rels
    assert "native/LocalStorage/appStorage.json" in rels
    assert all(not s.is_dir for s in specs)  # file-by-file, never a directory copy


def test_plan_native_root_copy_ignores_non_allowlisted_extensions(tmp_path):
    root = tmp_path / "Roblox"
    root.mkdir()
    (root / "RobloxPlayerBeta.exe").write_bytes(b"MZ...")
    (root / "content.pak").write_bytes(b"binary")
    specs = plan_native_root_copy(root)
    assert specs == []


def test_plan_native_root_copy_skips_denylisted_dir_names(tmp_path):
    # "logs" itself is NOT denylisted (that's the whole point - it's the one
    # place Roblox keeps its own logs), but something like a Crashpad dump
    # dir sitting alongside it should still be skipped.
    root = tmp_path / "Roblox"
    (root / "Crashpad").mkdir(parents=True)
    (root / "Crashpad" / "settings.json").write_text("{}")
    (root / "logs").mkdir()
    (root / "logs" / "app.log").write_text("real log content")
    specs = plan_native_root_copy(root)
    rels = _rels(specs)
    assert "native/logs/app.log" in rels
    assert not any("Crashpad" in r for r in rels)


def test_plan_native_root_copy_skips_the_embedded_webview2_profile(tmp_path):
    # the WebView2 profile itself is already collected separately via
    # plan_profile_copy() on profile_dir - must not be picked up again here.
    root = tmp_path / "Roblox"
    wv = root / "UniversalApp" / "WebView2" / "EBWebView" / "Default"
    wv.mkdir(parents=True)
    (wv / "Preferences").write_text("{}")
    (wv / "some_config.json").write_text("{}")
    (root / "AnalysticsSettings.xml").write_text("<root/>")
    specs = plan_native_root_copy(root)
    rels = _rels(specs)
    assert "native/AnalysticsSettings.xml" in rels
    assert not any("EBWebView" in r for r in rels)


def test_plan_native_root_copy_respects_max_files(tmp_path):
    root = tmp_path / "Roblox"
    root.mkdir()
    for i in range(10):
        (root / f"f{i}.log").write_text("x")
    assert len(plan_native_root_copy(root, max_files=3)) == 3


def test_plan_native_root_copy_missing_root_returns_empty(tmp_path):
    assert plan_native_root_copy(tmp_path / "does-not-exist") == []


# ------------------------------------------------- Windows junction loops
#
# Found on a collaborator's real Windows evidence: the native-root walk
# produced `native/Application Data/Application Data/Application Data/
# Application Data/Application Data/Temp/xml_file.xml` - five levels of
# %APPDATA%'s legacy self-referential junction, stopped only by max_depth.
# `_copy_tree_resilient` had a symlink guard; neither directory WALKER did.


def test_native_root_walk_does_not_follow_a_self_referential_junction(tmp_path):
    if os.name == "nt":
        pytest.skip("symlinks need elevated privileges on Windows")
    root = tmp_path / "LocalState"
    root.mkdir()
    (root / "real.xml").write_text("<a/>")
    # %APPDATA%\Application Data -> %APPDATA%, the actual Windows shape.
    (root / "Application Data").symlink_to(root)

    specs = plan_native_root_copy(root)

    assert [s.dest_rel for s in specs] == ["native/real.xml"]
    assert not any("Application Data" in s.dest_rel for s in specs)


def test_the_file_budget_is_not_spent_on_junction_duplicates(tmp_path):
    """The damaging half. plan_native_root_copy stops dead at max_files, so
    a loop that re-copies the same files at five different paths can exhaust
    the budget and leave genuine files uncollected - silently, with no
    warning and no skipped-reason anywhere."""
    if os.name == "nt":
        pytest.skip("symlinks need elevated privileges on Windows")
    root = tmp_path / "LocalState"
    (root / "deep").mkdir(parents=True)
    for i in range(4):
        (root / f"loot{i}.xml").write_text("<a/>")
    (root / "Application Data").symlink_to(root)

    specs = plan_native_root_copy(root, max_files=4)

    # Every real file fits in the budget of 4 - which it cannot if the
    # junction is walked, since the same files reappear under it.
    assert sorted(s.dest_rel for s in specs) == [
        "native/loot0.xml", "native/loot1.xml", "native/loot2.xml", "native/loot3.xml",
    ]


def test_webview2_scan_does_not_follow_a_self_referential_junction(tmp_path):
    if os.name == "nt":
        pytest.skip("symlinks need elevated privileges on Windows")
    app_root = tmp_path / "App"
    prof = app_root / "EBWebView" / "Default"
    prof.mkdir(parents=True)
    (prof / "Preferences").write_text("{}")
    (prof / "Local State").write_text("{}")
    (app_root / "Application Data").symlink_to(app_root)

    found = find_webview2_profiles(app_root)

    # Terminates, and reports the one real profile exactly once - not once
    # per level of the loop.
    assert found == [prof]


def test_walk_terminates_on_a_loop_that_is_not_a_symlink(tmp_path, monkeypatch):
    """Second layer. A filesystem backend can model a reparse point as a
    plain directory - is_symlink() returns False and the symlink guard never
    fires. The resolved-identity set has to catch it."""
    root = tmp_path / "LocalState"
    root.mkdir()
    (root / "real.xml").write_text("<a/>")
    loop = root / "Application Data"
    loop.symlink_to(root)
    # Pretend nothing is a symlink, exactly as such a backend would report.
    monkeypatch.setattr(Path, "is_symlink", lambda self: False)

    specs = plan_native_root_copy(root)

    assert [s.dest_rel for s in specs] == ["native/real.xml"]
