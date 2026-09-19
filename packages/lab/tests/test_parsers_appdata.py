"""dd_lab.parsers.appdata - the generic parser for native (non-Chromium)
config/log/JSON files no named parser claims: a WebView2/MSIX-hosted app's
own native root (collected under evidence/<app>/native/ - see
dd_core.evidence.plan_native_root_copy) and any unclaimed file loose in a
normal Electron app's own profile root (e.g. Discord's own settings.json,
distinct from Chromium's Preferences)."""

from __future__ import annotations

from pathlib import Path

from dd_core.enums import SourceType
from dd_lab.parsers import appdata
from dd_lab.parsers.base import ParseContext


def _ctx(tmp_path: Path, app: str = "discord") -> ParseContext:
    return ParseContext(case_id="c1", app=app, evidence_root=tmp_path / "evidence")


def test_parses_a_loose_json_file_not_claimed_by_any_named_parser(tmp_path):
    # Discord's own settings.json, sitting right next to Chromium's own
    # Preferences file - already copied today (whole-profile copy), never
    # decoded by any named parser.
    ctx = _ctx(tmp_path)
    ctx.app_root.mkdir(parents=True)
    (ctx.app_root / "settings.json").write_text('{"locale": "en-US", "SKIP_HOST_UPDATE": true}')

    artifacts = list(appdata.parse(ctx))
    assert artifacts
    assert all(a.source_type == SourceType.APP_DATA for a in artifacts)
    # a JSON *object* (not array) is always one artifact whole, same
    # established policy every other parser here follows (jsonflat.py) -
    # not split into one row per field.
    assert artifacts[0].key == "appdata.settings.json"
    assert "en-US" in artifacts[0].value


def test_parses_native_json_xml_and_log_files(tmp_path):
    ctx = _ctx(tmp_path, app="Roblox__WebView2_via_UniversalApp_WebView2")
    native = ctx.app_root / "native"
    (native / "LocalStorage").mkdir(parents=True)
    (native / "LocalStorage" / "appStorage.json").write_text('{"userId": "12345", "lastSession": "abc"}')
    (native / "AnalysticsSettings.xml").write_text(
        '<Settings><Metric name="fps"><Value>60</Value></Metric></Settings>'
    )
    (native / "logs").mkdir()
    (native / "logs" / "20260101_Player_1_last.log").write_text(
        "hi\n"  # below MIN_LINE_LEN (4) - must be dropped
        "player connected to game server region us-east\n"
    )

    artifacts = list(appdata.parse(ctx))

    json_rows = [a for a in artifacts if "appStorage.json" in a.source_file]
    assert json_rows and "12345" in json_rows[0].value  # object kept whole, per jsonflat.py's policy

    xml_rows = [a for a in artifacts if "AnalysticsSettings.xml" in a.source_file]
    assert xml_rows  # the XML tree walk produced at least one leaf artifact
    assert any("60" in a.value for a in xml_rows)

    log_rows = [a for a in artifacts if "_last.log" in a.source_file]
    assert any("player connected" in a.value for a in log_rows)
    assert not any(a.value == "hi" for a in log_rows)  # below MIN_LINE_LEN, dropped


def test_skips_files_already_claimed_by_a_named_parser(tmp_path):
    """The one real collision risk: LevelDB's own write-ahead-log files use
    the exact same ".log" extension this parser otherwise treats as a plain
    text log - must never try to decode Local Storage/leveldb/000001.log as
    application log lines."""
    ctx = _ctx(tmp_path)
    ctx.app_root.mkdir(parents=True)
    (ctx.app_root / "Preferences").write_text('{"already": "handled by prefs.py"}')
    ldb = ctx.app_root / "Local Storage" / "leveldb"
    ldb.mkdir(parents=True)
    (ldb / "000001.log").write_bytes(b"\x00\x01binary leveldb wal content, not text\x02\x03")

    artifacts = list(appdata.parse(ctx))
    assert not any("Preferences" in a.source_file for a in artifacts)
    assert not any("leveldb" in a.source_file for a in artifacts)


def test_ini_style_key_value_lines(tmp_path):
    ctx = _ctx(tmp_path, app="SomeApp")
    native = ctx.app_root / "native"
    native.mkdir(parents=True)
    (native / "app.cfg").write_text("; a comment\nRegion=us-east\nFullscreen=true\n")

    artifacts = list(appdata.parse(ctx))
    values = {a.key: a.value for a in artifacts}
    assert values.get("appdata.native/app.cfg.Region") == "us-east"
    assert values.get("appdata.native/app.cfg.Fullscreen") == "true"


def test_ignores_non_allowlisted_extensions(tmp_path):
    ctx = _ctx(tmp_path, app="Roblox")
    native = ctx.app_root / "native"
    native.mkdir(parents=True)
    (native / "RobloxPlayerBeta.exe").write_bytes(b"MZ...")
    assert list(appdata.parse(ctx)) == []


def test_missing_app_root_yields_nothing(tmp_path):
    ctx = _ctx(tmp_path, app="does-not-exist")
    assert list(appdata.parse(ctx)) == []


def test_appdata_coexists_with_named_parsers_via_parse_app(tmp_path):
    """End-to-end through the real dispatch (dd_lab.parsers.parse_app), not
    appdata.parse() in isolation - confirms it's actually wired into
    PARSERS, runs after the named parsers without erroring, and that a
    Preferences file (claimed by prefs.py) doesn't also get double-parsed
    here even though it's not one of the LevelDB/Cache paths."""
    from dd_lab.parsers import parse_app

    evidence_root = tmp_path / "evidence"
    app_root = evidence_root / "discord"
    app_root.mkdir(parents=True)
    (app_root / "Preferences").write_text('{"real": "chromium prefs"}')
    (app_root / "settings.json").write_text('{"discord_own_setting": true}')

    artifacts = list(parse_app("case1", evidence_root, "discord"))
    source_types = {a.source_type for a in artifacts}
    assert SourceType.PREFERENCES in source_types
    assert SourceType.APP_DATA in source_types

    appdata_files = {a.source_file for a in artifacts if a.source_type == SourceType.APP_DATA}
    assert appdata_files == {"discord/settings.json"}  # Preferences never double-claimed
