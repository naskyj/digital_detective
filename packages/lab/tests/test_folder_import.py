"""dd_lab.imaging's folder-based acquisition - extracting evidence from a
plain folder that already holds one subfolder per app, rather than a disk
image or an image-shaped folder (dissect's own directory loader, reached
via open_image()/acquire_from_image(), already handles that case).

Not "live collection": dd-lab still never reads a live machine's own
AppData paths - this is for a folder of already-extracted, static evidence
(an examiner's own prior extraction, files handed over some other way).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dd_core.casedir import CaseLayout
from dd_core.manifest import verify_case
from dd_lab.agent import run_pipeline
from dd_lab.case import CaseMeta
from dd_lab.imaging import acquire_from_folder, discover_apps_in_folder


def _plant_profile(app_dir: Path) -> None:
    (app_dir / "Local Storage" / "leveldb").mkdir(parents=True)
    (app_dir / "Local State").write_text('{"os_crypt":{"encrypted_key":"abc"}}')
    (app_dir / "Preferences").write_text('{"token":"discordapp.com faketoken.value.here"}')


@pytest.fixture()
def evidence_folder(tmp_path) -> Path:
    folder = tmp_path / "extracted_evidence"
    _plant_profile(folder / "Discord")
    return folder


def test_discover_apps_in_folder_finds_the_planted_profile(evidence_folder):
    apps = discover_apps_in_folder(evidence_folder)
    assert any(a.name == "Discord" for a in apps)


def test_discover_apps_in_folder_ignores_non_profile_siblings(tmp_path):
    folder = tmp_path / "mixed"
    _plant_profile(folder / "Discord")
    (folder / "readme.txt").write_text("not a profile")
    (folder / "EmptyDir").mkdir(parents=True)
    apps = discover_apps_in_folder(folder)
    names = {a.name for a in apps}
    assert names == {"Discord"}


def test_acquire_from_folder_produces_a_verifiable_case(tmp_path, evidence_folder):
    result = acquire_from_folder(evidence_folder, tmp_path, operator="Test Examiner")

    assert result.file_count > 0
    assert "discord" in [a.lower() for a in result.apps]
    assert result.image_sha256 == ""  # no single "whole image" hash for a folder

    v = verify_case(result.case_dir)
    assert v.ok, v.issues
    assert v.signature_valid


def test_acquire_from_folder_host_is_honestly_unknown_not_the_analysts_machine(tmp_path, evidence_folder):
    """No dissect Target to read a host identity from for a plain folder -
    must fall back to an honest "unknown", the same principle
    _image_host() already applies when an image's own OS can't be
    determined (never silently substituting the analyst's own machine)."""
    import platform

    from dd_core.manifest import Manifest

    result = acquire_from_folder(evidence_folder, tmp_path)
    manifest = Manifest.read(result.case_dir)
    assert manifest.host.hostname != platform.node()
    assert "unknown" in manifest.host.hostname.lower()


def test_folder_derived_case_runs_through_the_normal_pipeline(tmp_path, evidence_folder):
    """The whole point: nothing downstream needs to know the evidence came
    from a folder instead of an image or a live collection."""
    result = acquire_from_folder(evidence_folder, tmp_path, operator="Test Examiner", case_id="folder_case")
    CaseMeta(case_id="folder_case", name="Folder case", keywords=["token"]).save(
        CaseLayout(tmp_path, "folder_case")
    )

    outcome = run_pipeline(tmp_path, "folder_case")
    assert outcome.ok
    assert outcome.report_path and Path(outcome.report_path).is_file()
    assert result.case_dir.name == "folder_case"


_DECOY = Path.home() / "Downloads/decoy"


@pytest.mark.skipif(not _DECOY.is_dir(), reason="no local decoy evidence to test against")
def test_acquire_from_folder_against_real_multi_app_evidence(tmp_path):
    """Real evidence: a folder with Discord, an MSIX-packaged Slack, and a
    Roblox install with no Electron profile of its own but two embedded
    WebView2 profiles - all four should come through in one case."""
    result = acquire_from_folder(_DECOY, tmp_path, operator="Test Examiner")
    assert result.file_count > 0
    lowered = [a.lower() for a in result.apps]
    assert any("discord" in a for a in lowered)
    assert any("slack" in a for a in lowered)
    assert any("roblox" in a and "webview2" in a for a in lowered)

    v = verify_case(result.case_dir)
    assert v.ok, v.issues
