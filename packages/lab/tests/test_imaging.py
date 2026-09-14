"""dd_lab.imaging — extraction straight from a real raw disk image.

Builds an actual FAT32 raw disk image (via macOS's `hdiutil`/`newfs_msdos`,
root-free) containing a planted Chromium/Electron profile, then opens that
image file directly with `dissect` — never mounted by the OS — the same way
an investigator's real raw/dd image would be read. This is a real integration
test, not a mock: if `dissect` or its file-discovery logic regresses, this
catches it.

Skipped where the platform/tools to BUILD a test image aren't available
(non-macOS CI); the extraction code itself is platform-independent — only
this test's fixture-building step is macOS-specific.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("dissect.target", reason="'image' extra not installed")

from dd_core.casedir import CaseLayout  # noqa: E402
from dd_core.manifest import verify_case  # noqa: E402
from dd_lab.agent import run_pipeline  # noqa: E402
from dd_lab.case import CaseMeta  # noqa: E402
from dd_lab.imaging import acquire_from_image, discover_apps, open_image  # noqa: E402

from tests.imaging_fixtures import HAVE_MAC_IMAGE_TOOLS, SKIP_REASON, build_fat_image  # noqa: E402

pytestmark = pytest.mark.skipif(not HAVE_MAC_IMAGE_TOOLS, reason=SKIP_REASON)


@pytest.fixture(scope="module")
def raw_fat_image(tmp_path_factory) -> Path:
    return build_fat_image(tmp_path_factory.mktemp("image") / "suspect_disk.img")


def test_discover_apps_finds_the_planted_profile(raw_fat_image):
    target = open_image(raw_fat_image)
    apps = discover_apps(target)
    names = [a.name.lower() for a in apps]
    assert any("discord" in n for n in names)


def test_acquire_from_image_produces_a_verifiable_case(tmp_path, raw_fat_image):
    result = acquire_from_image(raw_fat_image, tmp_path, operator="Test Examiner")

    assert result.file_count > 0
    assert "discord" in [a.lower() for a in result.apps]
    assert len(result.image_sha256) == 64

    v = verify_case(result.case_dir)
    assert v.ok, v.issues
    assert v.signature_valid


def test_image_derived_case_runs_through_the_normal_pipeline(tmp_path, raw_fat_image):
    """The whole point: nothing downstream needs to know the evidence came
    from an image instead of a live collection."""
    result = acquire_from_image(raw_fat_image, tmp_path, operator="Test Examiner", case_id="image_case")
    CaseMeta(case_id="image_case", name="Image case", keywords=["token"]).save(
        CaseLayout(tmp_path, "image_case")
    )

    outcome = run_pipeline(tmp_path, "image_case")
    assert outcome.ok
    assert outcome.report_path and Path(outcome.report_path).is_file()
    assert result.case_dir.name == "image_case"
