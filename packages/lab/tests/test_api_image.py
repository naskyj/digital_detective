"""/api/image/list and /api/image/extract — the UI wiring for image-based
acquisition. Uses the same real raw-image fixture as test_imaging.py."""

from __future__ import annotations

import importlib
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("dissect.target", reason="'image' extra not installed")

from tests.imaging_fixtures import HAVE_MAC_IMAGE_TOOLS, SKIP_REASON, build_fat_image  # noqa: E402

pytestmark = pytest.mark.skipif(not HAVE_MAC_IMAGE_TOOLS, reason=SKIP_REASON)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DD_CONFIG_DIR", str(tmp_path / "cfg"))
    from dd_lab.api import settings as settings_mod

    importlib.reload(settings_mod)
    from dd_lab.api import main as main_mod

    importlib.reload(main_mod)
    main_mod._settings.cases_root = str(tmp_path / "cases")
    main_mod._settings.save()
    return TestClient(main_mod.app)


@pytest.fixture(scope="module")
def raw_fat_image(tmp_path_factory) -> Path:
    return build_fat_image(tmp_path_factory.mktemp("image") / "suspect_disk.img")


def test_image_list_finds_the_planted_profile(client, raw_fat_image):
    r = client.post("/api/image/list", json={"image_path": str(raw_fat_image)})
    assert r.status_code == 200
    names = [a["name"].lower() for a in r.json()["apps"]]
    assert any("discord" in n for n in names)


def test_image_list_bad_path_is_a_clean_400_not_a_crash(client):
    r = client.post("/api/image/list", json={"image_path": "/nonexistent/path/to/image.raw"})
    assert r.status_code == 400


def _poll(client, job_id, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.2)
    raise TimeoutError("job did not finish")


def test_image_extract_produces_a_case_reachable_via_the_normal_api(client, raw_fat_image):
    r = client.post("/api/image/extract", json={
        "image_path": str(raw_fat_image), "operator": "Test Examiner", "case_id": "api_image_case",
    })
    assert r.status_code == 200
    job = _poll(client, r.json()["job_id"])
    assert job["status"] == "done", job

    result = job["result"]
    assert result["case_id"] == "api_image_case"
    assert "discord" in [a.lower() for a in result["apps"]]

    # the case is now a normal case as far as the rest of the API is concerned
    from dd_core.casedir import CaseLayout
    from dd_lab.api import main as main_mod
    from dd_lab.case import CaseMeta

    CaseMeta(case_id="api_image_case", name="API image case").save(
        CaseLayout(main_mod._settings.cases_root, "api_image_case")
    )
    got = client.get("/api/cases/api_image_case")
    assert got.status_code == 200
    assert got.json()["integrity"]["ok"] is True
