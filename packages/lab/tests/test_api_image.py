"""/api/image/list and /api/cases/{id}/sources — the UI wiring for image-
based acquisition. Uses the same real raw-image fixture as test_imaging.py."""

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


def test_add_source_produces_a_case_reachable_via_the_normal_api(client, raw_fat_image):
    created = client.post("/api/cases", json={"name": "API image case", "case_id": "api_image_case"})
    assert created.status_code == 200

    r = client.post("/api/cases/api_image_case/sources", json={
        "source_path": str(raw_fat_image), "operator": "Test Examiner", "source_label": "",
    })
    assert r.status_code == 200
    job = _poll(client, r.json()["job_id"])
    assert job["status"] == "done", job

    result = job["result"]
    assert result["case_id"] == "api_image_case"
    assert "discord" in [a.lower() for a in result["apps"]]

    # the case is now a normal, fully-evidenced case as far as the rest of
    # the API is concerned - no separate "now register it" step needed
    got = client.get("/api/cases/api_image_case")
    assert got.status_code == 200
    assert got.json()["integrity"]["ok"] is True


@pytest.fixture()
def evidence_folder(tmp_path) -> Path:
    """A plain folder holding one already-extracted app subfolder - the
    third input shape /api/image/list and /api/image/extract accept
    alongside a real image file and an OS-root-shaped folder."""
    folder = tmp_path / "extracted_evidence"
    profile = folder / "Discord"
    (profile / "Local Storage" / "leveldb").mkdir(parents=True)
    (profile / "Local State").write_text('{"os_crypt":{"encrypted_key":"abc"}}')
    return folder


def test_image_list_accepts_a_flat_evidence_folder(client, evidence_folder):
    r = client.post("/api/image/list", json={"image_path": str(evidence_folder)})
    assert r.status_code == 200
    names = [a["name"].lower() for a in r.json()["apps"]]
    assert any("discord" in n for n in names)


def test_add_source_accepts_a_flat_evidence_folder(client, evidence_folder):
    created = client.post("/api/cases", json={"name": "API folder case", "case_id": "api_folder_case"})
    assert created.status_code == 200

    r = client.post("/api/cases/api_folder_case/sources", json={
        "source_path": str(evidence_folder), "operator": "Test Examiner", "source_label": "",
    })
    assert r.status_code == 200
    job = _poll(client, r.json()["job_id"])
    assert job["status"] == "done", job

    result = job["result"]
    assert result["case_id"] == "api_folder_case"
    assert "discord" in [a.lower() for a in result["apps"]]
    assert result["image_sha256"] == ""


@pytest.fixture()
def evidence_folder_2(tmp_path) -> Path:
    """A SECOND flat evidence folder with an app of the SAME name ("Discord")
    as `evidence_folder` - the exact scenario source labelling exists for:
    two people's Discord data must not land in, and silently merge into,
    the same evidence folder."""
    folder = tmp_path / "extracted_evidence_2"
    profile = folder / "Discord"
    (profile / "Local Storage" / "leveldb").mkdir(parents=True)
    (profile / "Local State").write_text('{"os_crypt":{"encrypted_key":"xyz"}}')
    return folder


def test_two_labelled_sources_with_the_same_app_name_stay_separate(
    client, evidence_folder, evidence_folder_2
):
    created = client.post("/api/cases", json={"name": "Multi-source case", "case_id": "multi_src"})
    assert created.status_code == 200
    # not evidenced yet - a normal, non-404 state
    empty = client.get("/api/cases/multi_src")
    assert empty.status_code == 200
    assert empty.json()["integrity"] is None
    assert empty.json()["evidence"]["apps"] == []

    r1 = client.post("/api/cases/multi_src/sources", json={
        "source_path": str(evidence_folder), "operator": "Examiner", "source_label": "Suspect Laptop",
    })
    job1 = _poll(client, r1.json()["job_id"])
    assert job1["status"] == "done", job1

    r2 = client.post("/api/cases/multi_src/sources", json={
        "source_path": str(evidence_folder_2), "operator": "Examiner", "source_label": "Victim Phone",
    })
    job2 = _poll(client, r2.json()["job_id"])
    assert job2["status"] == "done", job2

    # both sources' "Discord" apps got distinct evidence folders, not one merged folder
    apps1 = {a.lower() for a in job1["result"]["apps"]}
    apps2 = {a.lower() for a in job2["result"]["apps"]}
    assert apps1 and apps2 and apps1.isdisjoint(apps2)

    from dd_core.casedir import CaseLayout
    from dd_lab.api import main as main_mod

    layout = CaseLayout(main_mod._settings.cases_root, "multi_src")
    evidence_dirs = {p.name for p in layout.evidence.iterdir()}
    assert len(evidence_dirs) == 2  # not collapsed into one

    # the SAME signing key signed both acquisitions - not silently regenerated
    from dd_core.manifest import Manifest

    manifest = Manifest.read(layout.dir)
    assert len(manifest.target_apps) == 2
    key_path = layout.dir / "keys" / "collector_key.pem"
    assert key_path.is_file()  # one key file, not overwritten with a second identity

    # /cases/{id} reflects both apps, and CaseMeta remembers both sources
    got = client.get("/api/cases/multi_src")
    assert sorted(got.json()["evidence"]["apps"]) == sorted(t.name for t in manifest.target_apps)
    meta_sources = got.json()["meta"]["sources"]
    assert {s["label"] for s in meta_sources} == {"Suspect Laptop", "Victim Phone"}


def test_colliding_source_labels_rejected(client, evidence_folder, evidence_folder_2):
    """Two labels that sanitise to the same evidence-folder prefix would
    merge their apps right back together - rejected with a clear 400
    instead of silently corrupting the second source's evidence."""
    created = client.post("/api/cases", json={"name": "Collision case", "case_id": "collide_case"})
    assert created.status_code == 200
    r1 = client.post("/api/cases/collide_case/sources", json={
        "source_path": str(evidence_folder), "operator": "Examiner", "source_label": "Bob's Laptop",
    })
    assert _poll(client, r1.json()["job_id"])["status"] == "done"

    # "Bob's Laptop" -> safe_name -> "Bob_s_Laptop" (apostrophe and space
    # each collapse to one "_") - this literal string sanitises to itself
    # (underscore is already an allowed character), so it collides exactly.
    r2 = client.post("/api/cases/collide_case/sources", json={
        "source_path": str(evidence_folder_2), "operator": "Examiner", "source_label": "Bob_s_Laptop",
    })
    assert r2.status_code == 400
