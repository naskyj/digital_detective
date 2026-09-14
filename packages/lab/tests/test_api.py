"""API surface — exercised with FastAPI's TestClient against a real sample case."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dd_lab.case import CaseMeta

from tests.sample_case import build_sample_case

_CANDIDATES = [
    Path.home() / "Library/Application Support/Code",
    Path.home() / "Library/Application Support/Postman",
    Path.home() / ".config/Code",
]
_FAST = frozenset({"local_storage", "session_storage", "preferences", "cookies", "local_state"})


def _profile():
    for p in _CANDIDATES:
        if (p / "Local Storage" / "leveldb").is_dir() or (p / "Preferences").is_file():
            return p
    return None


pytestmark = pytest.mark.skipif(_profile() is None, reason="no Electron profile available")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # point the app at a throwaway cases root before importing the module state.
    # DD_CONFIG_DIR (not XDG_CONFIG_HOME) is what actually isolates this on
    # macOS — settings.py's config dir ignores XDG_CONFIG_HOME there.
    monkeypatch.setenv("DD_CONFIG_DIR", str(tmp_path / "cfg"))
    import importlib

    from dd_lab.api import settings as settings_mod

    importlib.reload(settings_mod)
    from dd_lab.api import main as main_mod

    importlib.reload(main_mod)
    main_mod._settings.cases_root = str(tmp_path / "cases")
    main_mod._settings.save()

    # a collected + case-metadata'd sample case
    profile = _profile()
    build_sample_case(tmp_path / "cases", "case_api_1", "app", profile, include=_FAST)
    from dd_core.casedir import CaseLayout

    CaseMeta(case_id="case_api_1", name="API test", keywords=["token"]).save(
        CaseLayout(tmp_path / "cases", "case_api_1")
    )
    return TestClient(main_mod.app)


def test_health_and_settings(client):
    assert client.get("/health").json()["ok"] is True
    s = client.get("/api/settings").json()
    assert "llm" in s and s["llm"]["provider"] in {"ollama", "openai", "anthropic"}


def test_apps_listing(client):
    r = client.get("/api/apps").json()
    assert "apps" in r


def test_case_lifecycle_via_api(client):
    cases = client.get("/api/cases").json()
    assert cases["counts"]["total"] >= 1

    detail = client.get("/api/cases/case_api_1").json()
    assert detail["integrity"]["ok"] is True
    assert detail["warehouse"] is None  # not processed yet

    job = client.post("/api/cases/case_api_1/run").json()
    for _ in range(120):
        j = client.get(f"/api/jobs/{job['job_id']}").json()
        if j["status"] in {"done", "error"}:
            break
        time.sleep(0.5)
    assert j["status"] == "done", j.get("error")

    detail = client.get("/api/cases/case_api_1").json()
    assert detail["warehouse"]["artifacts"] > 0
    assert detail["reports"]

    arts = client.get("/api/cases/case_api_1/artifacts?limit=10").json()
    assert "rows" in arts

    bad = client.post("/api/cases/case_api_1/query", json={"sql": "DROP TABLE artifacts"}).json()
    assert bad["ok"] is False

    report = client.get("/api/cases/case_api_1/report")
    assert report.status_code == 200
    assert "Forensic Analysis Report" in report.text

    audit = client.get("/api/cases/case_api_1/audit").json()
    assert audit["custody"] and audit["audit"]
