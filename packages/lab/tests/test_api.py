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
    # the judge is configurable from Settings alone - no CLI/env required
    assert "judge_llm" in s and "use_judge" in s

    updated = client.put("/api/settings", json={
        "judge_llm_provider": "anthropic", "judge_llm_model": "claude-sonnet-5", "use_judge": False,
    }).json()
    assert updated["judge_llm"]["provider"] == "anthropic"
    assert updated["judge_llm"]["model"] == "claude-sonnet-5"
    assert updated["use_judge"] is False
    # the primary agent's own config is untouched by a judge-only update
    assert updated["llm"]["provider"] in {"ollama", "openai", "anthropic"}

    probe = client.post("/api/settings/test-judge-llm").json()
    assert "ok" in probe and probe["provider"] == "anthropic"


def test_active_jobs_listing(client):
    """Backs the persistent sidebar indicator: a job shows up while it's
    queued/running and disappears once it finishes - proof the indicator
    reflects genuine server-side state, not just what the current page thinks."""
    from dd_lab.api import main as main_mod

    def slow_work(progress):
        time.sleep(0.3)
        return {"ok": True}

    job_id = main_mod._jobs.submit("test", slow_work)

    found = False
    for _ in range(20):
        if any(j["id"] == job_id for j in client.get("/api/jobs").json()["jobs"]):
            found = True
            break
        time.sleep(0.05)
    assert found, "job never appeared in the active list while running"

    for _ in range(40):
        if client.get(f"/api/jobs/{job_id}").json()["status"] == "done":
            break
        time.sleep(0.05)
    jobs = client.get("/api/jobs").json()["jobs"]
    assert not any(j["id"] == job_id for j in jobs), "finished job still listed as active"


def test_interrupted_run_status_is_reconciled_not_stuck_running(client, tmp_path):
    """Found on real evidence: a run interrupted by the app being closed or
    restarted mid-analysis leaves case.json saying "running" forever, since
    nothing survives to ever update it. GET /api/cases/{id} and GET
    /api/cases must both notice (no job is actually active for this case)
    and correct the status instead of showing a phantom in-progress run."""
    from dd_core.casedir import CaseLayout
    from dd_lab.case import CaseStatus

    layout = CaseLayout(tmp_path / "cases", "case_api_1")
    meta = CaseMeta.load(layout)
    meta.status = CaseStatus.RUNNING
    meta.last_message = "agent (claude-sonnet-5)"
    meta.save(layout)

    detail = client.get("/api/cases/case_api_1").json()
    assert detail["meta"]["status"] == "error"
    assert "interrupted" in detail["meta"]["last_message"].lower()

    # and the fix persists to disk, not just the one response
    assert CaseMeta.load(layout).status == CaseStatus.ERROR

    listing = client.get("/api/cases").json()
    case_row = next(c for c in listing["cases"] if c["case_id"] == "case_api_1")
    assert case_row["status"] == "error"
    assert listing["counts"]["running"] == 0


def test_no_live_collection_routes(client):
    """The lab deliberately never collects evidence live from the machine it
    runs on - that's dd_collector's job. /api/apps and /api/collect must not
    exist here."""
    assert client.get("/api/apps").status_code == 404
    assert client.post("/api/collect", json={"profiles": []}).status_code == 404


def test_case_lifecycle_via_api(client):
    cases = client.get("/api/cases").json()
    assert cases["counts"]["total"] >= 1

    detail = client.get("/api/cases/case_api_1").json()
    assert detail["integrity"]["ok"] is True
    assert detail["warehouse"] is None  # not processed yet

    assert client.get("/api/cases/case_api_1/classification").status_code == 409
    assert client.get("/api/cases/case_api_1/artifact-tree").status_code == 409

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
    # per-voter classification breakdown must be present on every row, not
    # just category/confidence - this is what the UI's classification panel
    # and artifact-browser "why" view read.
    row = arts["rows"][0]
    for col in ("artifact_id", "regex_vote", "keyword_vote", "ner_vote", "model_vote", "evidence_spans"):
        assert col in row

    clf = client.get("/api/cases/case_api_1/classification").json()
    assert clf["by_category"]
    for c in clf["by_category"]:
        for col in ("category", "n", "mean_conf", "regex_agree", "keyword_agree", "ner_agree", "model_agree"):
            assert col in c
        assert c["category"] in clf["examples"]

    # Backs the case page's left-nav explorer: per-app, per-source-type
    # counts, each app's sources summing to its total.
    tree = client.get("/api/cases/case_api_1/artifact-tree").json()
    assert tree["apps"]
    app_row = next(a for a in tree["apps"] if a["app"] == "app")
    assert app_row["sources"]
    assert app_row["total"] == sum(s["n"] for s in app_row["sources"])
    src = app_row["sources"][0]
    filtered = client.get(
        "/api/cases/case_api_1/artifacts",
        params={"app": "app", "source_type": src["source_type"], "limit": 1000},
    ).json()
    assert filtered["total"] == src["n"]

    bad = client.post("/api/cases/case_api_1/query", json={"sql": "DROP TABLE artifacts"}).json()
    assert bad["ok"] is False

    report = client.get("/api/cases/case_api_1/report")
    assert report.status_code == 200
    assert "Forensic Analysis Report" in report.text

    audit = client.get("/api/cases/case_api_1/audit").json()
    assert audit["custody"] and audit["audit"]


def test_cache_files_route_works_without_a_run(client):
    """Unlike /artifacts, the cache-file browser reads straight off the
    evidence directory - it must work before "Run analysis" has ever
    executed, and must not error just because this sample case's fixture
    didn't copy a Cache directory (it copies only the fast sources)."""
    detail = client.get("/api/cases/case_api_1").json()
    assert detail["warehouse"] is None  # confirms this case is genuinely unprocessed

    d = client.get("/api/cases/case_api_1/cache-files").json()
    assert d["rows"] == []
    assert d["total"] == 0
    assert "app" in d["apps"]


def test_cache_files_open_rejects_a_directory_not_in_this_case(client):
    """`dir` is validated against the app's real cache directories before
    anything is opened - it must never resolve an arbitrary path."""
    r = client.get(
        "/api/cases/case_api_1/cache-files/open",
        params={"app": "app", "dir": "../../../../etc", "key": "x"},
    )
    assert r.status_code == 404


@pytest.mark.slow
def test_cache_files_route_with_real_cache_content(tmp_path, monkeypatch):
    """End to end against a real Chromium HTTP cache: the listing names a
    real physical file per entry, and "open" returns its actual decoded
    bytes - the ChromeCacheView-style workflow this API exists for."""
    monkeypatch.setenv("DD_CONFIG_DIR", str(tmp_path / "cfg"))
    import importlib

    from dd_lab.api import settings as settings_mod

    importlib.reload(settings_mod)
    from dd_lab.api import main as main_mod

    importlib.reload(main_mod)
    main_mod._settings.cases_root = str(tmp_path / "cases")
    main_mod._settings.save()

    profile = _profile()
    build_sample_case(tmp_path / "cases", "case_cache_api", "app", profile, include=frozenset({"cache"}))
    from dd_core.casedir import CaseLayout

    CaseMeta(case_id="case_cache_api", name="cache API test").save(
        CaseLayout(tmp_path / "cases", "case_cache_api")
    )
    client = TestClient(main_mod.app)

    d = client.get("/api/cases/case_cache_api/cache-files", params={"app": "app", "limit": 20}).json()
    assert d["total"] > 0
    row = d["rows"][0]
    for col in ("entry_id", "app", "cache_dir", "physical_file", "key", "url"):
        assert col in row
    assert row["physical_file"]

    openable = next(r for r in d["rows"] if r["openable"])
    body = client.get(
        "/api/cases/case_cache_api/cache-files/open",
        params={"app": "app", "dir": openable["cache_dir"], "key": openable["key"]},
    )
    assert body.status_code == 200
    assert len(body.content) > 0

    q = client.get(
        "/api/cases/case_cache_api/cache-files", params={"app": "app", "q": openable["url"][:20]}
    ).json()
    assert q["total"] >= 1
