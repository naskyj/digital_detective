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
    # relevance_model_dir is a SEPARATE field/model from classifier_model_dir
    # (see ensemble.py's module docstring) - both must round-trip independently.
    assert "relevance_model_dir" in s
    updated = client.put("/api/settings", json={"relevance_model_dir": "/tmp/some-relevance-model"}).json()
    assert updated["relevance_model_dir"] == "/tmp/some-relevance-model"
    assert updated["classifier_model_dir"] == s["classifier_model_dir"]  # untouched by this update

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


def test_saving_settings_does_not_kill_a_running_job(client):
    """Regression for a real, destructive bug hit on a live 675k-artifact
    case. JobManager.__init__ marks every running job as errored (correct
    on a genuine app restart - those threads are gone). put_settings used
    to rebuild it unconditionally, so saving settings mid-run falsely
    flagged the in-flight job "interrupted", which ALSO cleared the
    active_for() guard that blocks a second job on the same case - letting
    a full Run analysis start on top of it and delete an app's artifacts
    via reset_app() mid-flight."""
    from dd_lab.api import main as main_mod

    def slow_work(progress):
        time.sleep(1.5)
        return {"ok": True}

    job_id = main_mod._jobs.submit("test", slow_work, case_id="case_api_1")
    for _ in range(20):
        if main_mod._jobs.get(job_id)["status"] == "running":
            break
        time.sleep(0.05)

    client.put("/api/settings", json={"use_zeroshot": False})

    assert main_mod._jobs.get(job_id)["status"] == "running", "settings save killed a live job"
    assert main_mod._jobs.active_for("case_api_1"), "settings save cleared the concurrent-job guard"
    assert client.post("/api/cases/case_api_1/run").status_code == 409


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
    for col in (
        "artifact_id", "relevant", "relevance_score", "regex_vote", "keyword_vote", "ner_vote",
        "model_vote", "entropy_vote", "evidence_spans", "deleted",
    ):
        assert col in row
    assert isinstance(row["deleted"], bool)
    assert isinstance(row["relevant"], bool)

    clf = client.get("/api/cases/case_api_1/classification").json()
    assert clf["by_category"]
    for c in clf["by_category"]:
        for col in (
            "category", "n", "mean_conf", "regex_agree", "keyword_agree", "ner_agree", "model_agree",
            "entropy_agree",
        ):
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

    # Relevance breakdown - the now-primary output, alongside the existing
    # per-category classification breakdown checked above.
    rel = client.get("/api/cases/case_api_1/relevance").json()
    assert rel["by_relevance"]
    for r in rel["by_relevance"]:
        for col in ("relevant", "artifacts", "mean_relevance_score", "mean_relevance_model_score"):
            assert col in r

    # classify_warehouse's actual entry point - previously unreachable from
    # any endpoint. Cheap in-place re-classify, no re-parse.
    reclf_job = client.post("/api/cases/case_api_1/reclassify").json()
    for _ in range(120):
        j = client.get(f"/api/jobs/{reclf_job['job_id']}").json()
        if j["status"] in {"done", "error"}:
            break
        time.sleep(0.5)
    assert j["status"] == "done", j.get("error")
    assert j["result"]["classified"] == detail["warehouse"]["artifacts"]

    # Artifact detail now surfaces the vote fields it always had access to
    # but never selected.
    one = client.get(f"/api/cases/case_api_1/artifacts/{row['artifact_id']}").json()
    vote_cols = ("regex_vote", "keyword_vote", "ner_vote", "model_vote", "entropy_vote")
    for col in (*vote_cols, "relevance_model_score"):
        assert col in one


def test_artifacts_search_matches_source_locator(client):
    """The `q` filter must also match source_locator, not just key/value -
    added so a process-detail view can find every string carved from one
    PID ("pid<PID>:string#n" locators, see dd_lab/memory/analyze.py)
    without a dedicated endpoint. Load one hand-built artifact directly
    (bypassing the slow full-ingest job) whose key/value do NOT contain the
    search term at all - only its locator does."""
    from datetime import UTC, datetime

    from dd_core.casedir import CaseLayout
    from dd_core.enums import SourceType, ValueType
    from dd_core.schema import NormalizedArtifact
    from dd_lab.api import main as main_mod
    from dd_lab.etl.warehouse import Warehouse

    layout = CaseLayout(main_mod._settings.cases_root, "case_api_1")
    wh = Warehouse(layout.warehouse)
    try:
        wh.load_artifacts([NormalizedArtifact(
            artifact_id="test-pid-locator-artifact",
            case_id="case_api_1", app="discord", source_type=SourceType.MEMORY_STRING,
            source_file="_system/memory/physmem.raw#pid14380",
            source_locator="pid14380:string#7",
            key="mem.string", value="totally unrelated text content",
            value_type=ValueType.STRING, extracted_ts=datetime.now(UTC),
        )])
    finally:
        wh.close()

    hit = client.get("/api/cases/case_api_1/artifacts", params={"q": "pid14380:"}).json()
    assert hit["total"] == 1
    assert hit["rows"][0]["artifact_id"] == "test-pid-locator-artifact"

    miss = client.get("/api/cases/case_api_1/artifacts", params={"q": "pid99999:"}).json()
    assert miss["total"] == 0


def test_correlations_endpoint_finds_a_cross_source_entity(client):
    """Phase 5 - the interactive counterpart to the static report's §5.
    Same setup as the design-discussion scenario in test_investigate.py: one
    Roblox asset id in a Discord Cache blob and an IndexedDB message row,
    hand-loaded directly (bypassing the slow full-ingest job)."""
    from datetime import UTC, datetime

    from dd_core.casedir import CaseLayout
    from dd_core.enums import ArtifactCategory as C
    from dd_core.enums import SourceType as S
    from dd_core.schema import ClassificationResult, NormalizedArtifact
    from dd_lab.api import main as main_mod
    from dd_lab.etl.warehouse import Warehouse

    layout = CaseLayout(main_mod._settings.cases_root, "case_api_1")
    wh = Warehouse(layout.warehouse)
    try:
        wh.load_artifacts([
            NormalizedArtifact(
                artifact_id="corr-1", case_id="case_api_1", app="discord", source_type=S.CACHE,
                source_file="discord/Cache/f1", source_locator="http-body",
                key="cache.body.embed", value="rbxassetid://5551112223",
                extracted_ts=datetime.now(UTC),
                classification=ClassificationResult(category=C.CHAT_CONTENT, confidence=0.9),
            ),
            NormalizedArtifact(
                artifact_id="corr-2", case_id="case_api_1", app="discord", source_type=S.INDEXEDDB,
                source_file="discord/IndexedDB/messages.leveldb", source_locator="store/998",
                key="content", value="check this rbxassetid://5551112223",
                extracted_ts=datetime.now(UTC),
                classification=ClassificationResult(category=C.CHAT_CONTENT, confidence=0.9),
            ),
        ])
    finally:
        wh.close()

    d = client.get("/api/cases/case_api_1/correlations").json()
    assert d["ok"] is True
    hit = next((e for e in d["entities"] if e["value"] == "5551112223"), None)
    assert hit is not None, d["entities"]
    assert hit["entity_type"] == "roblox_asset_id"
    assert set(hit["source_types"]) >= {"cache", "indexeddb"}


def test_timeline_endpoint_paginates_and_filters(client):
    """Phase 5 - the interactive counterpart to the static report's §7,
    which is generated once and hard-capped at 120 rows excluding GENERAL.
    This one is a normal paginated/filterable browse endpoint instead."""
    from datetime import UTC, datetime, timedelta

    from dd_core.casedir import CaseLayout
    from dd_core.enums import ArtifactCategory as C
    from dd_core.enums import SourceType as S
    from dd_core.schema import ClassificationResult, NormalizedArtifact
    from dd_lab.api import main as main_mod
    from dd_lab.etl.warehouse import Warehouse

    layout = CaseLayout(main_mod._settings.cases_root, "case_api_1")
    now = datetime.now(UTC)
    wh = Warehouse(layout.warehouse)
    try:
        wh.load_artifacts([
            NormalizedArtifact(
                artifact_id=f"tl-{i}", case_id="case_api_1", app="discord", source_type=S.MEMORY_PROCESS,
                source_file="_system/memory/physmem.raw", source_locator=f"windows.pslist.PsList#{i}",
                key="mem.process.CreateTime", value=f"process event {i}",
                record_ts=now + timedelta(minutes=i), extracted_ts=now,
                classification=ClassificationResult(
                    category=C.GENERAL if i == 0 else C.SYSTEM_DETAIL, confidence=0.5,
                ),
            )
            for i in range(3)
        ] + [
            # no record_ts at all - must never appear on the timeline
            NormalizedArtifact(
                artifact_id="tl-no-ts", case_id="case_api_1", app="discord", source_type=S.MEMORY_PROCESS,
                source_file="_system/memory/physmem.raw", source_locator="windows.pslist.PsList#99",
                key="mem.process.PID", value="99999", extracted_ts=now,
            ),
        ])
    finally:
        wh.close()

    d = client.get("/api/cases/case_api_1/timeline").json()
    assert d["total"] == 3  # the no-record_ts row is excluded
    assert [r["value"] for r in d["rows"]] == [f"process event {i}" for i in range(3)]  # oldest first

    paged = client.get("/api/cases/case_api_1/timeline", params={"limit": 2}).json()
    assert paged["shown"] == 2
    assert paged["total"] == 3

    filtered = client.get(
        "/api/cases/case_api_1/timeline", params={"category": "SYSTEM_DETAIL"}
    ).json()
    assert filtered["total"] == 2
    assert all(r["category"] == "SYSTEM_DETAIL" for r in filtered["rows"])
    assert "GENERAL" in filtered["facets"]["categories"]  # facets aren't narrowed by the active filter


def test_chat_send_and_history_via_api(client, monkeypatch):
    """Phase 6 - the model itself is stubbed (a real LLM call has no place
    in this test suite, same reasoning as test_agent_llm.py's FunctionModel
    pattern) via chat.py's own build_model, exercised through the real HTTP
    endpoints end to end: send a message, read the persisted history back."""
    from datetime import UTC, datetime

    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel

    import dd_lab.agent.chat as chat_mod
    from dd_core.casedir import CaseLayout
    from dd_core.enums import ArtifactCategory as C
    from dd_core.enums import SourceType as S
    from dd_core.schema import ClassificationResult, NormalizedArtifact
    from dd_lab.api import main as main_mod
    from dd_lab.etl.warehouse import Warehouse

    monkeypatch.setattr(
        chat_mod, "build_model",
        lambda cfg: FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart("hello from chat")])),
    )

    layout = CaseLayout(main_mod._settings.cases_root, "case_api_1")
    wh = Warehouse(layout.warehouse)
    try:
        wh.load_artifacts([NormalizedArtifact(
            artifact_id="chat-a1", case_id="case_api_1", app="discord", source_type=S.INDEXEDDB,
            source_file="discord/IndexedDB/x", source_locator="loc1", key="content", value="hi",
            extracted_ts=datetime.now(UTC),
            classification=ClassificationResult(category=C.CHAT_CONTENT, confidence=0.9),
        )])
    finally:
        wh.close()

    empty = client.get("/api/cases/case_api_1/chat").json()
    assert empty["turns"] == []

    sent = client.post("/api/cases/case_api_1/chat", json={"message": "hi there"}).json()
    assert sent == {"ok": True, "reply": "hello from chat"}

    history = client.get("/api/cases/case_api_1/chat").json()
    assert history["turns"] == [
        {"role": "user", "text": "hi there"},
        {"role": "assistant", "text": "hello from chat"},
    ]


def test_chat_routes_404_for_an_unknown_case(client):
    assert client.get("/api/cases/does-not-exist/chat").status_code == 404
    assert client.post("/api/cases/does-not-exist/chat", json={"message": "hi"}).status_code == 404


def test_investigator_can_save_a_finding_via_the_api(client):
    """The "Save as finding" action (Phase C, plan magical-dazzling-dragon.md)
    - an investigator-triggered save, NOT reachable from the chat LLM itself
    (chat.py's toolset deliberately excludes write_finding entirely). Goes
    through the exact same citation-verification write_finding always uses,
    so a real artifact_id is genuinely verified/attributed to "investigator",
    and a wrong one fails safely rather than being silently accepted."""
    from datetime import UTC, datetime

    from dd_core.casedir import CaseLayout
    from dd_core.enums import ArtifactCategory as C
    from dd_core.enums import SourceType as S
    from dd_core.schema import ClassificationResult, NormalizedArtifact
    from dd_lab.api import main as main_mod
    from dd_lab.etl.warehouse import Warehouse

    layout = CaseLayout(main_mod._settings.cases_root, "case_api_1")
    wh = Warehouse(layout.warehouse)
    try:
        wh.load_artifacts([NormalizedArtifact(
            artifact_id="finding-a1", case_id="case_api_1", app="discord", source_type=S.INDEXEDDB,
            source_file="discord/IndexedDB/x", source_locator="loc1", key="content",
            value="the password is hunter2", extracted_ts=datetime.now(UTC),
            classification=ClassificationResult(category=C.CREDENTIAL, confidence=0.9),
        )])
    finally:
        wh.close()

    good = client.post("/api/cases/case_api_1/findings", json={
        "claim": "Discord evidence shows a credential.",
        "citations": [{"artifact_id": "finding-a1", "quote": "hunter2"}],
    }).json()
    assert good["ok"] is True
    assert good["unverified_citations"] == []
    # Getting this into an actual generated report (attributed to
    # "investigator", not "agent") needs Phase D's regenerate-report
    # endpoint - covered by that phase's own test, not here (no report
    # exists yet for a case that's never been run/regenerated).

    bad = client.post("/api/cases/case_api_1/findings", json={
        "claim": "made up claim", "citations": [{"artifact_id": "does-not-exist"}],
    }).json()
    assert bad["ok"] is False
    assert bad["unverified_citations"] == ["does-not-exist"]

    assert client.post(
        "/api/cases/does-not-exist/findings", json={"claim": "x", "citations": []}
    ).status_code == 404


def test_regenerate_report_gets_a_saved_finding_into_an_actual_report(client):
    """The other half of "Save as finding": build_report/generate_report
    exist and work, but the only way to get a NEW report was the full /run
    job - a finding saved via POST /findings had nowhere to actually reach
    the report without it. This is the full round trip: load one artifact,
    save a finding citing it, regenerate the report (no /run at all), and
    confirm the finding shows up - attributed to "investigator", not
    "agent", proving provenance survives all the way through."""
    from datetime import UTC, datetime

    from dd_core.casedir import CaseLayout
    from dd_core.enums import ArtifactCategory as C
    from dd_core.enums import SourceType as S
    from dd_core.schema import ClassificationResult, NormalizedArtifact
    from dd_lab.api import main as main_mod
    from dd_lab.etl.warehouse import Warehouse

    layout = CaseLayout(main_mod._settings.cases_root, "case_api_1")
    wh = Warehouse(layout.warehouse)
    try:
        wh.load_artifacts([NormalizedArtifact(
            artifact_id="regen-a1", case_id="case_api_1", app="discord", source_type=S.INDEXEDDB,
            source_file="discord/IndexedDB/x", source_locator="loc1", key="content",
            value="a very specific claim marker xyz123", extracted_ts=datetime.now(UTC),
            classification=ClassificationResult(category=C.CHAT_CONTENT, confidence=0.9),
        )])
        wh.build_views()  # not automatic outside the full ingest pipeline - build_report needs the v_* views
    finally:
        wh.close()

    saved = client.post("/api/cases/case_api_1/findings", json={
        "claim": "Marker xyz123 found in Discord evidence.",
        "citations": [{"artifact_id": "regen-a1", "quote": "xyz123"}],
    }).json()
    assert saved["ok"] is True

    assert client.get("/api/cases/case_api_1/report").status_code == 404  # no report yet, confirms the gap

    job = client.post("/api/cases/case_api_1/regenerate-report").json()
    for _ in range(60):
        j = client.get(f"/api/jobs/{job['job_id']}").json()
        if j["status"] in {"done", "error"}:
            break
        time.sleep(0.2)
    assert j["status"] == "done", j.get("error")

    report_text = client.get("/api/cases/case_api_1/report").text
    assert "Marker xyz123 found in Discord evidence." in report_text
    assert "investigator" in report_text

    assert client.post("/api/cases/does-not-exist/regenerate-report").status_code == 404


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

    # "openable" only promises a body location was resolvable, not that the
    # body is non-empty - a real 0-byte response (e.g. a 304/204) is
    # legitimately openable and legitimately empty. Assert the real contract
    # (200, never a crash) for whichever the real cache happens to hand us,
    # and separately require at least one genuinely non-empty entry decodes
    # to real bytes - both facts matter, for different reasons.
    openables = [r for r in d["rows"] if r["openable"]]
    assert openables
    for r in openables:
        body = client.get(
            "/api/cases/case_cache_api/cache-files/open",
            params={"app": "app", "dir": r["cache_dir"], "key": r["key"]},
        )
        assert body.status_code == 200, (r["key"], body.status_code)
    non_empty = [r for r in openables if (r.get("size_bytes") or 0) > 0]
    assert non_empty, "no non-empty cache entry found to verify real content decoding"
    body = client.get(
        "/api/cases/case_cache_api/cache-files/open",
        params={"app": "app", "dir": non_empty[0]["cache_dir"], "key": non_empty[0]["key"]},
    )
    assert len(body.content) > 0
    openable = openables[0]

    q = client.get(
        "/api/cases/case_cache_api/cache-files", params={"app": "app", "q": openable["url"][:20]}
    ).json()
    assert q["total"] >= 1


@pytest.mark.slow
def test_artifact_browsing_extras_with_real_evidence(tmp_path, monkeypatch):
    """The per-app/source artifact table's derived cache URL and real
    pagination, the CSV export, and the new raw LevelDB/SQLite file
    browsers - all against real Local Storage, Cache, and Cookies data."""
    monkeypatch.setenv("DD_CONFIG_DIR", str(tmp_path / "cfg"))
    import importlib

    from dd_lab.api import settings as settings_mod

    importlib.reload(settings_mod)
    from dd_lab.api import main as main_mod

    importlib.reload(main_mod)
    main_mod._settings.cases_root = str(tmp_path / "cases")
    main_mod._settings.save()

    profile = _profile()
    build_sample_case(
        tmp_path / "cases", "case_browse_extras", "app", profile,
        include=frozenset({"cache", "local_storage", "cookies"}),
    )
    from dd_core.casedir import CaseLayout

    CaseMeta(case_id="case_browse_extras", name="browsing extras test").save(
        CaseLayout(tmp_path / "cases", "case_browse_extras")
    )
    # Ingest directly (parse + classify + load, like ingest_case elsewhere in
    # this suite) rather than through /run - the full deterministic pipeline
    # (memory analysis, keyword search, correlation, report) adds nothing
    # this test needs and is much slower against the full real cache.
    from dd_lab.ingest import ingest_case

    ingest_case(tmp_path / "cases", "case_browse_extras")
    client = TestClient(main_mod.app)

    # --- derived URL + real pagination on the per-app/source table --------
    cache_rows = client.get(
        "/api/cases/case_browse_extras/artifacts",
        params={"app": "app", "source_type": "cache", "limit": 5},
    ).json()
    assert cache_rows["total"] > 5
    assert all(r["url"] for r in cache_rows["rows"])

    non_cache = client.get(
        "/api/cases/case_browse_extras/artifacts",
        params={"app": "app", "source_type": "local_storage", "limit": 5},
    ).json()
    if non_cache["rows"]:
        assert all(r["url"] is None for r in non_cache["rows"])

    page1 = client.get(
        "/api/cases/case_browse_extras/artifacts",
        params={"app": "app", "source_type": "cache", "limit": 5, "offset": 0},
    ).json()
    page2 = client.get(
        "/api/cases/case_browse_extras/artifacts",
        params={"app": "app", "source_type": "cache", "limit": 5, "offset": 5},
    ).json()
    ids1 = {r["artifact_id"] for r in page1["rows"]}
    ids2 = {r["artifact_id"] for r in page2["rows"]}
    assert ids1 and ids2 and ids1.isdisjoint(ids2)

    # --- CSV export ----------------------------------------------------
    csv_resp = client.get(
        "/api/cases/case_browse_extras/artifacts/export.csv",
        params={"app": "app", "source_type": "cache"},
    )
    assert csv_resp.status_code == 200
    assert csv_resp.headers["content-type"].startswith("text/csv")
    import csv
    import io

    csv_rows = list(csv.reader(io.StringIO(csv_resp.text)))
    assert csv_rows[0][0] == "app"
    # capped at 200k rows even though this real case's cache has far more -
    # a safety cap, not a bug, so compare against min() rather than the raw total.
    # Parsed with csv.reader (not a naive newline split): a cached response
    # body can legitimately contain embedded newlines, which the writer
    # correctly quotes as part of one row.
    assert len(csv_rows) - 1 == min(cache_rows["total"], 200_000)

    # --- raw Local Storage physical-file browser ------------------------
    ls_files = client.get(
        "/api/cases/case_browse_extras/localstorage-files", params={"app": "app"}
    ).json()
    assert ls_files["files"]
    f = ls_files["files"][0]
    for col in ("filename", "kind", "size_bytes"):
        assert col in f
    opened = client.get(
        "/api/cases/case_browse_extras/localstorage-files/open",
        params={"app": "app", "file": f["filename"], "limit": 5},
    ).json()
    assert "records" in opened and opened["total"] >= 0

    bad = client.get(
        "/api/cases/case_browse_extras/localstorage-files/open",
        params={"app": "app", "file": "../../../../etc/passwd"},
    )
    assert bad.status_code == 404

    # --- raw database (Cookies) browser ---------------------------------
    db_files = client.get(
        "/api/cases/case_browse_extras/db-files", params={"app": "app"}
    ).json()
    assert db_files["files"]
    dbf = db_files["files"][0]
    assert dbf["tables"]
    table = dbf["tables"][0]["table"]
    db_opened = client.get(
        "/api/cases/case_browse_extras/db-files/open",
        params={"app": "app", "path": dbf["path"], "table": table, "limit": 5},
    ).json()
    assert "columns" in db_opened

    bad_path = client.get(
        "/api/cases/case_browse_extras/db-files/open",
        params={"app": "app", "path": "/etc/passwd", "table": table},
    )
    assert bad_path.status_code == 404


def test_narrative_endpoint_returns_episodes_and_coverage(client):
    """GET /narrative is the "what happened" view, computed server-side with
    no model and no network - the flat /timeline endpoint stays the row-level
    one. Coverage is returned alongside so a caller can never present the
    account as though it describes the whole case."""
    from datetime import UTC, datetime, timedelta

    from dd_core.casedir import CaseLayout
    from dd_core.enums import ArtifactCategory as C
    from dd_core.enums import SourceType as S
    from dd_core.manifest import AcquisitionHost, Manifest
    from dd_core.schema import ClassificationResult, NormalizedArtifact
    from dd_core.signing import generate_keypair
    from dd_lab.api import main as main_mod
    from dd_lab.etl.warehouse import Warehouse

    now = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)
    layout = CaseLayout(main_mod._settings.cases_root, "case_narr")
    # the endpoint requires a real case, i.e. a signed manifest - a bare
    # warehouse on disk is not one
    layout.dir.mkdir(parents=True, exist_ok=True)
    layout.create_evidence_skeleton()
    priv, pub = generate_keypair()
    Manifest(
        case_id="case_narr", collector_version="t", operator="t",
        host=AcquisitionHost(hostname="h", os="t", os_version="0", username="u", timezone="UTC"),
        started_ts=now, completed_ts=now, target_apps=[], entries=[],
        public_key_pem=pub.decode(),
    ).write(layout.dir, priv)
    wh = Warehouse(layout.warehouse)
    try:
        wh.load_artifacts([
            NormalizedArtifact(
                artifact_id=f"narr-{i}", case_id="case_narr", app=app,
                source_type=S.CACHE, source_file=f"{app}/Cache/f{i}",
                source_locator=f"loc{i}", key="cache.url", value=f"https://x/{i}",
                record_ts=now + timedelta(minutes=m), extracted_ts=now,
                classification=ClassificationResult(
                    category=C.CHAT_CONTENT, confidence=0.9, relevant=True,
                    relevance_score=0.9, classifier_version="test"),
            )
            for i, (app, m) in enumerate([("discord", 0), ("Slack", 10), ("discord", 600)])
        ])
        wh.build_views()
    finally:
        wh.close()

    d = client.get("/api/cases/case_narr/narrative").json()

    assert d["ok"] is True
    # 10-hour gap splits the third event into its own episode
    assert d["episode_count"] == 2
    assert d["episodes"][0]["apps"] == ["discord", "Slack"]   # first appearance order
    assert "Begins in discord" in d["episodes"][0]["summary"]
    assert d["coverage"]["events"] == 3
    assert "What happened" in d["narrative"]


def test_narrative_endpoint_404s_for_an_unknown_case(client):
    assert client.get("/api/cases/nope-not-a-case/narrative").status_code == 404


def test_correlations_are_cached_per_warehouse_state(client, monkeypatch):
    """build_correlation_report runs one un-indexed ILIKE full scan per
    candidate entity - 58 seconds on the real case - and was paid again on
    every page load. The result is a pure function of the warehouse, so it is
    memoised on (size, mtime) and any re-classify invalidates it by
    construction."""
    from datetime import UTC, datetime

    from dd_core.casedir import CaseLayout
    from dd_core.enums import ArtifactCategory as C
    from dd_core.enums import SourceType as S
    from dd_core.schema import ClassificationResult, NormalizedArtifact
    from dd_lab.api import main as main_mod
    from dd_lab.etl.warehouse import Warehouse

    layout = CaseLayout(main_mod._settings.cases_root, "case_api_1")
    wh = Warehouse(layout.warehouse)
    try:
        wh.load_artifacts([
            NormalizedArtifact(
                artifact_id=f"corr-{i}", case_id="case_api_1", app=app,
                source_type=st, source_file=f"{app}/f{i}", source_locator=f"l{i}",
                key="cache.body", value="join https://discord.gg/CACHEDXYZ",
                extracted_ts=datetime.now(UTC),
                classification=ClassificationResult(
                    category=C.CHAT_CONTENT, confidence=0.9, relevant=True,
                    relevance_score=0.9, classifier_version="test"),
            )
            for i, (app, st) in enumerate([("discord", S.CACHE), ("Slack", S.INDEXEDDB)])
        ])
        wh.build_views()
    finally:
        wh.close()

    main_mod._CORRELATION_CACHE.clear()
    first = client.get("/api/cases/case_api_1/correlations").json()
    assert first["entity_count"] >= 1

    calls = []
    real = main_mod.correlate_entities
    monkeypatch.setattr(main_mod, "correlate_entities",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])

    second = client.get("/api/cases/case_api_1/correlations").json()
    assert second == first
    assert calls == [], "a second identical request must not recompute"

    # a warehouse change must invalidate it
    wh = Warehouse(layout.warehouse)
    try:
        wh.load_artifacts([NormalizedArtifact(
            artifact_id="corr-new", case_id="case_api_1", app="discord",
            source_type=S.MEMORY_STRING, source_file="m", source_locator="l",
            key="mem.string", value="join https://discord.gg/CACHEDXYZ",
            extracted_ts=datetime.now(UTC),
            classification=ClassificationResult(
                category=C.CHAT_CONTENT, confidence=0.9, relevant=True,
                relevance_score=0.9, classifier_version="test"))])
        wh.build_views()
    finally:
        wh.close()

    client.get("/api/cases/case_api_1/correlations")
    assert calls == [1], "a changed warehouse must recompute"
