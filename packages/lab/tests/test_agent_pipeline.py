"""Deterministic pipeline runner + tool behaviour (no LLM)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dd_lab.agent import run_pipeline
from dd_lab.agent.audit import AuditLog
from dd_lab.agent.tools import InvestigationDeps, query_warehouse
from dd_lab.case import CaseMeta, CaseStatus

from tests.sample_case import build_sample_case

_CANDIDATES = {
    "vscode": Path.home() / "Library/Application Support/Code",
    "postman": Path.home() / "Library/Application Support/Postman",
}
_FAST = frozenset({"local_storage", "session_storage", "preferences", "cookies", "local_state"})


def _available():
    for app, path in _CANDIDATES.items():
        if (path / "Local Storage" / "leveldb").is_dir() or (path / "Preferences").is_file():
            return app, path
    return None


pytestmark = pytest.mark.skipif(_available() is None, reason="no Electron profile available")


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    app, profile = _available()
    root = tmp_path_factory.mktemp("cases")
    cid = "case_agent_001"
    build_sample_case(root, cid, app, profile, include=_FAST)
    meta = CaseMeta(case_id=cid, name="Test Investigation", keywords=["token", "user"])
    from dd_core.casedir import CaseLayout

    meta.save(CaseLayout(root, cid))
    return root, cid


def test_deterministic_pipeline_completes(case):
    root, cid = case
    progress_log: list[str] = []
    outcome = run_pipeline(root, cid, progress=progress_log.append, enable_judge=False)
    assert outcome.ok
    assert outcome.report_path and Path(outcome.report_path).is_file()

    # real-time progress, not just a message at the very start and very end -
    # this is what the job's live log in the UI streams as the run happens.
    assert len(progress_log) >= 6
    joined = " ".join(progress_log)
    assert "starting" in joined.lower()
    assert "integrity" in joined.lower()
    assert "Analysis complete" in progress_log[-1]
    assert any("Judge review disabled" in m for m in progress_log)  # enable_judge=False, above

    html = Path(outcome.report_path).read_text("utf-8")
    assert "Forensic Analysis Report" in html
    assert "Chain of Custody" in html
    assert "Investigation Keyword Hits" in html

    from dd_core.casedir import CaseLayout

    layout = CaseLayout(root, cid)
    meta = CaseMeta.load(layout)
    assert meta.status == CaseStatus.COMPLETED

    kinds = {r.kind for r in AuditLog(layout.audit_log).records()}
    assert {"run_start", "decision", "run_end"} <= kinds


def test_pipeline_is_resumable_state(case):
    root, cid = case
    from dd_core.casedir import CaseLayout
    from dd_lab.agent.state import PipelineState

    st = PipelineState.load(CaseLayout(root, cid).dir / "logs" / "pipeline_state.json", cid)
    assert st.integrity_verified
    assert st.classified
    assert st.finished


def test_query_warehouse_rejects_writes(case):
    root, cid = case
    deps = InvestigationDeps(cases_root=root, case_id=cid)
    assert query_warehouse(deps, "DELETE FROM artifacts")["ok"] is False
    assert query_warehouse(deps, "DROP TABLE artifacts")["ok"] is False
    good = query_warehouse(deps, "SELECT category, count(*) c FROM artifacts GROUP BY 1")
    assert good["ok"] is True
    assert good["row_count"] >= 1


def test_pipeline_runs_memory_analysis_exactly_once(tmp_path_factory, monkeypatch):
    """Regression: run_pipeline() used to analyse memory twice - once inside
    ingest_case() (its analyze_memory=True default), once more via the
    explicit run_memory_analysis step - re-scanning the whole dump and
    re-running every Volatility plugin a second time for nothing (the
    warehouse's ON CONFLICT DO NOTHING silently dropped the duplicate rows).
    Also checks the final summary's totals still include the memory pass,
    since it no longer arrives folded into ingest_case()'s own count."""
    import dd_lab.memory as mem_pkg

    app, profile = _available() or (None, None)
    if app is None:
        pytest.skip("no Electron profile available")

    root = tmp_path_factory.mktemp("cases_mem")
    cid = "case_agent_mem_001"
    blob = b"\x00" * 4096 + b"hey are you free to chat after school tomorrow" + b"\x00" * 64
    build_sample_case(root, cid, app, profile, include=_FAST, memory_dumps={"dump.raw": blob})
    from dd_core.casedir import CaseLayout

    meta = CaseMeta(case_id=cid, name="Test Investigation")
    meta.save(CaseLayout(root, cid))

    calls = []
    real_analyze_memory = mem_pkg.analyze_memory

    def _spy(*a, **kw):
        calls.append((a, kw))
        return real_analyze_memory(*a, **kw)

    monkeypatch.setattr(mem_pkg, "analyze_memory", _spy)

    outcome = run_pipeline(root, cid, enable_judge=False)
    assert outcome.ok
    assert len(calls) == 1, f"analyze_memory called {len(calls)} times, expected exactly once"

    deps = InvestigationDeps(cases_root=root, case_id=cid)
    strings_row = query_warehouse(
        deps, "SELECT count(*) c FROM artifacts WHERE source_type = 'memory_string'"
    )
    assert strings_row["ok"]
    assert strings_row["rows"][0]["c"] > 0  # the memory pass actually loaded rows, not skipped

    # the final summary's total must include ALL memory rows (string-carved,
    # plus Volatility's if `vol` happens to be installed here), not just the
    # disk-parsed ones - ingest_case()'s own count no longer folds them in.
    disk_only = query_warehouse(
        deps, "SELECT count(*) c FROM artifacts WHERE source_type NOT LIKE 'memory%'"
    )["rows"][0]["c"]
    mem_total = query_warehouse(
        deps, "SELECT count(*) c FROM artifacts WHERE source_type LIKE 'memory%'"
    )["rows"][0]["c"]
    assert f"{disk_only + mem_total:,} artifacts" in outcome.summary
