"""Two ways to run an investigation over the same tools:

- `run_pipeline`   : deterministic fixed sequence (no LLM). Always available,
                     used for tests and for users without a model configured.
- `run_agent`      : the LLM agent decides the sequence and does the reasoning.

Both write to the case audit log and update the resumable pipeline state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from dd_lab.agent import tools as T
from dd_lab.agent.config import AgentConfig
from dd_lab.agent.tools import InvestigationDeps
from dd_lab.case import CaseStatus

log = logging.getLogger("dd_lab.agent")


@dataclass
class RunOutcome:
    case_id: str
    ok: bool
    report_path: str | None
    summary: str
    steps: int = 0


def _set_status(deps: InvestigationDeps, status: CaseStatus, message: str) -> None:
    deps.meta.status = status
    deps.meta.last_message = message
    deps.meta.save(deps.layout)


def _judge_findings_best_effort(deps: InvestigationDeps) -> None:
    """The adversarial judge needs a reachable LLM (its own DD_JUDGE_LLM_*
    config, falling back to DD_LLM_*/Ollama) — never let its absence or
    failure abort the run; a case with no model configured still gets
    findings, just unjudged (visibly so in the report)."""
    try:
        from dd_lab.agent.judge import judge_findings

        judged = judge_findings(deps.cases_root, deps.case_id)
        deps.audit.append("decision", "judge_findings", f"{len(judged)} finding(s) judged")
    except Exception as e:  # noqa: BLE001 - judging is best-effort, never fatal
        deps.audit.append("decision", "judge_findings", f"skipped: {e}")


def run_pipeline(cases_root: str | Path, case_id: str, *, classifier=None) -> RunOutcome:
    deps = InvestigationDeps(cases_root=Path(cases_root), case_id=case_id, classifier=classifier)
    deps.audit.append("run_start", "deterministic", "fixed-sequence pipeline")
    _set_status(deps, CaseStatus.RUNNING, "processing")

    try:
        integrity = T.verify_integrity(deps)
        deps.audit.append("decision", "verify_integrity", f"ok={integrity['ok']}")
        if not integrity["ok"]:
            deps.audit.append("error", "verify_integrity", "integrity check failed; aborting")
            _set_status(deps, CaseStatus.ERROR, "integrity check failed")
            return RunOutcome(case_id, False, None, "Evidence integrity check failed.")

        ingest = T.run_ingest(deps)
        deps.audit.append(
            "decision", "run_ingest",
            f"{ingest.get('artifact_count', 0)} artifacts; categories={ingest.get('by_category')}",
        )
        if not ingest["ok"]:
            _set_status(deps, CaseStatus.ERROR, ingest.get("error", "ingest failed"))
            return RunOutcome(case_id, False, None, ingest.get("error", "ingest failed"))

        mem = T.run_memory_analysis(deps)
        deps.audit.append("decision", "run_memory_analysis", mem.get("reason", "done"))

        for kw in deps.meta.keywords:
            hit = T.search_keyword(deps, kw)
            deps.audit.append("decision", "search_keyword", f"{kw!r}: {hit['hits']} hits")

        correlated = T.correlate_and_write_findings(deps)
        deps.audit.append(
            "decision", "correlate_and_write_findings", f"{correlated['finding_count']} findings written"
        )
        _judge_findings_best_effort(deps)

        report = T.generate_report(deps)
        deps.audit.append("decision", "generate_report", report["report_path"])

        deps.state.finished = True
        deps.state.save()
        deps.audit.append("run_end", "deterministic", "complete")
        _set_status(deps, CaseStatus.COMPLETED, "analysis complete")
        summary = (
            f"{ingest['artifact_count']:,} artifacts across {list(ingest['per_app'])}; "
            f"categories: {ingest['by_category']}. Report: {report['report_path']}"
        )
        return RunOutcome(case_id, True, report["report_path"], summary)
    except Exception as e:  # noqa: BLE001
        log.exception("pipeline failed")
        deps.audit.append("error", "pipeline", str(e))
        _set_status(deps, CaseStatus.ERROR, str(e))
        return RunOutcome(case_id, False, None, f"pipeline error: {e}")


def run_agent(
    cases_root: str | Path,
    case_id: str,
    cfg: AgentConfig | None = None,
    instruction: str | None = None,
    *,
    classifier=None,
    model=None,
) -> RunOutcome:
    from dd_lab.agent.agent import build_agent

    cfg = cfg or AgentConfig.from_env()
    deps = InvestigationDeps(cases_root=Path(cases_root), case_id=case_id, classifier=classifier)
    deps.audit.append("run_start", "agent", f"provider={cfg.provider} model={cfg.model}")
    _set_status(deps, CaseStatus.RUNNING, f"agent ({cfg.model})")

    prompt = instruction or (
        f"Investigate case {case_id!r} end to end and produce the forensic report. "
        f"Investigation keywords: {deps.meta.keywords or 'none provided'}."
    )
    agent = build_agent(cfg, model)
    try:
        result = agent.run_sync(prompt, deps=deps)
        deps.audit.append("model_text", "final", str(result.output)[:2000])

        # Independent judge review runs unconditionally here — never left to
        # the primary agent's own discretion — then the report is (re)built so
        # it reflects post-judge statuses rather than whatever was true when
        # the agent itself called generate_report (if it did).
        _judge_findings_best_effort(deps)
        report = T.generate_report(deps)
        deps.audit.append(
            "decision", "generate_report", f"final report (post-judge): {report['report_path']}"
        )

        deps.state.finished = deps.state.report_path is not None
        deps.state.save()
        deps.audit.append("run_end", "agent", "complete")
        status = CaseStatus.COMPLETED if deps.state.report_path else CaseStatus.STOPPED
        _set_status(deps, status, "agent run complete")
        return RunOutcome(
            case_id, True, deps.state.report_path, str(result.output),
            steps=len(result.all_messages()),
        )
    except Exception as e:  # noqa: BLE001
        log.exception("agent run failed")
        deps.audit.append("error", "agent", str(e))
        _set_status(deps, CaseStatus.ERROR, str(e))
        return RunOutcome(case_id, False, deps.state.report_path, f"agent error: {e}")
