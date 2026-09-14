"""Adversarial judge — an independent model review of every verified finding,
before a human ever sees it.

Deliberately separate from the primary investigation agent:
- its own `AgentConfig` (`DD_JUDGE_LLM_*` env vars, falling back to the
  primary's `DD_LLM_*`, then Ollama) so it CAN be pointed at a different
  provider/model than the one that produced the findings — same-model-judging-
  itself shares blind spots and hallucination modes, a different model is more
  likely to catch what the first missed.
- no tools and no evidence access: its only input is the finding's claim plus
  the citations `write_finding` already independently re-verified (with their
  snapshotted quoted text) — it cannot go fetch new "evidence" of its own, it
  can only judge whether the claim is supported by what's already been proven.

The verdict is never applied silently: `judge_findings` writes it onto the
Finding (visible in the report and to a human reviewer) but a "reject" does
not delete anything, and only a human decision (`agent/tools.py` human review,
not yet wired to a UI) can finalize a finding either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from dd_core.casedir import CaseLayout
from dd_lab.agent.audit import AuditLog
from dd_lab.agent.config import AgentConfig, build_model
from dd_lab.agent.findings import Finding, FindingsStore, FindingStatus, JudgeVerdict

SYSTEM_PROMPT = """\
You are an independent, adversarial reviewer of forensic findings produced by
another AI system investigating Electron-application evidence (Discord,
Roblox, and similar apps). You do not trust that system. Your only job is to
catch problems before a human investigator sees this finding.

You are given a claim and the citations that were already independently
verified against the evidence warehouse (their exact stored text and source).
Check:
1. Does the cited/quoted text actually support the claim, or does the claim
   say more than the evidence shows?
2. Is this an evidentiary observation ("artifact X shows...") or does it cross
   into a legal/criminal conclusion ("this proves...", "this is grooming",
   "this confirms abuse")? Any legal conclusion must be flagged or rejected —
   only a human may draw one, never this pipeline.
3. Is there an obvious innocent alternative explanation the claim ignores?
4. Are the citations sufficient to support the claim at all, or is it a leap
   from thin evidence?

Return "accept" only if the claim is fully supported and appropriately worded.
Return "flag" if plausible but a human should look (ambiguous, borderline
wording, thin-but-not-absent support). Return "reject" if unsupported or if it
states a legal conclusion outright. Be concise. Be skeptical. When genuinely
unsure, flag rather than accept.
"""


class Verdict(BaseModel):
    verdict: str = Field(pattern="^(accept|flag|reject)$")
    reasoning: str


@dataclass
class JudgeResult:
    finding_id: str
    verdict: str
    reasoning: str


def judge_config() -> AgentConfig:
    return AgentConfig.from_env(prefix="DD_JUDGE_LLM")


def build_judge_agent(cfg: AgentConfig, model=None) -> Agent[None, Verdict]:
    return Agent(model or build_model(cfg), output_type=Verdict, system_prompt=SYSTEM_PROMPT, retries=2)


def _finding_prompt(f: Finding) -> str:
    lines = [f"CLAIM: {f.claim}"]
    if f.entity_type:
        lines.append(f"entity: {f.entity_type} = {f.entity_value}")
    lines.append("CITATIONS (already independently re-verified against the evidence warehouse):")
    for c in f.citations:
        lines.append(
            f"  - app={c.app} file={c.source_file} locator={c.source_locator}\n"
            f"    stored value: {(c.quote or '')!r}"
        )
    return "\n".join(lines)


def judge_findings(
    cases_root: str | Path, case_id: str, *, cfg: AgentConfig | None = None, model=None, actor: str = "judge",
) -> list[JudgeResult]:
    """Judge every VERIFIED (not yet judged) finding for a case. Findings that
    were already REJECTED_UNVERIFIED never reach here — they failed citation
    checking before the judge is even involved."""
    layout = CaseLayout(cases_root, case_id)
    store = FindingsStore(layout.findings_log)
    audit = AuditLog(layout.audit_log)
    cfg = cfg or judge_config()
    agent = build_judge_agent(cfg, model)

    results: list[JudgeResult] = []
    for f in store.list(FindingStatus.VERIFIED):
        run = agent.run_sync(_finding_prompt(f))
        v = run.output
        store.set_judge_verdict(f.finding_id, JudgeVerdict(
            model=f"{cfg.provider}:{cfg.model}", verdict=v.verdict, reasoning=v.reasoning,
        ))
        audit.append(
            "judge_verdict", f.finding_id, v.reasoning[:500],
            verdict=v.verdict, model=f"{cfg.provider}:{cfg.model}",
        )
        results.append(JudgeResult(f.finding_id, v.verdict, v.reasoning))
    return results
