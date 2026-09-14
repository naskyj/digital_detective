"""Findings — the agent's conclusions, as first-class, citable, reviewable records.

Before this module, a "finding" was just prose inside the agent's final text
output or hand-assembled report HTML — nothing stopped a claim from reaching
the report without ever being checked against the evidence it named. A
`Finding` is the unit that makes the rest of the forensic-soundness pipeline
possible:

    write_finding   -> every citation is verified against the warehouse and its
                        value/hash is SNAPSHOTTED at write time (drift after the
                        fact becomes detectable, not silently followed)
    judge_findings  -> an independent model reviews each verified finding
                        (agent/judge.py)
    human review    -> an investigator approves/rejects before a report is final

A finding that fails citation verification is marked REJECTED_UNVERIFIED
immediately and never reaches the judge or the report.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class FindingStatus(StrEnum):
    REJECTED_UNVERIFIED = "rejected_unverified"  # a cited artifact_id didn't check out
    VERIFIED = "verified"  # citations check out, not yet judged
    JUDGE_ACCEPTED = "judge_accepted"
    JUDGE_FLAGGED = "judge_flagged"  # judge has concerns; still needs a human look
    JUDGE_REJECTED = "judge_rejected"
    HUMAN_APPROVED = "human_approved"
    HUMAN_REJECTED = "human_rejected"


# statuses under which a finding may appear in a report if the case wants a
# fully-reviewed report; the report itself always shows the status either way.
FINAL_STATUSES = (FindingStatus.HUMAN_APPROVED, FindingStatus.HUMAN_REJECTED)


class Citation(BaseModel):
    artifact_id: str
    ok: bool
    reason: str
    app: str | None = None
    source_file: str | None = None
    source_locator: str | None = None
    quote: str | None = None  # the specific claimed substring, if any
    value_sha256: str | None = None  # snapshotted at write time, for drift detection


class JudgeVerdict(BaseModel):
    model: str
    verdict: str  # "accept" | "flag" | "reject"
    reasoning: str
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))


class HumanReview(BaseModel):
    reviewer: str
    decision: str  # "approve" | "reject"
    note: str = ""
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Finding(BaseModel):
    finding_id: str
    case_id: str
    claim: str
    entity_type: str | None = None
    entity_value: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    status: FindingStatus = FindingStatus.VERIFIED
    created_by: str = "agent"
    created_ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    judge_verdict: JudgeVerdict | None = None
    human_review: HumanReview | None = None


class FindingsStore:
    """One JSON file per case (`logs/findings.json`), like PipelineState —
    small enough that a full-file rewrite on every change is simpler and
    just as safe as a database for this volume of records."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._findings: dict[str, Finding] = {}
        if self.path.is_file():
            for raw in json.loads(self.path.read_text("utf-8")):
                f = Finding.model_validate(raw)
                self._findings[f.finding_id] = f

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [f.model_dump(mode="json") for f in self._findings.values()]
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def add(self, finding: Finding) -> Finding:
        self._findings[finding.finding_id] = finding
        self._save()
        return finding

    def get(self, finding_id: str) -> Finding | None:
        return self._findings.get(finding_id)

    def list(self, status: FindingStatus | None = None) -> list[Finding]:
        vals = sorted(self._findings.values(), key=lambda f: f.created_ts)
        return [f for f in vals if status is None or f.status == status]

    def set_judge_verdict(self, finding_id: str, verdict: JudgeVerdict) -> Finding:
        f = self._findings[finding_id]
        f.judge_verdict = verdict
        f.status = {
            "accept": FindingStatus.JUDGE_ACCEPTED,
            "flag": FindingStatus.JUDGE_FLAGGED,
            "reject": FindingStatus.JUDGE_REJECTED,
        }[verdict.verdict]
        self._save()
        return f

    def set_human_review(self, finding_id: str, review: HumanReview) -> Finding:
        f = self._findings[finding_id]
        f.human_review = review
        f.status = (
            FindingStatus.HUMAN_APPROVED if review.decision == "approve" else FindingStatus.HUMAN_REJECTED
        )
        self._save()
        return f
