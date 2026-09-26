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
    #: The finding's CITATIONS still verify, but its PREMISE no longer holds:
    #: the entity it is about is not one the current extractor recognises.
    #: Findings are an append-only record and are never deleted - a wrong
    #: conclusion that was drawn is itself part of the case history - so a
    #: superseded one keeps its text, its citations and its timestamps and is
    #: simply no longer presented as a live finding.
    #:
    #: Why this exists: an email pattern once matched retina-image filenames,
    #: and agent runs on 2026-09-17 and 09-19 wrote 199 findings asserting
    #: things like "the email 'wechat@3x.png' appears in N artifacts". The
    #: pattern was fixed on 09-23, but findings.json is append-only, so every
    #: report generated afterwards still carried all 199.
    SUPERSEDED = "superseded"


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

    def supersede_stale(self, *, dry_run: bool = False) -> list[Finding]:
        """Mark findings whose entity the current extractor no longer
        recognises, and return them.

        Checks the PREMISE, which nothing else does: `write_finding` verifies
        that each cited artifact exists and contains the quoted text, and both
        remain true here - the artifact really does contain `wechat@3x.png`.
        What stopped being true is that the string is an email address. So the
        test is whether re-running the entity extractor over the finding's own
        entity_value still yields that entity type.

        Only entity types the extractor actually produces are checked. An
        LLM-authored finding about a `verification_flow` or a `session_token`
        is outside the extractor's vocabulary and is left alone - absence of a
        match there means nothing.

        Never deletes. A conclusion that was drawn and later invalidated is
        part of the case history and a report that quietly loses it is worse
        than one that shows it marked.
        """
        from dd_lab.investigate.entities import KNOWN_ENTITY_TYPES, extract_entities

        stale: list[Finding] = []
        for f in self._findings.values():
            if f.status is FindingStatus.SUPERSEDED:
                continue
            if not f.entity_type or not f.entity_value:
                continue
            if f.entity_type not in KNOWN_ENTITY_TYPES:
                continue
            # Context matters: several extractors are gated on the app name
            # (a Slack team id shape is too generic to trust outside Slack's
            # own evidence). Asking with an empty app answers the wrong
            # question - it says "is this an entity with no context", not
            # "could this value ever be this entity type". Caught by a dry
            # run that would have superseded the case's top-ranked
            # correlation, slack_team_id T0C1J4SDLG2.
            app_hint = f.entity_type.split("_", 1)[0]
            hits = extract_entities(app_hint, "", f.entity_value)
            if not any(h.entity_type == f.entity_type for h in hits):
                stale.append(f)
        if not dry_run:
            for f in stale:
                f.status = FindingStatus.SUPERSEDED
            if stale:
                self._save()
        return stale

    def set_human_review(self, finding_id: str, review: HumanReview) -> Finding:
        f = self._findings[finding_id]
        f.human_review = review
        f.status = (
            FindingStatus.HUMAN_APPROVED if review.decision == "approve" else FindingStatus.HUMAN_REJECTED
        )
        self._save()
        return f
