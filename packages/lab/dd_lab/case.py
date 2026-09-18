"""Case metadata and a filesystem-backed case registry.

`case.json` in the case directory holds the investigator-supplied context
(name, description, keywords) and the processing status. The registry is just
"every directory under the cases root that has a manifest".
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from dd_core.casedir import CaseLayout


def new_case_id() -> str:
    """A fresh, sortable case_id - the one place this is generated for
    `dd_lab` (imaging/acquire.py reuses this instead of keeping its own
    copy). `dd_collector` intentionally keeps its own independent copy - it
    never depends on `dd_lab`."""
    return f"{datetime.now(UTC):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"


class CaseStatus(StrEnum):
    NEW = "new"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"


class EvidenceSourceMeta(BaseModel):
    """One evidence source added to a case - a case can have several (e.g.
    a suspect's laptop image AND a victim's phone extraction, correlated in
    one investigation). Purely a human-facing registry for the UI ("this
    case has N sources", pre-filling the Add-source form) - unsigned, lives
    in case.json alongside the rest of CaseMeta, not in the signed
    manifest.json (which stays the sole evidentiary source of truth for
    what evidence actually exists; this is just labels/bookkeeping on top).
    `label` is the same string `acquire_from_image`/`acquire_from_folder`'s
    `source_label` param used to prefix this source's app names
    ("<label> / <app name>") - kept here so the UI can show it without
    having to reverse-parse it back out of an app name."""

    label: str
    kind: Literal["image", "folder"]
    path: str
    operator: str = ""
    added_ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    apps: list[str] = Field(default_factory=list)


class CaseMeta(BaseModel):
    case_id: str
    name: str
    description: str = ""
    keywords: list[str] = Field(default_factory=list)
    status: CaseStatus = CaseStatus.NEW
    created_ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_message: str = ""
    sources: list[EvidenceSourceMeta] = Field(default_factory=list)

    def save(self, layout: CaseLayout) -> None:
        self.updated_ts = datetime.now(UTC)
        layout.case_meta.parent.mkdir(parents=True, exist_ok=True)
        layout.case_meta.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, layout: CaseLayout) -> CaseMeta:
        return cls.model_validate_json(layout.case_meta.read_text("utf-8"))

    @classmethod
    def load_or_none(cls, layout: CaseLayout) -> CaseMeta | None:
        return cls.load(layout) if layout.case_meta.is_file() else None


def parse_keywords(raw: str) -> list[str]:
    parts: list[str] = []
    for chunk in raw.replace("\r", "\n").split("\n"):
        parts.extend(p.strip() for p in chunk.split(","))
    return [p for p in parts if p]


def create_case(
    cases_root: str | Path, case_id: str | None, name: str,
    description: str = "", keywords: list[str] | None = None,
) -> CaseMeta:
    """Create a case's metadata - `case_id` generated with `new_case_id()`
    if not given. No evidence is required to exist yet: a case is now a
    valid, visible thing (`CaseStatus.NEW`) the moment it has a name, and
    evidence sources are added to it afterwards (possibly much afterwards -
    "ongoing", not a one-shot wizard) via the imaging acquisition functions
    targeting this same case_id. This used to require a signed manifest to
    already exist (metadata could only be ATTACHED to evidence someone
    already collected) - dropped so case creation and evidence import don't
    have to happen in that order any more.
    """
    case_id = case_id or new_case_id()
    layout = CaseLayout(cases_root, case_id)
    meta = CaseMeta(
        case_id=case_id, name=name, description=description, keywords=keywords or []
    )
    meta.save(layout)
    return meta


def list_cases(cases_root: str | Path) -> list[CaseMeta]:
    """Every case folder under the root.

    A folder is listed if it has a signed manifest (a real case) OR a `case.json`.
    A folder that looks like an aborted acquisition — it has `evidence/` or a
    custody log but no `manifest.json` — is also listed, flagged ERROR, so the
    investigator can see it exists and why it can't be analysed.
    """
    root = Path(cases_root)
    if not root.is_dir():
        return []
    out: list[CaseMeta] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        layout = CaseLayout(root, d.name)
        meta = CaseMeta.load_or_none(layout)
        if meta is not None:
            out.append(meta)
            continue
        if layout.manifest.is_file():
            out.append(CaseMeta(case_id=d.name, name=d.name))
            continue
        if (layout.evidence.is_dir() or layout.custody_log.is_file()):
            out.append(CaseMeta(
                case_id=d.name, name=d.name, status=CaseStatus.ERROR,
                last_message="incomplete acquisition — no signed manifest.json; "
                             "re-run the collector for this case",
            ))
    return out
