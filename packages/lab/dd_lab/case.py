"""Case metadata and a filesystem-backed case registry.

`case.json` in the case directory holds the investigator-supplied context
(name, description, keywords) and the processing status. The registry is just
"every directory under the cases root that has a manifest".
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from dd_core.casedir import CaseLayout


class CaseStatus(StrEnum):
    NEW = "new"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"


class CaseMeta(BaseModel):
    case_id: str
    name: str
    description: str = ""
    keywords: list[str] = Field(default_factory=list)
    status: CaseStatus = CaseStatus.NEW
    created_ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_message: str = ""

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
    cases_root: str | Path, case_id: str, name: str, description: str = "", keywords: list[str] | None = None
) -> CaseMeta:
    layout = CaseLayout(cases_root, case_id)
    if not layout.manifest.is_file():
        raise FileNotFoundError(
            f"no signed manifest at {layout.dir} — import the collected case first"
        )
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
