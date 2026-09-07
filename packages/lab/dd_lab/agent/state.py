"""Resumable pipeline state so a crashed or interrupted run can continue."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field


class PipelineState(BaseModel):
    case_id: str
    integrity_verified: bool = False
    ingested_apps: list[str] = Field(default_factory=list)
    classified: bool = False
    memory_analyzed: bool = False
    report_path: str | None = None
    finished: bool = False
    updated_ts: datetime = Field(default_factory=lambda: datetime.now(UTC))

    _path: Path | None = None

    @classmethod
    def load(cls, path: str | Path, case_id: str) -> PipelineState:
        path = Path(path)
        state = (
            cls.model_validate_json(path.read_text("utf-8"))
            if path.is_file()
            else cls(case_id=case_id)
        )
        state._path = path
        return state

    def save(self) -> None:
        if self._path is None:
            return
        self.updated_ts = datetime.now(UTC)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    def summary(self) -> str:
        done = []
        if self.integrity_verified:
            done.append("integrity verified")
        if self.ingested_apps:
            done.append(f"ingested: {', '.join(self.ingested_apps)}")
        if self.classified:
            done.append("classified")
        if self.memory_analyzed:
            done.append("memory analyzed")
        if self.report_path:
            done.append("report built")
        if self.finished:
            done.append("FINISHED")
        return "; ".join(done) if done else "nothing done yet"
