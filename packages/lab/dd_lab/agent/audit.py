"""Agent audit log — an append-only record of everything the agent did.

Distinct from the chain-of-custody log (which is about evidence integrity): this
is investigator work-product showing how the analysis was produced — every tool
call, its arguments and result summary, every model message, and the model /
prompt used. It makes an agent-driven investigation reproducible and reviewable.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class AuditRecord(BaseModel):
    seq: int
    ts: datetime
    kind: str  # tool_call | tool_result | model_text | decision | error | run_start | run_end
    name: str = ""
    detail: str = ""
    data: dict[str, Any] = {}


class AuditLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._seq = self._last_seq()

    def _last_seq(self) -> int:
        if not self.path.is_file():
            return -1
        last = -1
        for line in self.path.read_text("utf-8").splitlines():
            if line.strip():
                last = json.loads(line).get("seq", last)
        return last

    def append(self, kind: str, name: str = "", detail: str = "", **data: Any) -> AuditRecord:
        self._seq += 1
        rec = AuditRecord(
            seq=self._seq, ts=datetime.now(UTC), kind=kind, name=name,
            detail=detail[:2000], data=_trim(data),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(rec.model_dump_json() + "\n")
        return rec

    def records(self) -> list[AuditRecord]:
        if not self.path.is_file():
            return []
        return [
            AuditRecord.model_validate_json(line)
            for line in self.path.read_text("utf-8").splitlines()
            if line.strip()
        ]


def _trim(data: dict[str, Any], limit: int = 4000) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in data.items():
        s = v if isinstance(v, (int, float, bool, type(None))) else str(v)
        if isinstance(s, str) and len(s) > limit:
            s = s[:limit] + f"…[+{len(s) - limit}]"
        out[k] = s
    return out
