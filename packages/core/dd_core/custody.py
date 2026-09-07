"""Hash-chained chain-of-custody log.

Every custody-relevant action (acquisition, transfer, import, integrity check,
analysis step, report) appends one JSON line to `chain_of_custody.jsonl`. Each
record carries the hash of the previous record, so removing or altering any line
breaks the chain and `verify_chain` will point at the first bad link.

This is deliberately simple and append-only; it is not a substitute for storing
the evidence on write-once media, but it makes tampering with the log evident.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from dd_core.enums import CustodyEvent
from dd_core.hashing import sha256_canonical

LOG_NAME = "chain_of_custody.jsonl"
GENESIS = "0" * 64


class CustodyRecord(BaseModel):
    seq: int
    ts: datetime
    event: CustodyEvent
    actor: str  # operator name or tool identifier
    detail: str
    data: dict = {}
    prev_hash: str
    entry_hash: str = ""

    def compute_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"entry_hash"})
        return sha256_canonical(payload)


class CustodyLog:
    def __init__(self, case_dir: str | Path):
        self.path = Path(case_dir) / LOG_NAME

    def _records(self) -> list[CustodyRecord]:
        if not self.path.is_file():
            return []
        out = []
        for line in self.path.read_text("utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(CustodyRecord.model_validate_json(line))
        return out

    def append(
        self,
        event: CustodyEvent,
        actor: str,
        detail: str,
        data: dict | None = None,
        ts: datetime | None = None,
    ) -> CustodyRecord:
        records = self._records()
        prev_hash = records[-1].entry_hash if records else GENESIS
        rec = CustodyRecord(
            seq=len(records),
            ts=ts or datetime.now(UTC),
            event=event,
            actor=actor,
            detail=detail,
            data=data or {},
            prev_hash=prev_hash,
        )
        rec.entry_hash = rec.compute_hash()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec.model_dump(mode="json"), sort_keys=True) + "\n")
        return rec

    def verify_chain(self) -> tuple[bool, int | None]:
        """Return (ok, first_bad_seq)."""
        prev = GENESIS
        for i, rec in enumerate(self._records()):
            if rec.prev_hash != prev or rec.entry_hash != rec.compute_hash() or rec.seq != i:
                return False, i
            prev = rec.entry_hash
        return True, None

    def entries(self) -> list[CustodyRecord]:
        return self._records()
