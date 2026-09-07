"""Cached evidence verification.

`verify_case` re-hashes every evidence file. For a 400 MB case that is a second
or two; for one with a full memory image it is much more. Re-doing it on every
page load of the case is wasteful, so the result is cached to
`<case>/logs/verification.json` and only recomputed on demand or when the
evidence set changes (file count / total size).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from dd_core.casedir import CaseLayout
from dd_core.manifest import VerificationResult, verify_case

_CACHE = "logs/verification.json"


def _signature(evidence_dir: Path) -> tuple[int, int]:
    n = size = 0
    for f in evidence_dir.rglob("*"):
        if f.is_file():
            n += 1
            size += f.stat().st_size
    return n, size


def verify_cached(
    case_root: str | Path, case_id: str, *, force: bool = False
) -> tuple[VerificationResult, datetime]:
    layout = CaseLayout(case_root, case_id)
    cache_path = layout.dir / _CACHE
    sig = _signature(layout.evidence) if layout.evidence.is_dir() else (0, 0)

    if not force and cache_path.is_file():
        try:
            blob = json.loads(cache_path.read_text("utf-8"))
            if blob.get("evidence_signature") == list(sig):
                return (
                    VerificationResult.model_validate(blob["result"]),
                    datetime.fromisoformat(blob["verified_ts"]),
                )
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    result = verify_case(layout.dir)
    ts = datetime.now(UTC)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "verified_ts": ts.isoformat(),
                "evidence_signature": list(sig),
                "result": result.model_dump(mode="json"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return result, ts
