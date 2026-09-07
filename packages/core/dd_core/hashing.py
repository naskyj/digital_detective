"""SHA-256 helpers used for evidence integrity and stable identifiers.

Everything here is deterministic. These functions are the trust anchor for the
chain of custody, so they must never depend on wall-clock time, iteration order,
or locale.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_CHUNK = 1024 * 1024


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Stream a file through SHA-256 without loading it into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    """Deterministic JSON serialization: sorted keys, no insignificant whitespace.

    Used so that a manifest or custody record hashes to the same value on any
    machine regardless of dict ordering.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_canonical(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def stable_id(*parts: str) -> str:
    """A short, stable identifier derived from its inputs (first 16 hex chars)."""
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]
