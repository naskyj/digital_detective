"""Source parsers: raw evidence -> NormalizedArtifact rows.

`parse_app` runs every applicable parser for one app's evidence and yields a
single stream of artifacts. Parsers self-skip when their source is absent, so
this works for any Electron app regardless of which stores it uses.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

from dd_core.schema import NormalizedArtifact
from dd_lab.parsers import appdata, cache, cookies, indexeddb, localstorage, prefs, sessionstorage
from dd_lab.parsers.base import ParseContext

log = logging.getLogger("dd_lab.parsers")

# ordered: cheap/structured first, memory-heavy last. appdata is last of all -
# it only picks up what every other parser here left unclaimed (see its own
# module docstring), so it needs them to have run first to know what "unclaimed"
# means.
PARSERS = [
    ("preferences", prefs.parse),
    ("local_storage", localstorage.parse),
    ("session_storage", sessionstorage.parse),
    ("cookies", cookies.parse),
    ("cache", cache.parse),
    ("indexeddb", indexeddb.parse),
    ("appdata", appdata.parse),
]


def parse_app(case_id: str, evidence_root: str | Path, app: str) -> Iterator[NormalizedArtifact]:
    ctx = ParseContext(case_id=case_id, app=app, evidence_root=Path(evidence_root))
    if not ctx.app_root.is_dir():
        log.warning("no evidence directory for app %r under %s", app, evidence_root)
        return
    for name, fn in PARSERS:
        count = 0
        try:
            for artifact in fn(ctx):
                count += 1
                yield artifact
        except Exception:
            log.exception("parser %r crashed for app %r", name, app)
        log.info("parser %s: %d artifacts from %s", name, count, app)


def discover_apps(evidence_root: str | Path) -> list[str]:
    root = Path(evidence_root)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


__all__ = ["ParseContext", "discover_apps", "parse_app"]
