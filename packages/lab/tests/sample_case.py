"""Build a signed sample case from a live Chromium/Electron profile.

Used by the lab tests to exercise the full ingest path against real evidence
(VS Code / Postman on the dev machine). Also a preview of what the collector
does: plan the copies, execute them, hash + sign a manifest.
"""

from __future__ import annotations

import platform
from datetime import UTC, datetime
from pathlib import Path

from dd_core.casedir import CaseLayout
from dd_core.custody import CustodyLog
from dd_core.enums import CustodyEvent, SourceType
from dd_core.evidence import execute_copy, plan_profile_copy
from dd_core.manifest import AcquisitionHost, Manifest, build_entry
from dd_core.schema import TargetApp
from dd_core.signing import generate_keypair

_SOURCE_TYPE_BY_PREFIX = {
    "local_storage": SourceType.LOCAL_STORAGE,
    "session_storage": SourceType.SESSION_STORAGE,
    "indexeddb": SourceType.INDEXEDDB,
    "cache": SourceType.CACHE,
    "cookies": SourceType.COOKIES,
    "preferences.json": SourceType.PREFERENCES,
    "local_state.json": SourceType.LOCAL_STATE,
}


def _source_type_for(rel_path: str) -> SourceType:
    tail = rel_path.split("/", 1)[1] if "/" in rel_path else rel_path
    for prefix, st in _SOURCE_TYPE_BY_PREFIX.items():
        if tail.startswith(prefix):
            return st
    return SourceType.PREFERENCES


def build_sample_case(
    case_root: Path,
    case_id: str,
    app: str,
    profile_dir: Path,
    include: frozenset[str] | None = None,
    memory_dumps: dict[str, bytes] | None = None,
) -> CaseLayout:
    layout = CaseLayout(case_root, case_id)
    layout.create_evidence_skeleton()
    app_evidence = layout.evidence / app
    app_evidence.mkdir(parents=True, exist_ok=True)

    started = datetime.now(UTC)
    specs = plan_profile_copy(profile_dir, include)
    if not specs:
        raise RuntimeError(f"no recognisable Chromium profile data in {profile_dir}")
    written = execute_copy(specs, app_evidence)

    for fname, content in (memory_dumps or {}).items():
        mem_path = app_evidence / "memory" / fname
        mem_path.parent.mkdir(parents=True, exist_ok=True)
        mem_path.write_bytes(content)
        written.append(mem_path)

    now = datetime.now(UTC)
    entries = [
        build_entry(layout.evidence, f, _source_type_for(f.relative_to(layout.evidence).as_posix()), now)
        for f in written
        if f.is_file()
    ]

    priv, pub = generate_keypair()
    manifest = Manifest(
        case_id=case_id,
        collector_version="sample-0.1.0",
        operator="test-harness",
        host=AcquisitionHost(
            hostname=platform.node() or "dev",
            os=platform.system(),
            os_version=platform.release(),
            username="dev",
            timezone="UTC",
        ),
        started_ts=started,
        completed_ts=now,
        target_apps=[TargetApp(name=app, user_data_dir=str(profile_dir))],
        entries=entries,
        public_key_pem=pub.decode(),
    )
    manifest.write(layout.dir, priv)

    custody = CustodyLog(layout.dir)
    custody.append(CustodyEvent.ACQUISITION_START, "test-harness", f"sample acquisition of {app}")
    custody.append(
        CustodyEvent.ARTIFACT_ACQUIRED, "test-harness",
        f"{len(entries)} files copied from {profile_dir}",
    )
    custody.append(CustodyEvent.ACQUISITION_COMPLETE, "test-harness", "sample acquisition complete")
    custody.append(CustodyEvent.MANIFEST_SIGNED, "test-harness", "manifest signed")
    return layout
