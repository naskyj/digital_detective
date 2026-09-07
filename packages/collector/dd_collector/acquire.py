"""Acquisition orchestrator: copy → hash → sign → chain-of-custody.

Read-only with respect to the source machine. Produces a self-contained,
signed case directory under the chosen output location (an external drive in
real use). Never aborts on a missing memory tool.
"""

from __future__ import annotations

import logging
import platform
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from dd_collector.discover import ElectronApp
from dd_collector.memory import capture_full_memory, capture_process_memory
from dd_core.casedir import CaseLayout
from dd_core.custody import CustodyLog
from dd_core.enums import CustodyEvent, SourceType
from dd_core.evidence import (
    MEMORY,
    execute_copy,
    plan_profile_copy,
    source_type_for_path,
)
from dd_core.manifest import AcquisitionHost, Manifest, build_entry
from dd_core.schema import ProcessRef, TargetApp
from dd_core.signing import generate_keypair, public_key_fingerprint, write_keypair

log = logging.getLogger("dd_collector.acquire")

__version__ = "0.1.0"
ProgressFn = Callable[[str], None]


@dataclass
class AcquisitionResult:
    case_id: str
    case_dir: Path
    file_count: int
    total_bytes: int
    public_key_fingerprint: str
    apps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name).strip("_") or "app"


def _new_case_id() -> str:
    return f"{datetime.now(UTC):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"


def _host() -> AcquisitionHost:
    import getpass

    tz = datetime.now().astimezone().tzname()
    return AcquisitionHost(
        hostname=platform.node() or "unknown",
        os=platform.system(),
        os_version=platform.release(),
        username=_getuser_safe(getpass),
        timezone=tz,
    )


def _getuser_safe(getpass) -> str:
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


def acquire(
    apps: list[ElectronApp],
    output_root: str | Path,
    *,
    case_id: str | None = None,
    operator: str = "",
    collect_memory: bool = False,
    signing_key_pem: bytes | None = None,
    progress: ProgressFn | None = None,
) -> AcquisitionResult:
    say = progress or (lambda _m: None)
    case_id = case_id or _new_case_id()
    layout = CaseLayout(output_root, case_id)
    layout.create_evidence_skeleton()
    custody = CustodyLog(layout.dir)
    started = datetime.now(UTC)
    warnings: list[str] = []

    custody.append(
        CustodyEvent.ACQUISITION_START, operator or "collector",
        f"case {case_id} on {platform.node()} ({platform.system()})",
    )
    say(f"Starting acquisition {case_id}")

    # signing key --------------------------------------------------------
    if signing_key_pem:
        private_pem = signing_key_pem
        from cryptography.hazmat.primitives import serialization

        from dd_core.signing import load_private_key

        public_pem = load_private_key(private_pem).public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    else:
        private_pem, public_pem = generate_keypair()
        write_keypair(layout.dir / "keys", private_pem, public_pem)
        say("Generated acquisition signing key (keys/collector_key.pem)")

    # per-app copy ------------------------------------------------------
    target_apps: list[TargetApp] = []
    for app in apps:
        dest_name = _safe_name(app.name)
        app_evidence = layout.evidence / dest_name
        app_evidence.mkdir(parents=True, exist_ok=True)
        say(f"[{app.name}] copying profile from {app.profile_dir}")
        specs = plan_profile_copy(app.profile_dir)
        if not specs:
            warnings.append(f"{app.name}: no recognisable Chromium data at {app.profile_dir}")
            say(f"[{app.name}] WARNING: nothing to copy")
            continue
        written = execute_copy(specs, app_evidence)
        say(f"[{app.name}] copied {len(written)} files")
        custody.append(
            CustodyEvent.ARTIFACT_ACQUIRED, operator or "collector",
            f"{app.name}: {len(written)} files from {app.profile_dir}",
            data={"sources": [s.dest_rel for s in specs]},
        )

        procs = [ProcessRef(pid=pid, name=app.name) for pid in app.pids]
        if collect_memory and app.pids:
            mem_dir = app_evidence / MEMORY
            for pid in app.pids:
                say(f"[{app.name}] dumping process {pid}")
                r = capture_process_memory(pid, app.name, mem_dir)
                if r.ok:
                    say(f"[{app.name}] pid {pid}: {r.detail}")
                else:
                    warnings.append(f"{app.name} pid {pid}: {r.detail}")
                custody.append(
                    CustodyEvent.ARTIFACT_ACQUIRED, operator or "collector",
                    f"{app.name} process {pid} memory: {'ok' if r.ok else 'skipped'} - {r.detail}",
                )

        target_apps.append(
            TargetApp(name=dest_name, user_data_dir=str(app.profile_dir),
                      version=app.version, processes=procs)
        )

    # full memory -----------------------------------------------------
    if collect_memory:
        say("Capturing full system memory")
        sys_mem = layout.evidence / "_system" / MEMORY
        r = capture_full_memory(sys_mem)
        custody.append(
            CustodyEvent.ARTIFACT_ACQUIRED, operator or "collector",
            f"full memory: {'ok' if r.ok else 'skipped'} - {r.detail}",
        )
        if r.ok:
            say(f"Full memory: {r.detail}")
            target_apps.append(TargetApp(name="_system", user_data_dir="(physical memory)"))
        else:
            warnings.append(f"full memory: {r.detail}")
            say(f"WARNING: {r.detail}")

    # manifest --------------------------------------------------------
    say("Hashing evidence and building the manifest")
    now = datetime.now(UTC)
    entries = []
    for f in sorted(layout.evidence.rglob("*")):
        if f.is_file():
            rel = f.relative_to(layout.evidence).as_posix()
            entries.append(
                build_entry(layout.evidence, f, SourceType(source_type_for_path(rel)), now)
            )

    manifest = Manifest(
        case_id=case_id,
        collector_version=__version__,
        operator=operator or "unknown",
        host=_host(),
        started_ts=started,
        completed_ts=now,
        target_apps=target_apps,
        entries=entries,
        public_key_pem=public_pem.decode(),
    )
    manifest.write(layout.dir, private_pem)
    fpr = public_key_fingerprint(public_pem)
    custody.append(CustodyEvent.ACQUISITION_COMPLETE, operator or "collector",
                   f"{len(entries)} files, {manifest.total_bytes:,} bytes")
    custody.append(CustodyEvent.MANIFEST_SIGNED, operator or "collector",
                   f"Ed25519, public key fingerprint {fpr}")
    say(f"Done. {len(entries)} files signed. Public key fingerprint: {fpr}")

    return AcquisitionResult(
        case_id=case_id,
        case_dir=layout.dir,
        file_count=len(entries),
        total_bytes=manifest.total_bytes,
        public_key_fingerprint=fpr,
        apps=[t.name for t in target_apps],
        warnings=warnings,
    )
