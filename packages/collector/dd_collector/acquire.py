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
    sources: frozenset[str] | None = None,
) -> AcquisitionResult:
    """`sources` restricts which profile stores are copied (default: the whole
    profile). Mainly for tests — real acquisitions image everything."""
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

    # volatile first: memory before disk (system as found) ----------------
    if collect_memory:
        say("Capturing full physical memory (this can take several minutes)")
        sys_mem = layout.evidence / "_system" / MEMORY
        r = capture_full_memory(sys_mem)
        custody.append(
            CustodyEvent.ARTIFACT_ACQUIRED, operator or "collector",
            f"full memory: {'ok' if r.ok else 'skipped'} - {r.detail}",
        )
        (say if r.ok else warnings.append)(f"full memory: {r.detail}")

        for app in apps:
            if not app.pids:
                continue
            mem_dir = layout.evidence / _safe_name(app.name) / MEMORY
            for pid in app.pids:
                say(f"[{app.name}] dumping process {pid}")
                pr = capture_process_memory(pid, app.name, mem_dir)
                (say if pr.ok else warnings.append)(f"[{app.name}] pid {pid}: {pr.detail}")
                custody.append(
                    CustodyEvent.ARTIFACT_ACQUIRED, operator or "collector",
                    f"{app.name} process {pid} memory: {'ok' if pr.ok else 'skipped'} - {pr.detail}",
                )

    # per-app disk copy --------------------------------------------------
    target_apps: list[TargetApp] = []
    for app in apps:
        dest_name = _safe_name(app.name)
        app_evidence = layout.evidence / dest_name
        app_evidence.mkdir(parents=True, exist_ok=True)

        if app.running:
            msg = (f"{app.name} is RUNNING (pids {', '.join(map(str, app.pids))}) — "
                   f"its LevelDB write-ahead logs are locked; committed data still copies, "
                   f"quit the app for a fully consistent snapshot")
            warnings.append(msg)
            say(f"[{app.name}] NOTE: {msg}")

        say(f"[{app.name}] copying profile from {app.profile_dir}")
        specs = plan_profile_copy(app.profile_dir, sources)
        if not specs:
            warnings.append(f"{app.name}: no recognisable Chromium data at {app.profile_dir}")
            say(f"[{app.name}] WARNING: nothing to copy")
            continue

        try:
            report = execute_copy(specs, app_evidence)
        except Exception as e:  # noqa: BLE001 - one bad app must not kill the case
            warnings.append(f"{app.name}: copy failed ({e}); skipped")
            say(f"[{app.name}] ERROR: copy failed: {e}")
            custody.append(CustodyEvent.NOTE, operator or "collector",
                           f"{app.name}: copy failed - {e}")
            continue

        say(f"[{app.name}] copied {len(report.written)} files"
            + (f", skipped {len(report.skipped)} locked/unreadable" if report.skipped else ""))
        for src, reason in report.skipped[:20]:
            warnings.append(f"{app.name}: could not copy {src} ({reason})")
        custody.append(
            CustodyEvent.ARTIFACT_ACQUIRED, operator or "collector",
            f"{app.name}: {len(report.written)} files copied, {len(report.skipped)} skipped, "
            f"from {app.profile_dir}",
            data={"sources": [s.dest_rel for s in specs],
                  "skipped": [s for s, _ in report.skipped[:50]]},
        )

        target_apps.append(
            TargetApp(
                name=dest_name, user_data_dir=str(app.profile_dir), version=app.version,
                processes=[ProcessRef(pid=pid, name=app.name) for pid in app.pids],
            )
        )

    if collect_memory and (layout.evidence / "_system" / MEMORY).is_dir():
        target_apps.append(TargetApp(name="_system", user_data_dir="(physical memory)"))

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
