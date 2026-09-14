"""Extract evidence straight out of a disk image — the image-based
alternative to `dd_collector.acquire` (which needs a live machine).

Produces the exact same case shape: signed `manifest.json` + hash-chained
`chain_of_custody.jsonl` + `evidence/<app>/...` in the canonical Chromium
layout. Everything downstream (ingest, warehouse, classify, correlate,
findings, judge, report) is unaware of and unaffected by which acquisition
path produced the case.

The evidentiary root is different from a live acquisition, though: there the
root of trust is "we hashed each file as we copied it off a live machine";
here it's "we hashed the whole image file(s) first" — that hash (recorded in
the manifest's `AcquisitionHost.source_image*` fields and in the custody log)
is what ties every extracted artifact back to one, specific, unmodified image.
The image itself is opened read-only; nothing here can write to it.
"""

from __future__ import annotations

import logging
import platform
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from dd_core.casedir import CaseLayout
from dd_core.custody import CustodyLog
from dd_core.enums import CustodyEvent, SourceType
from dd_core.evidence import MEMORY, execute_copy, plan_profile_copy, source_type_for_path
from dd_core.hashing import sha256_file
from dd_core.manifest import AcquisitionHost, Manifest, build_entry
from dd_core.schema import TargetApp
from dd_core.signing import generate_keypair, public_key_fingerprint, write_keypair
from dd_lab.imaging.discover import ImageApp, discover_apps
from dd_lab.imaging.image import open_image

log = logging.getLogger("dd_lab.imaging.acquire")

__version__ = "0.1.0"
ProgressFn = Callable[[str], None]


@dataclass
class ImageAcquisitionResult:
    case_id: str
    case_dir: Path
    file_count: int
    total_bytes: int
    public_key_fingerprint: str
    image_sha256: str
    apps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name).strip("_") or "app"


def _new_case_id() -> str:
    return f"{datetime.now(UTC):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"


def acquire_from_image(
    image_path: str | Path,
    output_root: str | Path,
    *,
    apps: list[str] | None = None,
    memory_image_path: str | Path | None = None,
    case_id: str | None = None,
    operator: str = "",
    signing_key_pem: bytes | None = None,
    progress: ProgressFn | None = None,
) -> ImageAcquisitionResult:
    """Extract Chromium/Electron profile evidence from a disk image.

    `apps` names/substrings restrict which discovered profiles are extracted
    (default: every profile found). `memory_image_path` is an optional,
    separately supplied full-RAM capture (e.g. from the live collector's
    `--memory`, or acquired by other means) to fold into the same case.
    """
    say = progress or (lambda _m: None)
    case_id = case_id or _new_case_id()
    layout = CaseLayout(output_root, case_id)
    layout.create_evidence_skeleton()
    custody = CustodyLog(layout.dir)
    started = datetime.now(UTC)
    warnings: list[str] = []

    say(f"Hashing image {image_path} (this can take a while for a large image)")
    image_sha256 = sha256_file(image_path)
    custody.append(
        CustodyEvent.ACQUISITION_START, operator or "image-extractor",
        f"case {case_id}: extracting from image {Path(image_path).name}",
        data={"image_path": str(image_path), "image_sha256": image_sha256,
              "image_size_bytes": Path(image_path).stat().st_size},
    )
    say(f"Image sha256: {image_sha256}")

    target = open_image(image_path)

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

    say("Scanning the image for Electron/Chromium app profiles")
    found: list[ImageApp] = discover_apps(target)
    if apps:
        found = [a for a in found if any(spec.lower() in a.name.lower() for spec in apps)]
    if not found:
        warnings.append("no matching Electron app profiles found on this image")
        say("WARNING: no matching app profiles found")

    target_apps: list[TargetApp] = []
    for app in found:
        dest_name = _safe_name(app.name)
        app_evidence = layout.evidence / dest_name
        app_evidence.mkdir(parents=True, exist_ok=True)

        say(f"[{app.name}] extracting profile from {app.profile_dir}")
        specs = plan_profile_copy(app.profile_dir)
        if not specs:
            warnings.append(f"{app.name}: no recognisable Chromium data at {app.profile_dir}")
            say(f"[{app.name}] WARNING: nothing to extract")
            continue

        try:
            report = execute_copy(specs, app_evidence)
        except Exception as e:  # noqa: BLE001 - one bad app must not kill the case
            warnings.append(f"{app.name}: extraction failed ({e}); skipped")
            say(f"[{app.name}] ERROR: extraction failed: {e}")
            custody.append(CustodyEvent.NOTE, operator or "image-extractor",
                           f"{app.name}: extraction failed - {e}")
            continue

        say(f"[{app.name}] extracted {len(report.written)} files"
            + (f", skipped {len(report.skipped)}" if report.skipped else ""))
        for src, reason in report.skipped[:20]:
            warnings.append(f"{app.name}: could not extract {src} ({reason})")
        custody.append(
            CustodyEvent.ARTIFACT_ACQUIRED, operator or "image-extractor",
            f"{app.name}: {len(report.written)} files extracted, {len(report.skipped)} skipped, "
            f"from image path {app.profile_dir}",
            data={"sources": [s.dest_rel for s in specs], "skipped": [s for s, _ in report.skipped[:50]]},
        )
        target_apps.append(TargetApp(name=dest_name, user_data_dir=str(app.profile_dir), version=app.version))

    if memory_image_path:
        say(f"Copying supplied memory image {memory_image_path}")
        mem_dir = layout.evidence / "_system" / MEMORY
        mem_dir.mkdir(parents=True, exist_ok=True)
        dest = mem_dir / "physmem.raw"
        src = Path(memory_image_path)
        if src.is_file():
            shutil.copyfile(src, dest)
            target_apps.append(TargetApp(name="_system", user_data_dir="(supplied memory image)"))
            custody.append(CustodyEvent.ARTIFACT_ACQUIRED, operator or "image-extractor",
                           f"memory image copied from {memory_image_path}")
        else:
            warnings.append(f"memory image {memory_image_path} not found")

    say("Hashing evidence and building the manifest")
    now = datetime.now(UTC)
    entries = []
    for f in sorted(layout.evidence.rglob("*")):
        if f.is_file():
            rel = f.relative_to(layout.evidence).as_posix()
            entries.append(build_entry(layout.evidence, f, SourceType(source_type_for_path(rel)), now))

    manifest = Manifest(
        case_id=case_id,
        collector_version=f"image-extractor-{__version__}",
        operator=operator or "unknown",
        host=AcquisitionHost(
            hostname=platform.node() or "unknown", os=platform.system(),
            os_version=platform.release(), username="", timezone="",
            source_image=str(image_path), source_image_sha256=image_sha256,
        ),
        started_ts=started, completed_ts=now,
        target_apps=target_apps, entries=entries,
        public_key_pem=public_pem.decode(),
    )
    manifest.write(layout.dir, private_pem)
    fpr = public_key_fingerprint(public_pem)
    custody.append(CustodyEvent.ACQUISITION_COMPLETE, operator or "image-extractor",
                   f"{len(entries)} files, {manifest.total_bytes:,} bytes, from image {image_sha256[:16]}…")
    custody.append(CustodyEvent.MANIFEST_SIGNED, operator or "image-extractor",
                   f"Ed25519, public key fingerprint {fpr}")
    say(f"Done. {len(entries)} files signed. Public key fingerprint: {fpr}")

    return ImageAcquisitionResult(
        case_id=case_id, case_dir=layout.dir, file_count=len(entries),
        total_bytes=manifest.total_bytes, public_key_fingerprint=fpr,
        image_sha256=image_sha256, apps=[t.name for t in target_apps], warnings=warnings,
    )
