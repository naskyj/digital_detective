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
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from dd_core.casedir import CaseLayout
from dd_core.custody import CustodyLog
from dd_core.enums import CustodyEvent, SourceType
from dd_core.evidence import MEMORY, execute_copy, plan_profile_copy, safe_name, source_type_for_path
from dd_core.hashing import sha256_file
from dd_core.manifest import AcquisitionHost, Manifest, build_entry
from dd_core.schema import TargetApp
from dd_core.signing import generate_keypair, public_key_fingerprint, write_keypair
from dd_lab.case import new_case_id
from dd_lab.imaging.discover import ImageApp, discover_apps, discover_apps_in_folder
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


def source_app_prefix(source_label: str) -> str:
    """The exact prefix a labelled evidence source's apps get in both the
    evidence folder name and `TargetApp.name`/the warehouse's `app` field -
    `safe_name(source_label) + "_"`. `safe_name` is applied to the label and
    the app name SEPARATELY, then joined - not `safe_name(f"{label} / {app}")`
    as one combined string, which would collapse the space+slash separator
    into just another underscore, indistinguishable from the underscores
    `safe_name` already produces from a messy label on its own, and
    unrecoverable afterwards. The API layer uses this exact function to
    recognise which of a case's registered sources produced a given app
    entry (see `dd_lab.api.main::_source_label_for_app`), instead of trying
    to reverse-parse a label back out of an already-sanitised string."""
    return f"{safe_name(source_label)}_"




def _image_host(target, image_path: str, image_sha256: str) -> AcquisitionHost:
    """The EXAMINED machine's own identity, read from the image itself —
    never `platform.node()`/`platform.system()`, which report whichever
    machine is running this code (the investigator's), not the one the
    evidence came from. Confirmed wrong in practice: a Windows E01 opened on
    a Mac reported the analyst's own hostname ("MacBook-Pro.local") until
    this fix. `dissect` resolves hostname/os/version from the image's own
    registry/filesystem where available; falls back to "unknown" (never to
    the analyst's machine) when the image's OS can't be determined."""

    def attr(name: str) -> str | None:
        try:
            val = getattr(target, name, None)
        except Exception:  # noqa: BLE001 - dissect raises its own exception types here
            return None
        return str(val) if val else None

    return AcquisitionHost(
        hostname=attr("hostname") or "unknown (not determinable from this image)",
        os=attr("os") or "unknown", os_version=attr("version") or "unknown",
        username="", timezone="",
        source_image=image_path, source_image_sha256=image_sha256,
    )


def _extract_discovered_apps(
    found: list[tuple[str, Path, str | None]],
    *,
    layout: CaseLayout,
    custody: CustodyLog,
    operator: str,
    say: ProgressFn,
    warnings: list[str],
    provenance: str,
    source_label: str | None = None,
) -> list[TargetApp]:
    """Copy every discovered (name, profile_dir, version) into the case's
    evidence/ directory, hash + custody-log each one - the one per-app copy
    loop `acquire_from_image` and `acquire_from_folder` both drive, so a fix
    here (or to `plan_profile_copy`/`execute_copy` in dd_core) applies to
    both acquisition paths identically.

    `source_label`, when this is a SECOND (or third, ...) evidence source
    being added to a case, makes the folder + `TargetApp.name` for every app
    here start with `source_app_prefix(source_label)` - `safe_name(label)`
    and `safe_name(name)` sanitised SEPARATELY then joined, not the combined
    "<label> / <name>" string sanitised as one, which would lose the
    separator (see `source_app_prefix`'s own docstring). This is what keeps
    two sources' identically-named apps (two people's "discord") from
    silently landing in - and merging into - the same evidence folder."""
    target_apps: list[TargetApp] = []
    prefix = source_app_prefix(source_label) if source_label else ""
    for name, profile_dir, version in found:
        display_name = f"{source_label} / {name}" if source_label else name
        dest_name = prefix + safe_name(name)
        app_evidence = layout.evidence / dest_name
        app_evidence.mkdir(parents=True, exist_ok=True)

        say(f"[{display_name}] extracting profile from {profile_dir}")
        specs = plan_profile_copy(profile_dir)
        if not specs:
            warnings.append(f"{display_name}: no recognisable Chromium data at {profile_dir}")
            say(f"[{display_name}] WARNING: nothing to extract")
            continue

        try:
            report = execute_copy(specs, app_evidence)
        except Exception as e:  # noqa: BLE001 - one bad app must not kill the case
            warnings.append(f"{display_name}: extraction failed ({e}); skipped")
            say(f"[{display_name}] ERROR: extraction failed: {e}")
            custody.append(CustodyEvent.NOTE, operator, f"{display_name}: extraction failed - {e}")
            continue

        say(f"[{display_name}] extracted {len(report.written)} files"
            + (f", skipped {len(report.skipped)}" if report.skipped else ""))
        for src, reason in report.skipped[:20]:
            warnings.append(f"{display_name}: could not extract {src} ({reason})")
        custody.append(
            CustodyEvent.ARTIFACT_ACQUIRED, operator,
            f"{display_name}: {len(report.written)} files extracted, {len(report.skipped)} skipped, "
            f"{provenance} {profile_dir}",
            data={"sources": [s.dest_rel for s in specs], "skipped": [s for s, _ in report.skipped[:50]]},
        )
        target_apps.append(TargetApp(name=dest_name, user_data_dir=str(profile_dir), version=version))
    return target_apps


def _copy_supplied_memory_image(
    memory_image_path: str | Path,
    *,
    layout: CaseLayout,
    custody: CustodyLog,
    operator: str,
    say: ProgressFn,
    target_apps: list[TargetApp],
    warnings: list[str],
) -> None:
    say(f"Copying supplied memory image {memory_image_path}")
    mem_dir = layout.evidence / "_system" / MEMORY
    mem_dir.mkdir(parents=True, exist_ok=True)
    dest = mem_dir / "physmem.raw"
    src = Path(memory_image_path)
    if src.is_file():
        shutil.copyfile(src, dest)
        target_apps.append(TargetApp(name="_system", user_data_dir="(supplied memory image)"))
        custody.append(
            CustodyEvent.ARTIFACT_ACQUIRED, operator, f"memory image copied from {memory_image_path}"
        )
    else:
        warnings.append(f"memory image {memory_image_path} not found")


def _sign_and_finalize_manifest(
    *,
    case_id: str,
    layout: CaseLayout,
    host: AcquisitionHost,
    started: datetime,
    target_apps: list[TargetApp],
    operator: str,
    private_pem: bytes,
    public_pem: bytes,
    collector_version: str,
    custody: CustodyLog,
    say: ProgressFn,
    source_summary: str,
) -> tuple[Manifest, str]:
    """Hash every copied file, build + sign the manifest, seal the
    acquisition into the custody log. Shared by `acquire_from_image` and
    `acquire_from_folder` - the only thing that differs between them by
    this point is `host` (how the examined machine's identity was
    determined) and `source_summary` (what to call the evidentiary root in
    the custody log's completion message).

    `entries` (file hashes) are rebuilt from the FULL evidence/ tree every
    call, so adding a source to a case someone already extracted evidence
    for is automatically complete - but `target_apps` is only ever what
    THIS call's caller discovered, which is just the new source's apps, not
    a previous source's. Real bug found while testing multi-source: a case
    with 2 sources ended up with a manifest listing only the 2nd source's
    app - the 1st source's evidence files were still on disk and correctly
    hashed into `entries`, but `ingest_case()` gets its app list from
    `manifest.target_apps` (not by scanning evidence/ - see ingest.py),
    so the 1st source's evidence would have silently stopped being
    re-ingested on every future "Run analysis". Fixed by merging with
    whatever the previous manifest (if any) already had, by name - the new
    call's entry for a name wins on a genuine name clash (re-acquiring the
    same unlabelled source again, refreshing it)."""
    say("Hashing evidence and building the manifest")
    now = datetime.now(UTC)
    entries = []
    for f in sorted(layout.evidence.rglob("*")):
        if f.is_file():
            rel = f.relative_to(layout.evidence).as_posix()
            entries.append(build_entry(layout.evidence, f, SourceType(source_type_for_path(rel)), now))

    if layout.manifest.is_file():
        previous = {t.name: t for t in Manifest.read(layout.dir).target_apps}
        previous.update({t.name: t for t in target_apps})
        target_apps = list(previous.values())

    manifest = Manifest(
        case_id=case_id, collector_version=collector_version, operator=operator or "unknown",
        host=host, started_ts=started, completed_ts=now,
        target_apps=target_apps, entries=entries, public_key_pem=public_pem.decode(),
    )
    manifest.write(layout.dir, private_pem)
    fpr = public_key_fingerprint(public_pem)
    custody.append(CustodyEvent.ACQUISITION_COMPLETE, operator,
                   f"{len(entries)} files, {manifest.total_bytes:,} bytes, {source_summary}")
    custody.append(CustodyEvent.MANIFEST_SIGNED, operator, f"Ed25519, public key fingerprint {fpr}")
    say(f"Done. {len(entries)} files signed. Public key fingerprint: {fpr}")
    return manifest, fpr


def _generate_or_load_signing_key(
    layout: CaseLayout, signing_key_pem: bytes | None, say: ProgressFn
) -> tuple[bytes, bytes]:
    if signing_key_pem:
        from cryptography.hazmat.primitives import serialization

        from dd_core.signing import load_private_key

        public_pem = load_private_key(signing_key_pem).public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        return signing_key_pem, public_pem
    private_pem, public_pem = generate_keypair()
    write_keypair(layout.dir / "keys", private_pem, public_pem)
    say("Generated acquisition signing key (keys/collector_key.pem)")
    return private_pem, public_pem


def acquire_from_image(
    image_path: str | Path,
    output_root: str | Path,
    *,
    apps: list[str] | None = None,
    memory_image_path: str | Path | None = None,
    case_id: str | None = None,
    operator: str = "",
    signing_key_pem: bytes | None = None,
    source_label: str | None = None,
    progress: ProgressFn | None = None,
) -> ImageAcquisitionResult:
    """Extract Chromium/Electron profile evidence from a disk image.

    `apps` names/substrings restrict which discovered profiles are extracted
    (default: every profile found), matched against the RAW discovered name
    - before `source_label` (below) is applied. `memory_image_path` is an
    optional, separately supplied full-RAM capture (e.g. from the live
    collector's `--memory`, or acquired by other means) to fold into the
    same case.

    `source_label`, when given, prefixes every extracted app's evidence
    folder and `app` name with `source_app_prefix(source_label)` (see its
    own docstring for exactly how) - the investigator-facing way to add a
    SECOND (or third, ...) evidence source to a case someone else's
    evidence already lives in without its apps colliding with (and
    silently merging into) an identically-named app from a different
    source. Purely a naming convention on the already-free-text `app`
    field - no schema change needed, and the same "embed context into the
    name" pattern this app already uses for a WebView2-embedded profile
    ("Roblox (WebView2 via ...)").
    """
    say = progress or (lambda _m: None)
    case_id = case_id or new_case_id()
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
              "image_size_bytes": Path(image_path).stat().st_size,
              "source_label": source_label},
    )
    say(f"Image sha256: {image_sha256}")

    target = open_image(image_path)
    private_pem, public_pem = _generate_or_load_signing_key(layout, signing_key_pem, say)

    say("Scanning the image for Electron/Chromium app profiles")
    found: list[ImageApp] = discover_apps(target)
    if apps:
        found = [a for a in found if any(spec.lower() in a.name.lower() for spec in apps)]
    if not found:
        warnings.append("no matching Electron app profiles found on this image")
        say("WARNING: no matching app profiles found")

    target_apps = _extract_discovered_apps(
        [(a.name, a.profile_dir, a.version) for a in found],
        layout=layout, custody=custody, operator=operator or "image-extractor",
        say=say, warnings=warnings, provenance="from image path", source_label=source_label,
    )

    if memory_image_path:
        _copy_supplied_memory_image(
            memory_image_path, layout=layout, custody=custody, operator=operator or "image-extractor",
            say=say, target_apps=target_apps, warnings=warnings,
        )

    manifest, fpr = _sign_and_finalize_manifest(
        case_id=case_id, layout=layout, host=_image_host(target, str(image_path), image_sha256),
        started=started, target_apps=target_apps, operator=operator or "image-extractor",
        private_pem=private_pem, public_pem=public_pem,
        collector_version=f"image-extractor-{__version__}", custody=custody, say=say,
        source_summary=f"from image {image_sha256[:16]}…",
    )

    return ImageAcquisitionResult(
        case_id=case_id, case_dir=layout.dir, file_count=len(manifest.entries),
        total_bytes=manifest.total_bytes, public_key_fingerprint=fpr,
        image_sha256=image_sha256, apps=[t.name for t in target_apps], warnings=warnings,
    )


def acquire_from_folder(
    folder_path: str | Path,
    output_root: str | Path,
    *,
    apps: list[str] | None = None,
    memory_image_path: str | Path | None = None,
    case_id: str | None = None,
    operator: str = "",
    signing_key_pem: bytes | None = None,
    source_label: str | None = None,
    progress: ProgressFn | None = None,
) -> ImageAcquisitionResult:
    """Extract Chromium/Electron profile evidence from a plain folder that
    already holds one subfolder per app - an examiner's own prior
    extraction, evidence handed over by someone else, files pulled out of a
    mount some other way - rather than a disk image file or a folder shaped
    like an OS root (`open_image`/`acquire_from_image` already handle both
    of those, via `dissect`'s own directory loader).

    Not "live collection": this reads static files that already exist on
    disk, the same category of input an image is, just shaped differently -
    `dd-lab` still never touches a live, running machine's own AppData
    paths (that stays `dd_collector`'s job).

    No single file to hash as the evidentiary root the way an image has one
    (`acquire_from_image` hashes the whole image file first) - each copied
    file gets its own SHA-256 as it's copied instead, recorded in the
    manifest's `entries` exactly like the live collector's own acquisitions
    already work; `apps`/`memory_image_path`/`source_label`/etc. all mean
    the same thing they do for `acquire_from_image` (see its docstring for
    `source_label`).
    """
    say = progress or (lambda _m: None)
    case_id = case_id or new_case_id()
    layout = CaseLayout(output_root, case_id)
    layout.create_evidence_skeleton()
    custody = CustodyLog(layout.dir)
    started = datetime.now(UTC)
    warnings: list[str] = []
    folder = Path(folder_path)

    custody.append(
        CustodyEvent.ACQUISITION_START, operator or "folder-import",
        f"case {case_id}: importing from folder {folder}",
        data={"folder_path": str(folder), "source_label": source_label},
    )

    private_pem, public_pem = _generate_or_load_signing_key(layout, signing_key_pem, say)

    say("Scanning the folder for Electron/Chromium app profiles")
    found: list[ImageApp] = discover_apps_in_folder(folder)
    if apps:
        found = [a for a in found if any(spec.lower() in a.name.lower() for spec in apps)]
    if not found:
        warnings.append("no matching Electron app profiles found in this folder")
        say("WARNING: no matching app profiles found")

    target_apps = _extract_discovered_apps(
        [(a.name, a.profile_dir, a.version) for a in found],
        layout=layout, custody=custody, operator=operator or "folder-import",
        say=say, warnings=warnings, provenance="from folder path", source_label=source_label,
    )

    if memory_image_path:
        _copy_supplied_memory_image(
            memory_image_path, layout=layout, custody=custody, operator=operator or "folder-import",
            say=say, target_apps=target_apps, warnings=warnings,
        )

    host = AcquisitionHost(
        hostname="unknown (source: an imported folder, not a live machine or a disk image)",
        os="unknown", os_version="unknown", username="", timezone=None,
        source_image=str(folder), source_image_sha256=None,
    )
    manifest, fpr = _sign_and_finalize_manifest(
        case_id=case_id, layout=layout, host=host, started=started, target_apps=target_apps,
        operator=operator or "folder-import", private_pem=private_pem, public_pem=public_pem,
        collector_version=f"folder-import-{__version__}", custody=custody, say=say,
        source_summary=f"from folder {folder}",
    )

    return ImageAcquisitionResult(
        case_id=case_id, case_dir=layout.dir, file_count=len(manifest.entries),
        total_bytes=manifest.total_bytes, public_key_fingerprint=fpr,
        image_sha256="", apps=[t.name for t in target_apps], warnings=warnings,
    )
