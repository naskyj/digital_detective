"""Evidence manifest: the signed inventory of every file the collector acquired.

Layout on disk (written into <case>/):
    manifest.json        - the manifest document below
    manifest.json.sig    - detached Ed25519 signature over the canonical bytes

Verification on import re-hashes every file under evidence/ and compares to the
manifest, then checks the signature. Any mismatch is a hard failure.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from dd_core.enums import SourceType
from dd_core.hashing import canonical_json, sha256_canonical, sha256_file
from dd_core.schema import TargetApp
from dd_core.signing import sign, verify

SCHEMA_VERSION = "1.0"
MANIFEST_NAME = "manifest.json"
SIGNATURE_NAME = "manifest.json.sig"


class AcquisitionHost(BaseModel):
    hostname: str
    os: str
    os_version: str
    username: str
    timezone: str | None = None


class ManifestEntry(BaseModel):
    path: str  # relative to <case>/evidence/
    sha256: str
    size_bytes: int
    source_type: SourceType
    mtime: datetime | None = None
    acquired_ts: datetime
    notes: str | None = None


class Manifest(BaseModel):
    schema_version: str = SCHEMA_VERSION
    case_id: str
    collector_version: str
    operator: str
    host: AcquisitionHost
    started_ts: datetime
    completed_ts: datetime
    target_apps: list[TargetApp] = Field(default_factory=list)
    entries: list[ManifestEntry] = Field(default_factory=list)
    public_key_pem: str  # the key the signature can be checked against

    @property
    def entry_count(self) -> int:
        return len(self.entries)

    @property
    def total_bytes(self) -> int:
        return sum(e.size_bytes for e in self.entries)

    def content_digest(self) -> str:
        """Hash over everything except the signature. This is what gets signed."""
        payload = self.model_dump(mode="json", exclude={"entries"})
        payload["entries"] = [e.model_dump(mode="json") for e in sorted(self.entries, key=lambda e: e.path)]
        return sha256_canonical(payload)

    def signing_bytes(self) -> bytes:
        return self.content_digest().encode("ascii")

    # ------------------------------------------------------------------ io

    def write(self, case_dir: str | Path, private_pem: bytes, password: bytes | None = None) -> Path:
        case_dir = Path(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = case_dir / MANIFEST_NAME
        manifest_path.write_text(
            canonical_json(self.model_dump(mode="json")), encoding="utf-8"
        )
        signature = sign(private_pem, self.signing_bytes(), password)
        (case_dir / SIGNATURE_NAME).write_bytes(signature)
        return manifest_path

    @classmethod
    def read(cls, case_dir: str | Path) -> Manifest:
        case_dir = Path(case_dir)
        return cls.model_validate_json((case_dir / MANIFEST_NAME).read_text("utf-8"))

    @classmethod
    def read_signature(cls, case_dir: str | Path) -> bytes:
        return (Path(case_dir) / SIGNATURE_NAME).read_bytes()


class VerificationIssue(BaseModel):
    path: str
    kind: str  # "missing" | "hash_mismatch" | "unexpected" | "size_mismatch"
    expected: str | None = None
    actual: str | None = None


class VerificationResult(BaseModel):
    ok: bool
    signature_valid: bool
    checked: int
    issues: list[VerificationIssue] = Field(default_factory=list)


def verify_case(case_dir: str | Path) -> VerificationResult:
    """Re-hash every file under evidence/, compare to the manifest, check the signature."""
    case_dir = Path(case_dir)
    evidence_dir = case_dir / "evidence"
    manifest = Manifest.read(case_dir)
    signature = Manifest.read_signature(case_dir)

    sig_ok = verify(manifest.public_key_pem.encode("ascii"), manifest.signing_bytes(), signature)

    issues: list[VerificationIssue] = []
    expected = {e.path: e for e in manifest.entries}

    for rel, entry in expected.items():
        fpath = evidence_dir / rel
        if not fpath.is_file():
            issues.append(VerificationIssue(path=rel, kind="missing"))
            continue
        actual_size = fpath.stat().st_size
        if actual_size != entry.size_bytes:
            issues.append(
                VerificationIssue(
                    path=rel, kind="size_mismatch",
                    expected=str(entry.size_bytes), actual=str(actual_size),
                )
            )
        actual_hash = sha256_file(fpath)
        if actual_hash != entry.sha256:
            issues.append(
                VerificationIssue(
                    path=rel, kind="hash_mismatch",
                    expected=entry.sha256, actual=actual_hash,
                )
            )

    if evidence_dir.is_dir():
        for fpath in sorted(evidence_dir.rglob("*")):
            if fpath.is_file():
                rel = fpath.relative_to(evidence_dir).as_posix()
                if rel not in expected:
                    issues.append(VerificationIssue(path=rel, kind="unexpected"))

    return VerificationResult(
        ok=sig_ok and not issues,
        signature_valid=sig_ok,
        checked=len(expected),
        issues=issues,
    )


def build_entry(
    evidence_root: str | Path,
    file_path: str | Path,
    source_type: SourceType,
    acquired_ts: datetime | None = None,
    notes: str | None = None,
) -> ManifestEntry:
    evidence_root = Path(evidence_root)
    file_path = Path(file_path)
    st = file_path.stat()
    return ManifestEntry(
        path=file_path.relative_to(evidence_root).as_posix(),
        sha256=sha256_file(file_path),
        size_bytes=st.st_size,
        source_type=source_type,
        mtime=datetime.fromtimestamp(st.st_mtime, tz=UTC),
        acquired_ts=acquired_ts or datetime.now(UTC),
        notes=notes,
    )
