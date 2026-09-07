"""dd_core - shared foundations for the Digital Detective collector and lab app."""

from __future__ import annotations

from dd_core.casedir import CaseLayout
from dd_core.custody import CustodyLog, CustodyRecord
from dd_core.enums import (
    ArtifactCategory,
    CustodyEvent,
    SourceType,
    ValueType,
)
from dd_core.evidence import (
    CopySpec,
    execute_copy,
    plan_profile_copy,
)
from dd_core.manifest import (
    AcquisitionHost,
    Manifest,
    ManifestEntry,
    VerificationResult,
    build_entry,
    verify_case,
)
from dd_core.schema import (
    ClassificationResult,
    NormalizedArtifact,
    ProcessRef,
    TargetApp,
)

__version__ = "0.1.0"

__all__ = [
    "AcquisitionHost",
    "ArtifactCategory",
    "CaseLayout",
    "ClassificationResult",
    "CopySpec",
    "CustodyEvent",
    "CustodyLog",
    "CustodyRecord",
    "execute_copy",
    "plan_profile_copy",
    "Manifest",
    "ManifestEntry",
    "NormalizedArtifact",
    "ProcessRef",
    "SourceType",
    "TargetApp",
    "ValueType",
    "VerificationResult",
    "build_entry",
    "verify_case",
]
