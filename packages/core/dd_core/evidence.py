"""Evidence layout — the collector images the whole Electron profile.

The evidence under `<case>/evidence/<app>/` mirrors the source profile's own
Chromium directory structure exactly, so nothing is lost and the copy looks
like the original. A small denylist skips pure GPU/shader/ML caches that carry
no user data and can be large.

Paths the lab parsers rely on (Chromium-native, under `<app>/`):

    Local Storage/leveldb/       key/value store — tokens, settings
    Session Storage/             per-session key/value store
    IndexedDB/*.leveldb          structured store — messages, users, channels
    Cache/  (or Cache/Cache_Data/)   HTTP disk cache
    Service Worker/CacheStorage/ Cache API storage (chat apps cache API JSON here)
    Code Cache/                  compiled-script cache
    Network/Cookies  or  Cookies SQLite cookie store
    Preferences                  per-profile settings (JSON)
    Local State                  browser-wide state incl. the crypt key (JSON)
    blob_storage/                Blob data (attachments in some apps)
    Sessions/, Current Tabs, ... open-tab / navigation state
    memory/                      process / full memory dumps (collector-added)
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

# Chromium-native relative paths under <case>/evidence/<app>/
LOCAL_STORAGE = "Local Storage/leveldb"
SESSION_STORAGE = "Session Storage"
INDEXEDDB = "IndexedDB"
CACHE = "Cache"
SERVICE_WORKER_CACHE = "Service Worker/CacheStorage"
CODE_CACHE = "Code Cache"
COOKIES = "Network/Cookies"
COOKIES_LEGACY = "Cookies"
PREFERENCES = "Preferences"
LOCAL_STATE = "Local State"
MEMORY = "memory"

# Directories with no user-relevant content that can be large — never copied.
DENYLIST = frozenset({
    # GPU / shader / graphics caches
    "GPUCache", "GrShaderCache", "ShaderCache", "GraphiteDawnCache",
    "DawnCache", "DawnGraphiteCache", "DawnWebGPUCache",
    # component / model / ML stores shipped by Chromium
    "component_crx_cache", "extensions_crx_cache",
    "optimization_guide_model_store", "optimization_guide_hint_cache_store",
    "OptimizationGuidePredictionModels", "segmentation_platform",
    "PrivacySandboxAttestationsPreloaded", "FirstPartySetsPreloaded",
    "TrustTokenKeyCommitments", "MEIPreload", "WidevineCdm",
    "CertificateRevocation", "SafetyTips", "Subresource Filter",
    "Safe Browsing", "AutofillStates", "hyphen-data", "OnDeviceHeadSuggestModel",
    # crash / telemetry
    "Crashpad", "Crash Reports", "crashDumps", "BrowserMetrics",
    # VS Code / large-editor bloat (compiled data / editor UI state, not comms data)
    "CachedData", "CachedExtensions", "CachedExtensionVSIXs", "CachedProfilesData",
    "Dictionaries", "logs", "clp", "workspaceStorage",
    # compiled scripts — bytecode, no user data
    "ScriptCache",
})

# Friendly aliases accepted in `include` (so callers/tests need not know the
# exact Chromium path).
_ALIASES = {
    "local_storage": LOCAL_STORAGE, "session_storage": SESSION_STORAGE,
    "indexeddb": INDEXEDDB, "cache": CACHE, "cookies": COOKIES,
    "preferences": PREFERENCES, "local_state": LOCAL_STATE,
}


@dataclass
class CopySpec:
    """One source -> destination copy the collector performs for an app profile."""

    source: Path
    dest_rel: str
    is_dir: bool
    exclude_names: frozenset[str] = field(default_factory=frozenset)


def plan_profile_copy(
    profile_dir: str | Path, include: frozenset[str] | None = None
) -> list[CopySpec]:
    """Return the copies to make for a live Electron profile directory.

    Default: image the entire profile (recursively) minus `DENYLIST`, preserving
    structure. `include` (a set of the path constants above) restricts to a
    named subset — used by the tests.
    """
    profile = Path(profile_dir)
    specs: list[CopySpec] = []

    if include is None:
        if profile.is_dir():
            specs.append(CopySpec(profile, ".", True, exclude_names=DENYLIST))
        # Local State usually sits inside the profile; in a real Chromium install
        # it's one level up. Grab that copy too if present.
        parent_state = profile.parent / "Local State"
        if parent_state.is_file() and not (profile / "Local State").is_file():
            specs.append(CopySpec(parent_state, "Local State", False))
        return specs

    # subset mode (tests) — accept friendly aliases or exact paths
    include = frozenset(_ALIASES.get(x, x) for x in include)
    def add_file(rel: str) -> None:
        src = profile / rel
        if src.is_file():
            specs.append(CopySpec(src, rel, False))

    def add_dir(rel: str) -> None:
        src = profile / rel
        if src.is_dir():
            specs.append(CopySpec(src, rel, True))

    if LOCAL_STORAGE in include:
        add_dir(LOCAL_STORAGE)
    if SESSION_STORAGE in include:
        add_dir(SESSION_STORAGE)
    if INDEXEDDB in include:
        add_dir(INDEXEDDB)
    if CACHE in include:
        add_dir(CACHE)
    if PREFERENCES in include:
        add_file(PREFERENCES)
    if LOCAL_STATE in include:
        add_file(LOCAL_STATE)
        if not (profile / LOCAL_STATE).is_file() and (profile.parent / "Local State").is_file():
            specs.append(CopySpec(profile.parent / "Local State", LOCAL_STATE, False))
    if COOKIES in include or COOKIES_LEGACY in include:
        for rel in (COOKIES, COOKIES_LEGACY):
            add_file(rel)
    return specs


# path prefix (under <app>/) -> SourceType string
_SOURCE_TYPE_PREFIXES = (
    ("Local Storage/leveldb", "local_storage"),
    ("Session Storage", "session_storage"),
    ("IndexedDB", "indexeddb"),
    ("Service Worker/CacheStorage", "cache"),
    ("Code Cache", "cache"),
    ("Cache", "cache"),
    ("Network/Cookies", "cookies"),
    ("Cookies", "cookies"),
    ("Preferences", "preferences"),
    ("Local State", "local_state"),
    ("memory", "memory_process"),
)


def source_type_for_path(evidence_relative_path: str) -> str:
    """Map an evidence path like 'Slack/Cache/Cache_Data/f_00001' to a SourceType string."""
    parts = evidence_relative_path.replace("\\", "/").split("/")
    tail = "/".join(parts[1:]) if len(parts) > 1 else evidence_relative_path
    for prefix, source_type in _SOURCE_TYPE_PREFIXES:
        if tail.startswith(prefix):
            return source_type
    return "preferences"


@dataclass
class CopyReport:
    written: list[Path]
    skipped: list[tuple[str, str]]  # (source path, reason)


_MAX_COPY_DEPTH = 64  # defense in depth against a pathologically deep tree


def _copy_tree_resilient(
    src: Path, dst: Path, report: CopyReport, exclude: frozenset[str] = frozenset(), depth: int = 0,
) -> None:
    """Recursively copy, skipping unreadable files (e.g. a LevelDB .log locked by
    a running app) and any directory named in `exclude`, instead of aborting.

    We are copying from a machine under examination, which may be compromised
    or simply hostile: a symlink is never followed (it is recorded and
    skipped, not resolved and copied) so a symlink loop or a link out of the
    profile (to `/dev/zero`, another user's files, ...) can't turn a copy into
    a hang or an unbounded read. Recursion depth is capped as a second layer of
    defense against a pathologically deep directory tree.
    """
    if depth > _MAX_COPY_DEPTH:
        report.skipped.append((str(src), f"max copy depth ({_MAX_COPY_DEPTH}) exceeded"))
        return
    try:
        entries = sorted(src.iterdir())
    except (PermissionError, OSError) as e:
        report.skipped.append((str(src), str(e)))
        return
    dst.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        if entry.name in exclude:
            continue
        target = dst / entry.name
        try:
            if entry.is_symlink():
                report.skipped.append((str(entry), "symlink (not followed)"))
            elif entry.is_dir():
                _copy_tree_resilient(entry, target, report, exclude, depth + 1)
            elif entry.is_file():
                shutil.copy2(entry, target)
                report.written.append(target)
        except (PermissionError, OSError, shutil.SameFileError) as e:
            report.skipped.append((str(entry), str(e)))


def execute_copy(specs: list[CopySpec], app_evidence_dir: str | Path) -> CopyReport:
    """Copy each spec into <app_evidence_dir>.

    Never aborts on a single unreadable file — a locked file is recorded in
    `skipped` and the rest of the acquisition proceeds.
    """
    app_dir = Path(app_evidence_dir)
    report = CopyReport(written=[], skipped=[])
    for spec in specs:
        dest = app_dir if spec.dest_rel == "." else app_dir / spec.dest_rel
        if spec.is_dir:
            _copy_tree_resilient(spec.source, dest, report, spec.exclude_names)
        elif spec.source.is_symlink():
            report.skipped.append((str(spec.source), "symlink (not followed)"))
        else:
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(spec.source, dest)
                report.written.append(dest)
            except (PermissionError, OSError, shutil.SameFileError) as e:
                report.skipped.append((str(spec.source), str(e)))
    return report
