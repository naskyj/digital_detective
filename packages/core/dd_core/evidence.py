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

import os
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

def looks_like_profile(d: Path) -> bool:
    """Does `d` look like a Chromium/Electron profile directory? Shared by the
    live collector (`dd_collector.discover`) and image-based extraction
    (`dd_lab.imaging.discover`) — same signature files, whether `d` is a real
    directory or a path into an opened disk image."""
    if (d / "Local State").is_file():
        return True
    has_ls = (d / "Local Storage" / "leveldb").is_dir()
    has_cache = (d / "Cache").is_dir() or (d / "Cache" / "Cache_Data").is_dir()
    return has_ls or has_cache


_WEBVIEW2_MARKER = "EBWebView"


def find_webview2_profiles(root: Path, *, max_depth: int = 6) -> list[Path]:
    """Chromium profiles embedded via Microsoft Edge WebView2, nested
    somewhere under `root` rather than at `root` itself.

    A growing number of "not Electron" native Windows apps still embed a
    WebView2 control for part of their UI (an in-app browser, an auth or
    payment flow, ...) — and WebView2 keeps its own Chromium profile at
    `<some app-chosen folder>/EBWebView/<profile-name>/...` (almost always
    profile-name "Default"), completely separate from, and often much
    deeper than, where an Electron app's profile would sit. `looks_like_
    profile()` alone misses these entirely, since it only ever checks the
    directory it's given, not its descendants.

    Confirmed on real evidence: Roblox's desktop client has no Electron
    profile of its own (it isn't Electron), but its embedded WebView2
    instance has a full, real profile — Local Storage, Cache, IndexedDB —
    nested at `Roblox/UniversalApp/WebView2/EBWebView/Default/...` and
    again per-version at `Roblox/Versions/<hash>/<exe>.WebView2/EBWebView/
    Default/...`. It contains genuine forensic content (payment/identity-
    verification domains among them) a top-level-only scan never reaches.

    Depth-bounded (an app can keep several old versioned copies around,
    e.g. Roblox's `Versions/` folder) rather than an unbounded recursive
    glob, to keep this a contained cost per already-matched candidate app
    root — not a scan of the whole image.
    """
    found: list[Path] = []

    def walk(d: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            children = list(d.iterdir())
        except OSError:
            return
        for child in children:
            try:
                if not child.is_dir():
                    continue
            except OSError:
                continue
            if child.name == _WEBVIEW2_MARKER:
                for prof in _safe_iterdir(child):
                    try:
                        if prof.is_dir() and looks_like_profile(prof):
                            found.append(prof)
                    except OSError:
                        continue
                continue  # nothing profile-shaped lives any deeper inside EBWebView itself
            walk(child, depth + 1)

    walk(root, 0)
    return found


def _safe_iterdir(d: Path) -> list[Path]:
    try:
        return list(d.iterdir())
    except OSError:
        return []


_MSIX_REDIRECT_ROOTS = ("Roaming", "Local")


def iter_profiles_for_entry(entry: Path) -> list[tuple[str, Path]]:
    """Every (name, profile_dir) found at or under one candidate directory -
    `entry` itself, one level nested (`<name>/<name>`, some apps do this),
    an MSIX/UWP Store-package redirect (`entry/LocalCache/{Roaming,Local}/
    *`), or embedded via WebView2 (possibly more than one, if the app keeps
    several versioned copies around).

    The one place "is this candidate actually a profile, and exactly
    where" is decided - the live collector (`dd_collector.discover`),
    image-based discovery (`dd_lab.imaging.discover`), and folder-based
    import (`dd_lab.imaging.acquire::discover_apps_in_folder`) all call
    this instead of each re-implementing the same checks, so a fix or an
    addition (WebView2 support, this MSIX check) lands in exactly one
    place for all three acquisition paths at once.

    The MSIX check is a second, independent path to the same layout the
    image/live-collector discovery globs already anchor directly at
    (`Users/*/AppData/Local/Packages/*/LocalCache/{Roaming,Local}/*`) - it
    matters here because a *flat* evidence folder (one subfolder per app,
    no `AppData/Local/Packages/` prefix to glob against) has no such fixed
    prefix to anchor on; confirmed on real evidence, a Store-packaged
    Slack folder copied on its own (`<folder>/<PackageFamilyName>/...`)
    was invisible to a flat-folder scan without this.
    """
    if looks_like_profile(entry):
        return [(entry.name, entry)]
    nested = entry / entry.name
    if nested.is_dir() and looks_like_profile(nested):
        return [(entry.name, nested)]
    found = []
    for root_name in _MSIX_REDIRECT_ROOTS:
        redirect_root = entry / "LocalCache" / root_name
        if not redirect_root.is_dir():
            continue
        for child in _safe_iterdir(redirect_root):
            try:
                if child.is_dir() and looks_like_profile(child):
                    found.append((child.name, child))
            except OSError:
                continue
    if found:
        return found
    for wv in find_webview2_profiles(entry):
        via = str(wv.parent.parent.relative_to(entry))
        found.append((f"{entry.name} (WebView2 via {via})", wv))
    return found


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

    `profile_dir` may be a plain `pathlib.Path` (live collection) or any other
    `Path` subclass backed by a different filesystem — e.g. `dissect`'s
    `TargetPath`, reading straight out of a disk image (see `dd_lab.imaging`).
    Only a plain `str` gets wrapped; an already-Path-like object is used as-is,
    since re-wrapping it in `Path(...)` would discard whatever non-local
    filesystem it's actually backed by.
    """
    profile = profile_dir if isinstance(profile_dir, Path) else Path(profile_dir)
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


def _stream_copy(src: Path, dst: Path) -> None:
    """Copy `src` to `dst` by reading/writing through the `Path` API rather
    than `shutil.copy2` (which opens the source by its OS path string via the
    `open()` builtin) — this is the only way that also works when `src` is a
    non-local `Path` subclass such as `dissect`'s `TargetPath`, reading out of
    a disk image rather than the real filesystem. Behaviourally identical to
    `copy2` for a plain local path: content plus, best-effort, mtime."""
    with src.open("rb") as fsrc, open(dst, "wb") as fdst:
        shutil.copyfileobj(fsrc, fdst)
    try:
        st = src.stat()
        os.utime(dst, (st.st_atime, st.st_mtime))
    except (OSError, NotImplementedError, AttributeError):
        pass  # best-effort metadata; content is what matters forensically


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
                _stream_copy(entry, target)
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
                _stream_copy(spec.source, dest)
                report.written.append(dest)
            except (PermissionError, OSError, shutil.SameFileError) as e:
                report.skipped.append((str(spec.source), str(e)))
    return report
