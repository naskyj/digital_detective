"""Canonical evidence layout for one Electron app, shared by collector and lab.

Under <case>/evidence/<app>/ the collector writes a fixed structure so the lab
parsers always know where to look, regardless of the source OS or app:

    local_storage/leveldb/      <- <profile>/Local Storage/leveldb
    session_storage/            <- <profile>/Session Storage
    indexeddb/                  <- <profile>/IndexedDB/*   (leveldb + blob dirs)
    cache/                      <- <profile>/Cache  (or Cache/Cache_Data)
    cookies                     <- <profile>/Cookies  (or Network/Cookies)
    preferences.json            <- <profile>/Preferences
    local_state.json            <- <profile-parent>/Local State
    memory/                     <- process / full memory dumps
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

# relative destinations under <case>/evidence/<app>/
LOCAL_STORAGE = "local_storage/leveldb"
SESSION_STORAGE = "session_storage"
INDEXEDDB = "indexeddb"
CACHE = "cache"
COOKIES = "cookies"
PREFERENCES = "preferences.json"
LOCAL_STATE = "local_state.json"
MEMORY = "memory"


@dataclass
class CopySpec:
    """One source -> destination copy the collector performs for an app profile."""

    source: Path
    dest_rel: str
    is_dir: bool


ALL_SOURCES = frozenset(
    {"local_storage", "session_storage", "indexeddb", "cache", "cookies", "preferences", "local_state"}
)


def plan_profile_copy(
    profile_dir: str | Path, include: frozenset[str] | None = None
) -> list[CopySpec]:
    """Given a live Chromium/Electron profile directory, return the copies to make.

    Only entries whose source exists are returned. `include` restricts which
    sources are copied (default: all). `Local State` lives one level above the
    profile in a real Chromium install but sits inside it for many Electron
    apps, so both locations are checked.
    """
    profile = Path(profile_dir)
    want = ALL_SOURCES if include is None else include
    specs: list[CopySpec] = []

    def add(src: Path, dest_rel: str, is_dir: bool, source: str) -> None:
        if source in want and src.exists():
            specs.append(CopySpec(src, dest_rel, is_dir))

    add(profile / "Local Storage" / "leveldb", LOCAL_STORAGE, True, "local_storage")
    add(profile / "Session Storage", SESSION_STORAGE, True, "session_storage")
    add(profile / "IndexedDB", INDEXEDDB, True, "indexeddb")

    cache_src = profile / "Cache" / "Cache_Data"
    if not cache_src.is_dir():
        cache_src = profile / "Cache"
    add(cache_src, CACHE, True, "cache")

    cookies_src = profile / "Network" / "Cookies"
    if not cookies_src.is_file():
        cookies_src = profile / "Cookies"
    add(cookies_src, COOKIES, False, "cookies")

    add(profile / "Preferences", PREFERENCES, False, "preferences")

    local_state = profile / "Local State"
    if not local_state.is_file():
        local_state = profile.parent / "Local State"
    add(local_state, LOCAL_STATE, False, "local_state")

    return specs


_SOURCE_TYPE_PREFIXES = (
    ("local_storage", "local_storage"),
    ("session_storage", "session_storage"),
    ("indexeddb", "indexeddb"),
    ("cache", "cache"),
    ("cookies", "cookies"),
    ("preferences.json", "preferences"),
    ("local_state.json", "local_state"),
    ("memory", "memory_process"),
)


def source_type_for_path(evidence_relative_path: str) -> str:
    """Map an evidence path like 'slack/cache/Cache_Data/f_00001' to a SourceType value.

    Returns the string form (e.g. 'cache'); callers can pass it to SourceType(...).
    """
    parts = evidence_relative_path.replace("\\", "/").split("/")
    tail = "/".join(parts[1:]) if len(parts) > 1 else evidence_relative_path
    for prefix, source_type in _SOURCE_TYPE_PREFIXES:
        if tail.startswith(prefix):
            return source_type
    return "preferences"


def execute_copy(specs: list[CopySpec], app_evidence_dir: str | Path) -> list[Path]:
    """Copy each spec into <app_evidence_dir>. Returns the list of files written."""
    app_dir = Path(app_evidence_dir)
    written: list[Path] = []
    for spec in specs:
        dest = app_dir / spec.dest_rel
        if spec.is_dir:
            shutil.copytree(spec.source, dest, dirs_exist_ok=True)
            written.extend(p for p in dest.rglob("*") if p.is_file())
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(spec.source, dest)
            written.append(dest)
    return written
