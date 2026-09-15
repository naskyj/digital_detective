"""Find Electron app profiles inside an opened disk image.

Mirrors `dd_collector.discover`'s live-machine scan but walks the *image's*
own directory tree instead of the current process's environment, since the
image may be a different OS and user set than the machine running the lab.
A static image has no running processes, so unlike the live collector there
is no `running`/`pids` on the result — a profile is either present or it
isn't.

A real multi-partition Windows image is NOT a flat root: `target.fs.path("/")`
exposes each volume under its own prefix (`c:`, `efi`, `sysvol`, ...) —
confirmed against a real 48GB E01 (Windows, EFI + 2 NTFS partitions), where
`Users/...` at the bare root doesn't exist but `sysvol/Users/...` does.
`sysvol` is `dissect`'s own canonical alias for "the OS volume", regardless of
its actual drive letter, so it's preferred over guessing `c:`. A single-volume
image (a raw dd of one partition, or a mounted/extracted folder) usually *is*
flat, so bare patterns are tried too — an extra glob against a root that
doesn't exist just returns nothing, it isn't an error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from dd_core.evidence import looks_like_profile

if TYPE_CHECKING:
    from dissect.target import Target
    from dissect.target.helpers.fsutil import TargetPath

_ROOT_PREFIXES = ("", "sysvol/")  # "c:/" deliberately omitted: it's the same
# volume as "sysvol/" on a Windows target (dissect mounts both), so including
# it too would just double-list every profile found.
_USER_GLOBS = (
    "Users/*/AppData/Roaming/*",
    "Users/*/AppData/Local/*",
    # Microsoft Store (MSIX/UWP) installs redirect %APPDATA%/%LOCALAPPDATA%
    # into a per-package sandbox instead of the classic AppData/<App> layout
    # — confirmed on real evidence: a Store-installed Slack's entire Chromium
    # profile (Local State, Local Storage, IndexedDB, Cache, ...) lived only
    # at .../Packages/<PackageFamilyName>/LocalCache/Roaming/Slack, invisible
    # to the two patterns above.
    "Users/*/AppData/Local/Packages/*/LocalCache/Roaming/*",
    "Users/*/AppData/Local/Packages/*/LocalCache/Local/*",
    "Users/*/Library/Application Support/*",
    "home/*/.config/*",
)
_ROOT_GLOBS = tuple(prefix + glob for prefix in _ROOT_PREFIXES for glob in _USER_GLOBS)


@dataclass
class ImageApp:
    name: str
    profile_dir: TargetPath
    version: str | None = None


def _safe_iterdir(d: TargetPath) -> list[TargetPath]:
    try:
        return list(d.iterdir())
    except OSError:
        return []


def _guess_version(profile_dir: TargetPath) -> str | None:
    for parent in (profile_dir, profile_dir.parent):
        for child in _safe_iterdir(parent):
            if child.name.lower().startswith("app-") and child.name[4:5].isdigit():
                return child.name[4:]
    return None


def discover_apps(target: Target) -> list[ImageApp]:
    """Scan every user-profile root on the image for Chromium/Electron
    signatures (`Local State`, or a `Local Storage/leveldb` / `Cache` pair)."""
    root = target.fs.path("/")
    seen: set[str] = set()
    apps: list[ImageApp] = []

    for pattern in _ROOT_GLOBS:
        try:
            candidates = sorted(root.glob(pattern), key=str)
        except OSError:
            continue  # this root doesn't exist on this image (wrong OS, etc.)
        for entry in candidates:
            try:
                if not entry.is_dir():
                    continue
            except OSError:
                continue
            key = str(entry).lower()
            if key in seen:
                continue
            profile = entry
            if not looks_like_profile(profile):
                # some apps nest the profile one level down (<name>/<name>)
                nested = entry / entry.name
                if not (nested.is_dir() and looks_like_profile(nested)):
                    continue
                profile = nested
            seen.add(key)
            apps.append(ImageApp(name=entry.name, profile_dir=profile, version=_guess_version(profile)))
    return apps


def find_app(target: Target, name_or_path: str) -> ImageApp | None:
    lowered = name_or_path.lower()
    matches = [a for a in discover_apps(target) if lowered in a.name.lower()]
    return matches[0] if matches else None
