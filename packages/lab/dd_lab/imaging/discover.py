"""Find Electron app profiles inside an opened disk image, or in a plain
folder that already holds one subfolder per app.

Mirrors `dd_collector.discover`'s live-machine scan but walks the *image's*
own directory tree instead of the current process's environment, since the
image may be a different OS and user set than the machine running the lab.
A static image has no running processes, so unlike the live collector there
is no `running`/`pids` on the result — a profile is either present or it
isn't.

`discover_apps_in_folder` is the third of three discovery entry points
(live collector, image, folder) that all end up calling `dd_core.evidence
.iter_profiles_for_entry` for the actual "is this a profile" decision — the
only thing that differs between them is what candidate directories they
hand it: a live machine's real AppData roots, an opened image's `Users/*/
AppData/*` globs, or (here) a plain folder's own immediate children. It
exists because `dd-lab` never collects evidence live from the machine it
runs on (that's `dd_collector`'s job) but a directory of evidence someone
already extracted — by hand, by another tool, from a mounted image - isn't
"live collection"; it's the same kind of already-static input an image is,
just shaped differently.

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
from pathlib import Path
from typing import TYPE_CHECKING

from dd_core.evidence import iter_profiles_for_entry

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
    # see ElectronApp.native_root (dd_collector.discover) / iter_profiles_
    # for_entry's own docstring - the app's own root when its profile was
    # found nested well inside it (WebView2/MSIX), None for a normal
    # Electron app whose profile already covers everything.
    native_root: TargetPath | None = None


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
    # Two separate sets deliberately: an `entry` that IS itself a profile
    # would otherwise get marked seen before iter_profiles_for_entry() ever
    # got to return it - silently dropping every direct-match app the
    # moment its containing directory was recorded.
    seen_entries: set[str] = set()
    seen_profiles: set[str] = set()
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
            if key in seen_entries:
                continue
            seen_entries.add(key)
            for name, profile, native_root in iter_profiles_for_entry(entry):
                pkey = str(profile).lower()
                if pkey in seen_profiles:
                    continue
                seen_profiles.add(pkey)
                apps.append(ImageApp(name=name, profile_dir=profile, version=_guess_version(profile),
                                      native_root=native_root))
    return apps


def find_app(target: Target, name_or_path: str) -> ImageApp | None:
    lowered = name_or_path.lower()
    matches = [a for a in discover_apps(target) if lowered in a.name.lower()]
    return matches[0] if matches else None


def discover_apps_in_folder(folder: Path) -> list[ImageApp]:
    """Scan a plain folder's immediate children for Chromium/Electron
    profiles - the folder itself is NOT an image or an OS root (dissect's
    own directory loader, reached via `open_image`, already handles that
    shape); this is for a folder that holds one subfolder per already-
    extracted app instead, e.g. `<somewhere>/discord/`, `<somewhere>/
    Roblox/`, ... side by side."""
    seen_entries: set[str] = set()
    seen_profiles: set[str] = set()
    apps: list[ImageApp] = []
    for entry in sorted(_safe_iterdir(folder)):
        try:
            if not entry.is_dir():
                continue
        except OSError:
            continue
        key = str(entry).lower()
        if key in seen_entries:
            continue
        seen_entries.add(key)
        for name, profile, native_root in iter_profiles_for_entry(entry):
            pkey = str(profile).lower()
            if pkey in seen_profiles:
                continue
            seen_profiles.add(pkey)
            apps.append(ImageApp(name=name, profile_dir=profile, version=_guess_version(profile),
                                  native_root=native_root))
    return apps
