"""Find Electron applications on this machine.

An Electron app keeps a Chromium *profile* directory under the OS's per-user
application-data area. We detect one by its signature files (`Local State`, or a
`Local Storage/leveldb` / `Cache` pair) and report enough for the operator to
choose targets: display name, profile path, a version guess, and any running
process IDs.

Windows is the primary target; macOS and Linux locations are included so the
tool can be developed and demoed off-Windows.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from pathlib import Path

from dd_core.evidence import iter_profiles_for_entry, looks_like_profile

_WINDOWS_ROOTS = ("APPDATA", "LOCALAPPDATA")


@dataclass
class ElectronApp:
    name: str
    profile_dir: Path
    version: str | None = None
    pids: list[int] = field(default_factory=list)
    running: bool = False

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "profile_dir": str(self.profile_dir),
            "version": self.version,
            "pids": self.pids,
            "running": self.running,
        }


def _candidate_roots() -> list[Path]:
    system = platform.system()
    roots: list[Path] = []
    if system == "Windows":
        for env in _WINDOWS_ROOTS:
            if val := os.environ.get(env):
                roots.append(Path(val))
    elif system == "Darwin":
        roots.append(Path.home() / "Library" / "Application Support")
    else:
        roots.append(Path.home() / ".config")
        if val := os.environ.get("XDG_CONFIG_HOME"):
            roots.append(Path(val))
    return [r for r in dict.fromkeys(roots) if r.is_dir()]


# Shared with image-based extraction (dd_lab.imaging.discover) so both agree
# on what counts as a profile.
_looks_like_profile = looks_like_profile


def _guess_version(profile_dir: Path) -> str | None:
    # Many Electron apps keep an "app-<version>" folder next to the profile,
    # or a package.json in a nearby resources dir.
    for parent in (profile_dir, profile_dir.parent):
        for child in _safe_iterdir(parent):
            if child.name.lower().startswith("app-") and child.name[4:5].isdigit():
                return child.name[4:]
    return None


def _safe_iterdir(d: Path):
    try:
        return list(d.iterdir())
    except (PermissionError, OSError):
        return []


def _running_electron() -> dict[str, list[int]]:
    """Map lower-cased process name -> pids, for likely Electron processes."""
    out: dict[str, list[int]] = {}
    try:
        import psutil
    except ImportError:
        return out
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            name = (proc.info["name"] or "").lower()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if name:
            out.setdefault(name.removesuffix(".exe"), []).append(proc.info["pid"])
    return out


def discover_apps(extra_dirs: list[str | Path] | None = None) -> list[ElectronApp]:
    roots = _candidate_roots()
    for extra in extra_dirs or []:
        p = Path(extra)
        if p.is_dir():
            roots.append(p)

    running = _running_electron()
    # Two separate sets deliberately: an `entry` that IS itself a profile
    # would otherwise get marked seen before iter_profiles_for_entry() ever
    # got to return `(entry.name, entry)` for it - silently dropping every
    # direct-match app the moment its containing directory was recorded.
    seen_entries: set[Path] = set()
    seen_profiles: set[Path] = set()
    apps: list[ElectronApp] = []

    for root in roots:
        for entry in sorted(_safe_iterdir(root)):
            if not entry.is_dir() or entry in seen_entries:
                continue
            seen_entries.add(entry)
            key = entry.name.lower().replace(" ", "")
            pids = next((v for k, v in running.items() if key in k or k in key), [])
            for name, profile in iter_profiles_for_entry(entry):
                if profile in seen_profiles:
                    continue
                seen_profiles.add(profile)
                apps.append(
                    ElectronApp(
                        name=name,
                        profile_dir=profile,
                        version=_guess_version(profile),
                        pids=pids,
                        running=bool(pids),
                    )
                )
    return apps


def find_app(name_or_path: str, extra_dirs: list[str | Path] | None = None) -> ElectronApp | None:
    p = Path(name_or_path)
    if p.is_dir() and _looks_like_profile(p):
        return ElectronApp(name=p.name, profile_dir=p)
    lowered = name_or_path.lower()
    matches = [a for a in discover_apps(extra_dirs) if lowered in a.name.lower()]
    return matches[0] if matches else None
