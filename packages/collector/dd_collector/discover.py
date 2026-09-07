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


def _looks_like_profile(d: Path) -> bool:
    if (d / "Local State").is_file():
        return True
    has_ls = (d / "Local Storage" / "leveldb").is_dir()
    has_cache = (d / "Cache").is_dir() or (d / "Cache" / "Cache_Data").is_dir()
    return has_ls or has_cache


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
    seen: set[Path] = set()
    apps: list[ElectronApp] = []

    for root in roots:
        for entry in sorted(_safe_iterdir(root)):
            if not entry.is_dir() or entry in seen:
                continue
            profile = entry
            if not _looks_like_profile(profile):
                # some apps nest the profile one level down (e.g. <name>/<name>)
                nested = entry / entry.name
                if nested.is_dir() and _looks_like_profile(nested):
                    profile = nested
                else:
                    continue
            seen.add(entry)
            key = entry.name.lower().replace(" ", "")
            pids = next((v for k, v in running.items() if key in k or k in key), [])
            apps.append(
                ElectronApp(
                    name=entry.name,
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
