"""Open a forensic disk image so the lab can read files out of it directly.

Backed by Fox-IT's `dissect` stack (pure Python, no admin/root rights, no
mounting): raw/dd images (including split `.001`/`.002` segments), VMDK/VHD(X)
virtual disks, and — for free, as the same library — E01/EWF containers, over
NTFS/FAT/exFAT filesystems.

Optional `image` extra (`uv sync --package dd-lab --extra image`); everything
here raises `ImageUnavailable` with a clear message if it isn't installed or
the file doesn't look like an image `dissect` recognises — the same
graceful-degradation pattern as `dd_lab.memory`'s Volatility support: a
missing capability is reported, never a crash.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dissect.target import Target


class ImageUnavailable(RuntimeError):
    """The `image` extra isn't installed, or the file isn't a recognisable image."""


def open_image(path: str | Path) -> Target:
    """Open a disk image and return the `dissect` Target — a merged view of
    every filesystem/volume found on it, addressable like a single rooted
    filesystem via `target.fs.path(...)`."""
    try:
        from dissect.target import Target
        from dissect.target.exceptions import TargetError
    except ImportError as e:
        raise ImageUnavailable(
            "the 'image' extra is not installed — uv sync --package dd-lab --extra image"
        ) from e

    p = Path(path)
    if not p.exists():
        raise ImageUnavailable(f"{p} does not exist")
    try:
        return Target.open(str(p))
    except TargetError as e:
        raise ImageUnavailable(f"could not open {p} as a disk image: {e}") from e
    except Exception as e:  # noqa: BLE001 - dissect raises assorted container-specific errors
        raise ImageUnavailable(f"could not open {p} as a disk image: {e}") from e
