"""Extract evidence from a forensic disk image instead of a live collection.

Reads raw/dd images (including split segments), VMDK/VHD(X) virtual disks,
and E01/EWF containers directly — NTFS/FAT/exFAT, no mounting, no admin
rights — via Fox-IT's `dissect` stack (optional `image` extra). Produces the
exact same signed-case shape the live collector does, so everything
downstream (ingest, classify, correlate, findings, judge, report) is
unaffected by which acquisition path was used.

The live collector (`dd_collector`) remains the acquisition path when a disk
image isn't available or a live capture (including memory) is required.
"""

from __future__ import annotations

from dd_lab.imaging.acquire import ImageAcquisitionResult, acquire_from_image
from dd_lab.imaging.discover import ImageApp, discover_apps, find_app
from dd_lab.imaging.image import ImageUnavailable, open_image

__all__ = [
    "ImageAcquisitionResult",
    "ImageApp",
    "ImageUnavailable",
    "acquire_from_image",
    "discover_apps",
    "find_app",
    "open_image",
]
