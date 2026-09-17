"""Extract evidence from a forensic disk image - or a plain folder - instead
of a live collection.

Reads raw/dd images (including split segments), VMDK/VHD(X) virtual disks,
E01/EWF containers, AND folders shaped like an OS root directly — NTFS/FAT/
exFAT, no mounting, no admin rights — via Fox-IT's `dissect` stack (optional
`image` extra). `acquire_from_folder`/`discover_apps_in_folder` handle a
*different* folder shape: a plain directory holding one subfolder per
already-extracted app (no dissect involved at all - just `dd_core.evidence`'s
shared profile-matching logic on real filesystem paths). All three input
shapes produce the exact same signed-case shape the live collector does, so
everything downstream (ingest, classify, correlate, findings, judge, report)
is unaffected by which acquisition path was used.

None of this is "live collection" - it only ever reads static files that
already exist (an image, or evidence someone already extracted). The live
collector (`dd_collector`) remains the only path that touches a live,
running machine's own AppData paths, and a full-RAM/process memory capture.
"""

from __future__ import annotations

from dd_lab.imaging.acquire import ImageAcquisitionResult, acquire_from_folder, acquire_from_image
from dd_lab.imaging.discover import ImageApp, discover_apps, discover_apps_in_folder, find_app
from dd_lab.imaging.image import ImageUnavailable, open_image

__all__ = [
    "ImageAcquisitionResult",
    "ImageApp",
    "ImageUnavailable",
    "acquire_from_folder",
    "acquire_from_image",
    "discover_apps",
    "discover_apps_in_folder",
    "find_app",
    "open_image",
]
