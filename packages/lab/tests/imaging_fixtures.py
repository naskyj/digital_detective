"""Shared helper: build a real raw FAT32 disk image for imaging tests.

Uses macOS's `hdiutil`/`newfs_msdos`/`diskutil` (all root-free) to create,
format, populate, and detach a real `.img` file — the image is never mounted
by the time a test hands it to `dissect`, matching how an investigator's own
raw/dd image would be read. Not a mock: this is the same integration path a
real acquisition takes.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

HAVE_MAC_IMAGE_TOOLS = (
    sys.platform == "darwin"
    and bool(shutil.which("hdiutil"))
    and bool(shutil.which("newfs_msdos"))
    and bool(shutil.which("diskutil"))
)
SKIP_REASON = "needs hdiutil/newfs_msdos/diskutil (macOS) to build a test image"


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60).stdout.strip()


def _volume_name(device: str) -> str:
    info = _run(["diskutil", "info", device])
    for line in info.splitlines():
        if line.strip().startswith("Volume Name"):
            return line.split(":", 1)[1].strip()
    return "NO NAME"


def build_fat_image(dest: Path) -> Path:
    """A real 64MB raw FAT32 disk image (FAT32 needs at least ~33MB) at
    `dest`, with a planted Discord-like profile under
    `Users/Suspect/AppData/Roaming/Discord/...`."""
    subprocess.run(
        ["dd", "if=/dev/zero", f"of={dest}", "bs=1m", "count=64"],
        check=True, capture_output=True, timeout=60,
    )
    device = _run(["hdiutil", "attach", "-nomount", str(dest)]).splitlines()[0].split()[0]
    try:
        subprocess.run(["newfs_msdos", "-F", "32", device], check=True, capture_output=True, timeout=60)
        subprocess.run(["diskutil", "mount", device], check=True, capture_output=True, timeout=60)
        mount_point = Path(f"/Volumes/{_volume_name(device)}")
        _plant_profile(mount_point)
        subprocess.run(["diskutil", "unmount", device], check=True, capture_output=True, timeout=60)
    finally:
        time.sleep(0.2)  # give the OS a beat before detach
        subprocess.run(["hdiutil", "detach", device], capture_output=True, timeout=60)
    return dest


def _plant_profile(mount_point: Path) -> None:
    profile = mount_point / "Users" / "Suspect" / "AppData" / "Roaming" / "Discord"
    (profile / "Local Storage" / "leveldb").mkdir(parents=True)
    (profile / "Local State").write_text('{"os_crypt":{"encrypted_key":"abc"}}')
    (profile / "Preferences").write_text('{"token":"discordapp.com faketoken.value.here"}')
    (profile / "Local Storage" / "leveldb" / "000003.log").write_bytes(
        b"authorId 123456789012345678 rbxassetid://9988776655"
    )
