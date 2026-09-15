"""dd_lab.imaging.acquire._image_host — the examined machine's identity must
come from the image, never from whatever machine is running the analysis.

Uses a plain stub instead of a real dissect Target so this runs everywhere,
no image-building tools required; test_imaging.py covers the real-image case.
"""

from __future__ import annotations

from types import SimpleNamespace

from dd_lab.imaging.acquire import _image_host


def test_reads_hostname_os_version_from_the_target():
    target = SimpleNamespace(hostname="LAPTOP-9N23RMUE", os="windows", version="Windows 11 Home (NT 10.0)")

    host = _image_host(target, "/path/to/case.E01", "a" * 64)

    assert host.hostname == "LAPTOP-9N23RMUE"
    assert host.os == "windows"
    assert host.os_version == "Windows 11 Home (NT 10.0)"
    assert host.source_image == "/path/to/case.E01"
    assert host.source_image_sha256 == "a" * 64


def test_falls_back_to_unknown_not_to_the_analysts_machine_when_unresolvable():
    """A target whose OS plugins can't determine these (unrecognised/partial
    image) must report "unknown", never silently fall through to whatever
    machine is running this code."""
    target = SimpleNamespace()  # no hostname/os/version attributes at all

    host = _image_host(target, "/path/to/case.raw", "b" * 64)

    assert "unknown" in host.hostname.lower()
    assert host.os == "unknown"
    assert host.os_version == "unknown"


def test_tolerates_a_target_that_raises_on_attribute_access():
    """dissect.target.Target raises its own exception type (not a plain
    AttributeError) for an unresolvable plugin function - must be caught,
    not propagated."""

    class RaisingTarget:
        def __getattr__(self, name):
            raise RuntimeError(f"no plugin for {name}")

    host = _image_host(RaisingTarget(), "/path/to/case.raw", "c" * 64)

    assert "unknown" in host.hostname.lower()
    assert host.os == "unknown"
