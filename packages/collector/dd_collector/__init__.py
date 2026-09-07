"""dd_collector - portable evidence acquisition for Electron applications."""

from __future__ import annotations

from dd_collector.acquire import AcquisitionResult, acquire
from dd_collector.discover import ElectronApp, discover_apps, find_app

__version__ = "0.1.0"

__all__ = ["AcquisitionResult", "ElectronApp", "acquire", "discover_apps", "find_app"]
