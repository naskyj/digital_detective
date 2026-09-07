"""Frozen entry point for the full investigator application."""

from __future__ import annotations

from dd_lab.api.main import run

if __name__ == "__main__":
    run(open_window=True)
