"""Frozen entry point for the standalone collector.

No args -> GUI. Any args -> CLI (`--list`, `--app`, ...).
"""

from __future__ import annotations

import sys

from dd_collector.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
