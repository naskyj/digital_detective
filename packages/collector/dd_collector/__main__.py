"""`python -m dd_collector` -> GUI; `python -m dd_collector.cli` -> CLI."""

from __future__ import annotations

import sys


def main() -> int:
    if len(sys.argv) > 1:
        from dd_collector.cli import main as cli_main

        return cli_main()
    from dd_collector.gui import main as gui_main

    return gui_main()


if __name__ == "__main__":
    raise SystemExit(main())
