"""Command-line interface for the collector.

    dd-collect --list
    dd-collect --app Discord --app Slack --output E:\\evidence --memory --operator "Det. A. Smith"
    dd-collect --app "C:\\path\\to\\profile" --output E:\\evidence
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dd_collector.acquire import acquire
from dd_collector.discover import discover_apps, find_app


def _cmd_list(args: argparse.Namespace) -> int:
    apps = discover_apps(args.search_dir)
    if not apps:
        print("No Electron applications found.")
        return 1
    width = max(len(a.name) for a in apps)
    for a in apps:
        run = f"  running (pids {', '.join(map(str, a.pids))})" if a.running else ""
        ver = f"  v{a.version}" if a.version else ""
        print(f"{a.name:<{width}}{ver}{run}\n{'':<{width}}  {a.profile_dir}")
    return 0


def _cmd_collect(args: argparse.Namespace) -> int:
    targets = []
    for spec in args.app:
        app = find_app(spec, args.search_dir)
        if app is None:
            print(f"error: no Electron app matching {spec!r}", file=sys.stderr)
            return 2
        targets.append(app)
    if not targets:
        print("error: specify at least one --app (or use --list)", file=sys.stderr)
        return 2

    key_pem = Path(args.signing_key).read_bytes() if args.signing_key else None

    result = acquire(
        targets,
        args.output,
        case_id=args.case_id,
        operator=args.operator,
        collect_memory=args.memory,
        signing_key_pem=key_pem,
        progress=lambda m: print(m, flush=True),
    )

    print("\n--- Acquisition complete ---")
    print(f"case id            : {result.case_id}")
    print(f"case directory     : {result.case_dir}")
    print(f"files / bytes      : {result.file_count} / {result.total_bytes:,}")
    print(f"applications       : {', '.join(result.apps)}")
    print(f"pubkey fingerprint : {result.public_key_fingerprint}")
    if result.warnings:
        print("\nwarnings:")
        for w in result.warnings:
            print(f"  - {w}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dd-collect", description="Digital Detective evidence collector")
    p.add_argument("--list", action="store_true", help="list detected Electron applications and exit")
    p.add_argument("--app", action="append", default=[], metavar="NAME|PATH",
                   help="application name substring or a profile directory path (repeatable)")
    p.add_argument("--output", "-o", metavar="DIR", help="output root (e.g. an external drive)")
    p.add_argument("--case-id", help="use a specific case id instead of a generated one")
    p.add_argument("--operator", default="", help="operator name recorded in the chain of custody")
    p.add_argument("--memory", action="store_true", help="also capture process / full memory")
    p.add_argument("--signing-key", metavar="PEM", help="Ed25519 private key PEM to sign with")
    p.add_argument("--search-dir", action="append", default=[], metavar="DIR",
                   help="extra directory to scan for profiles (repeatable)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        return _cmd_list(args)
    if not args.output:
        print("error: --output is required (or use --list)", file=sys.stderr)
        return 2
    return _cmd_collect(args)


if __name__ == "__main__":
    raise SystemExit(main())
