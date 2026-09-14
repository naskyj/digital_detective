"""Command-line interface for image-based extraction.

    dd-lab-image --list --image E:\\case.E01
    dd-lab-image --image E:\\case.raw --output cases\\ --app Discord --app Slack
    dd-lab-image --image E:\\case.vmdk --output cases\\ --operator "Det. A. Smith"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dd_lab.imaging.acquire import acquire_from_image
from dd_lab.imaging.discover import discover_apps
from dd_lab.imaging.image import ImageUnavailable, open_image


def _cmd_list(args: argparse.Namespace) -> int:
    try:
        target = open_image(args.image)
        apps = discover_apps(target)
    except ImageUnavailable as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if not apps:
        print("No Electron application profiles found on this image.")
        return 1
    width = max(len(a.name) for a in apps)
    for a in apps:
        ver = f"  v{a.version}" if a.version else ""
        print(f"{a.name:<{width}}{ver}\n{'':<{width}}  {a.profile_dir}")
    return 0


def _cmd_extract(args: argparse.Namespace) -> int:
    if not args.output:
        print("error: --output is required (or use --list)", file=sys.stderr)
        return 2
    key_pem = Path(args.signing_key).read_bytes() if args.signing_key else None
    try:
        result = acquire_from_image(
            args.image, args.output,
            apps=args.app or None,
            memory_image_path=args.memory_image,
            case_id=args.case_id, operator=args.operator, signing_key_pem=key_pem,
            progress=lambda m: print(m, flush=True),
        )
    except ImageUnavailable as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print("\n--- Extraction complete ---")
    print(f"case id            : {result.case_id}")
    print(f"case directory     : {result.case_dir}")
    print(f"files / bytes      : {result.file_count} / {result.total_bytes:,}")
    print(f"applications       : {', '.join(result.apps)}")
    print(f"image sha256       : {result.image_sha256}")
    print(f"pubkey fingerprint : {result.public_key_fingerprint}")
    if result.warnings:
        print("\nwarnings:")
        for w in result.warnings:
            print(f"  - {w}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dd-lab-image", description="Digital Detective — extract evidence from a disk image"
    )
    p.add_argument("--image", required=True, metavar="PATH",
                   help="disk image: raw/.dd/.001, .vmdk, .vhd(x), or .E01")
    p.add_argument("--list", action="store_true", help="list Electron app profiles found and exit")
    p.add_argument("--app", action="append", default=[], metavar="NAME",
                   help="app name substring to extract (repeatable); default: every profile found")
    p.add_argument("--output", "-o", metavar="DIR", help="cases root to write the new case into")
    p.add_argument("--case-id", help="use a specific case id instead of a generated one")
    p.add_argument("--operator", default="", help="operator name recorded in the chain of custody")
    p.add_argument("--memory-image", metavar="PATH",
                   help="a separately acquired full-RAM capture to fold into this case")
    p.add_argument("--signing-key", metavar="PEM", help="Ed25519 private key PEM to sign with")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        return _cmd_list(args)
    return _cmd_extract(args)


if __name__ == "__main__":
    raise SystemExit(main())
