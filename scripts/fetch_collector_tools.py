"""Download the memory-acquisition tools the collector bundles.

Run this ONCE on the machine where you build ``dd-collect.exe`` (your build box
or investigator workstation) — **never on the evidence / suspect machine**. It
populates ``packaging/tools/`` so PyInstaller can embed the tools next to the
collector exe. After that the collector is fully self-contained: ``--memory``
works on an offline suspect laptop with nothing to download.

    python scripts/fetch_collector_tools.py            # fetch everything it can
    python scripts/fetch_collector_tools.py --list     # show what would be fetched
    python scripts/fetch_collector_tools.py --only procdump

What it fetches
--------------
* WinPmem   — full physical RAM  -> physmem.raw     (Apache-2.0, redistributable)
* ProcDump  — one process        -> pid_<n>.dmp     (Sysinternals EULA, see notes)

DumpIt (Comae/Magnet) has no stable public download and is gated behind a
signup, so it is not fetched automatically. If you have DumpIt.exe, drop it in
``packaging/tools/`` by hand — the collector picks it up as a WinPmem fallback.

Licensing note: ``packaging/tools/`` is git-ignored on purpose. WinPmem
(Apache-2.0) may be committed to the repo if you want a fully offline build.
ProcDump ships under the Sysinternals license, which restricts redistribution —
keep it out of the repo and let each build box fetch it here.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent.parent / "packaging" / "tools"
UA = {"User-Agent": "digital-detective-fetch-tools"}

WINPMEM_RELEASES = "https://api.github.com/repos/Velocidex/WinPmem/releases/latest"
WINPMEM_ASSET_RE = re.compile(r"winpmem.*x64.*\.exe$|winpmem_mini_x64.*\.exe$", re.I)
PROCDUMP_ZIP = "https://download.sysinternals.com/files/Procdump.zip"
PROCDUMP_MEMBERS = ("procdump64.exe", "procdump.exe")


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 - trusted hosts
        return r.read()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_winpmem() -> list[tuple[str, str, str]]:
    meta = json.loads(_get(WINPMEM_RELEASES))
    assets = meta.get("assets", [])
    match = next((a for a in assets if WINPMEM_ASSET_RE.search(a["name"])), None)
    if match is None:
        match = next((a for a in assets if a["name"].lower().endswith(".exe")), None)
    if match is None:
        raise RuntimeError("no WinPmem .exe asset found in the latest release")
    data = _get(match["browser_download_url"])
    (TOOLS_DIR / "winpmem.exe").write_bytes(data)
    return [("winpmem.exe", match["browser_download_url"], _sha256(data))]


def fetch_procdump() -> list[tuple[str, str, str]]:
    blob = _get(PROCDUMP_ZIP)
    out: list[tuple[str, str, str]] = []
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = {n.lower(): n for n in z.namelist()}
        for want in PROCDUMP_MEMBERS:
            real = names.get(want)
            if not real:
                continue
            data = z.read(real)
            (TOOLS_DIR / want).write_bytes(data)
            out.append((want, f"{PROCDUMP_ZIP}!{real}", _sha256(data)))
    if not out:
        raise RuntimeError("Procdump.zip did not contain the expected exe members")
    return out


FETCHERS = {"winpmem": fetch_winpmem, "procdump": fetch_procdump}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=sorted(FETCHERS), action="append",
                    help="fetch just this tool (repeatable); default: all")
    ap.add_argument("--list", action="store_true", help="print sources and exit")
    args = ap.parse_args(argv)

    wanted = args.only or list(FETCHERS)
    if args.list:
        print("Would fetch into", TOOLS_DIR)
        for name in wanted:
            print(f"  - {name}")
        return 0

    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    receipts: list[tuple[str, str, str]] = []
    failed: list[str] = []
    for name in wanted:
        print(f"fetching {name} …", flush=True)
        try:
            receipts += FETCHERS[name]()
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"  ! {name} failed: {e}", file=sys.stderr)
            failed.append(name)

    if receipts:
        lines = [
            "# packaging/tools/ contents",
            f"# fetched {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}"  # noqa: UP017 - 3.10 compat
            " by scripts/fetch_collector_tools.py",
            "",
        ]
        for fname, src, digest in receipts:
            lines += [f"{fname}", f"  source: {src}", f"  sha256: {digest}", ""]
        (TOOLS_DIR / "TOOLS.md").write_text("\n".join(lines))
        print("\nwrote:")
        for fname, _, digest in receipts:
            print(f"  {fname}  sha256={digest[:16]}…")
        print(f"\n{TOOLS_DIR} is ready — now build: uv run pyinstaller packaging/collector.spec")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
