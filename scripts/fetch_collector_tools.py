"""Populate ``packaging/tools/`` with the memory-acquisition tools the collector
bundles.

Run this ONCE on the machine where you build ``dd-collect.exe`` (your build box
or investigator workstation) — **never on the evidence / suspect machine**.
PyInstaller then embeds the tools next to the collector exe, so ``--memory``
works on an offline suspect laptop with nothing to download.

    python scripts/fetch_collector_tools.py                 # fetch what it can
    python scripts/fetch_collector_tools.py --list          # show sources, exit
    python scripts/fetch_collector_tools.py --only procdump
    python scripts/fetch_collector_tools.py --dumpit ~/Downloads/Magnet_DumpIt.zip
    python scripts/fetch_collector_tools.py --insecure      # skip TLS verify (see below)

Tools
-----
* DumpIt   — full physical RAM  (Magnet DumpIt for Windows). PREFERRED by the
             collector. Form-gated download — cannot be fetched automatically.
             Download it yourself from
             https://www.magnetforensics.com/resources/magnet-dumpit-for-windows/
             then import the .zip/.exe with ``--dumpit <path>``.
* WinPmem  — full physical RAM, fallback if DumpIt is absent. Apache-2.0,
             fetched automatically from the Velocidex GitHub releases.
* ProcDump — one process -> pid_<n>.dmp. Fetched from download.sysinternals.com.

TLS / SSL errors
----------------
"CERTIFICATE_VERIFY_FAILED" almost always means Python has no CA bundle (common
with the python.org macOS build). Fixes, best first:
  1. pip install certifi          — this script uses it automatically if present
  2. run "/Applications/Python 3.x/Install Certificates.command"
  3. --insecure                   — skips verification; the script still prints
                                    each file's SHA-256 so you can verify it
                                    against a known-good value out of band.

Licensing: ``packaging/tools/`` is git-ignored. WinPmem (Apache-2.0) may be
committed for a fully offline checkout (``git add -f``). ProcDump (Sysinternals
EULA) and DumpIt (Magnet EULA) restrict redistribution — keep them out of the
repo; each build box re-runs this script / re-imports DumpIt.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import ssl
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent.parent / "packaging" / "tools"
UA = {"User-Agent": "digital-detective-fetch-tools"}

WINPMEM_RELEASES = "https://api.github.com/repos/Velocidex/WinPmem/releases/latest"
WINPMEM_ASSET_RE = re.compile(r"winpmem_mini_x64.*\.exe$", re.I)
PROCDUMP_ZIP = "https://download.sysinternals.com/files/Procdump.zip"
PROCDUMP_MEMBERS = ("procdump64.exe", "procdump.exe")

_INSECURE = False


def _ctx() -> ssl.SSLContext | None:
    if _INSECURE:
        c = ssl.create_default_context()
        c.check_hostname = False
        c.verify_mode = ssl.CERT_NONE
        return c
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ModuleNotFoundError:
        return None  # urllib uses the system default store


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={**UA, "Accept": "*/*"})
    try:
        with urllib.request.urlopen(req, timeout=90, context=_ctx()) as r:  # noqa: S310
            return r.read()
    except urllib.error.URLError as e:
        reason = getattr(e, "reason", e)
        if isinstance(reason, ssl.SSLCertVerificationError):
            raise RuntimeError(
                "TLS certificate verification failed. Fix: `pip install certifi`, or run "
                "the Python 'Install Certificates.command', or pass --insecure "
                "(hashes are still printed for out-of-band verification)."
            ) from e
        raise RuntimeError(f"download failed: {reason}") from e


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(name: str, data: bytes, src: str) -> tuple[str, str, str]:
    (TOOLS_DIR / name).write_bytes(data)
    return (name, src, _sha256(data))


def fetch_winpmem() -> list[tuple[str, str, str]]:
    meta = json.loads(_get(WINPMEM_RELEASES))
    assets = meta.get("assets", [])
    match = next((a for a in assets if WINPMEM_ASSET_RE.search(a["name"])), None) or next(
        (a for a in assets if a["name"].lower().endswith(".exe")), None
    )
    if match is None:
        raise RuntimeError("no WinPmem .exe asset in the latest release")
    return [_write("winpmem.exe", _get(match["browser_download_url"]), match["browser_download_url"])]


def fetch_procdump() -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    with zipfile.ZipFile(io.BytesIO(_get(PROCDUMP_ZIP))) as z:
        names = {n.lower(): n for n in z.namelist()}
        for want in PROCDUMP_MEMBERS:
            real = names.get(want)
            if real:
                out.append(_write(want, z.read(real), f"{PROCDUMP_ZIP}!{real}"))
    if not out:
        raise RuntimeError("Procdump.zip missing the expected exe members")
    return out


def import_dumpit(path: Path) -> list[tuple[str, str, str]]:
    """Copy a manually downloaded DumpIt (a .zip from Magnet, or DumpIt.exe) in."""
    if not path.exists():
        raise RuntimeError(f"{path} not found")
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            exe = next((n for n in z.namelist() if n.lower().endswith("dumpit.exe")), None)
            arm = next((n for n in z.namelist() if n.lower().endswith("dumpit-arm64.exe")), None)
            out = []
            if exe:
                out.append(_write("DumpIt.exe", z.read(exe), f"{path.name}!{exe}"))
            if arm:
                out.append(_write("DumpIt-arm64.exe", z.read(arm), f"{path.name}!{arm}"))
            if not out:
                raise RuntimeError(f"no DumpIt.exe inside {path.name}")
            return out
    data = path.read_bytes()
    return [_write("DumpIt.exe", data, str(path))]


FETCHERS = {"winpmem": fetch_winpmem, "procdump": fetch_procdump}


def main(argv: list[str] | None = None) -> int:
    global _INSECURE
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--only", choices=sorted(FETCHERS), action="append",
                    help="fetch just this tool (repeatable); default: all")
    ap.add_argument("--dumpit", metavar="ZIP_OR_EXE", type=Path,
                    help="import a manually downloaded Magnet DumpIt (.zip or DumpIt.exe)")
    ap.add_argument("--insecure", action="store_true",
                    help="skip TLS certificate verification (hashes still printed)")
    ap.add_argument("--list", action="store_true", help="print sources and exit")
    args = ap.parse_args(argv)
    _INSECURE = args.insecure

    wanted = args.only or list(FETCHERS)
    if args.list:
        print("Would populate", TOOLS_DIR)
        for name in wanted:
            print(f"  - {name}  (auto)")
        print("  - dumpit   (manual: --dumpit <path>, or drop DumpIt.exe in by hand)")
        return 0

    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    receipts: list[tuple[str, str, str]] = []
    failed: list[str] = []

    if args.dumpit:
        print(f"importing DumpIt from {args.dumpit} …", flush=True)
        try:
            receipts += import_dumpit(args.dumpit)
        except Exception as e:  # noqa: BLE001
            print(f"  ! dumpit import failed: {e}", file=sys.stderr)
            failed.append("dumpit")

    for name in wanted:
        print(f"fetching {name} …", flush=True)
        try:
            receipts += FETCHERS[name]()
        except Exception as e:  # noqa: BLE001
            print(f"  ! {name} failed: {e}", file=sys.stderr)
            failed.append(name)

    if receipts:
        _write_receipt(receipts)
        print("\nwrote:")
        for fname, _, digest in receipts:
            print(f"  {fname:<20} sha256={digest}")
        print(f"\n{TOOLS_DIR} ready — build with: uv run pyinstaller packaging/collector.spec")

    if "dumpit" not in [r[0].lower().split(".")[0] for r in receipts] and not args.dumpit:
        print("\nNote: DumpIt not imported. The collector will use WinPmem. To use DumpIt "
              "instead, download it from magnetforensics.com and re-run with "
              "--dumpit <path>.", file=sys.stderr)

    return 1 if failed else 0


def _write_receipt(receipts: list[tuple[str, str, str]]) -> None:
    existing = ""
    rec_path = TOOLS_DIR / "TOOLS.md"
    if rec_path.exists():
        existing = rec_path.read_text()
    stamp = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}"  # noqa: UP017
    lines = [f"## fetched {stamp} by scripts/fetch_collector_tools.py", ""]
    for fname, src, digest in receipts:
        lines += [f"- **{fname}**", f"  - source: `{src}`", f"  - sha256: `{digest}`"]
    lines += ["", existing] if existing else [""]
    rec_path.write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
