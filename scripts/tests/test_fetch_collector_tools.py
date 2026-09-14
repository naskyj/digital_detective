"""fetch_collector_tools.py — logic that doesn't need the network.

Import/zip handling, error messaging, and CLI plumbing are covered here without
hitting GitHub/Sysinternals; the actual download paths were verified manually
against the real hosts (see docs/FIELD_GUIDE.md / packaging/README.md).
"""

from __future__ import annotations

import importlib.util
import io
import ssl
import sys
import urllib.error
import zipfile
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "fetch_collector_tools.py"
_spec = importlib.util.spec_from_file_location("fetch_collector_tools", _SCRIPT)
fct = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fct
_spec.loader.exec_module(fct)


@pytest.fixture(autouse=True)
def isolated_tools_dir(tmp_path, monkeypatch):
    d = tmp_path / "tools"
    monkeypatch.setattr(fct, "TOOLS_DIR", d)
    d.mkdir()
    return d


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in entries.items():
            z.writestr(name, data)
    return buf.getvalue()


def test_import_dumpit_from_zip(tmp_path, isolated_tools_dir):
    zpath = tmp_path / "Magnet_DumpIt.zip"
    zpath.write_bytes(_zip_bytes({"Magnet DumpIt for Windows/DumpIt.exe": b"fake-dumpit"}))

    receipts = fct.import_dumpit(zpath)

    assert (isolated_tools_dir / "DumpIt.exe").read_bytes() == b"fake-dumpit"
    names = [r[0] for r in receipts]
    assert names == ["DumpIt.exe"]


def test_import_dumpit_from_zip_with_arm64(tmp_path, isolated_tools_dir):
    zpath = tmp_path / "Magnet_DumpIt.zip"
    zpath.write_bytes(_zip_bytes({
        "DumpIt.exe": b"x64",
        "DumpIt-arm64.exe": b"arm64",
    }))

    receipts = fct.import_dumpit(zpath)

    assert {r[0] for r in receipts} == {"DumpIt.exe", "DumpIt-arm64.exe"}
    assert (isolated_tools_dir / "DumpIt-arm64.exe").read_bytes() == b"arm64"


def test_import_dumpit_bare_exe(tmp_path, isolated_tools_dir):
    exe = tmp_path / "DumpIt.exe"
    exe.write_bytes(b"raw-exe")

    receipts = fct.import_dumpit(exe)

    assert (isolated_tools_dir / "DumpIt.exe").read_bytes() == b"raw-exe"
    assert receipts[0][1] == str(exe)  # source path recorded for the TOOLS.md receipt


def test_import_dumpit_zip_without_dumpit_raises(tmp_path):
    zpath = tmp_path / "not_dumpit.zip"
    zpath.write_bytes(_zip_bytes({"readme.txt": b"hi"}))
    with pytest.raises(RuntimeError, match="no DumpIt.exe"):
        fct.import_dumpit(zpath)


def test_import_dumpit_missing_path_raises(tmp_path):
    with pytest.raises(RuntimeError, match="not found"):
        fct.import_dumpit(tmp_path / "nope.zip")


def test_ssl_verify_failure_gets_actionable_message(monkeypatch):
    def boom(*a, **kw):
        raise urllib.error.URLError(ssl.SSLCertVerificationError("certificate verify failed"))

    monkeypatch.setattr(fct.urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError, match="certifi|Install Certificates|--insecure"):
        fct._get("https://example.invalid/file")


def test_list_flag_is_a_dry_run(isolated_tools_dir, capsys):
    rc = fct.main(["--list"])
    assert rc == 0
    assert not any(isolated_tools_dir.iterdir())
    out = capsys.readouterr().out
    assert "winpmem" in out and "procdump" in out and "dumpit" in out


def test_only_filters_to_one_fetcher(monkeypatch, isolated_tools_dir):
    called = []
    monkeypatch.setitem(fct.FETCHERS, "winpmem", lambda: called.append("winpmem") or [])
    monkeypatch.setitem(fct.FETCHERS, "procdump", lambda: called.append("procdump") or [])

    fct.main(["--only", "winpmem"])

    assert called == ["winpmem"]
