# PyInstaller spec for the standalone collector — one file, minimal.
# Build:  pyinstaller packaging/collector.spec
# Output: dist/dd-collect(.exe)

# A bundled winpmem.exe / procdump.exe placed in packaging/tools/ is picked up
# automatically (see the binaries list). Without them the collector still
# acquires disk artifacts.

import os

block_cipher = None
_here = SPECPATH  # the packaging/ directory
tools_dir = os.path.join(_here, "tools")
extra_binaries = []
if os.path.isdir(tools_dir):
    for name in os.listdir(tools_dir):
        extra_binaries.append((os.path.join(tools_dir, name), "."))

a = Analysis(
    [os.path.join(_here, "collector_entry.py")],
    pathex=[],
    binaries=extra_binaries,
    datas=[],
    hiddenimports=["dd_collector.gui", "dd_collector.cli", "tkinter"],
    hookspath=[],
    excludes=["duckdb", "pydantic_ai", "fastapi", "transformers", "torch", "volatility3",
              "ccl_chromium_reader", "numpy", "pandas"],
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, a.binaries, a.zipfiles, a.datas, [],
    name="dd-collect",
    console=True,          # keep a console for --list / progress output
    upx=True,
    disable_windowed_traceback=False,
    onefile=True,
)
