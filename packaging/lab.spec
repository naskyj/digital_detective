# PyInstaller spec for the full investigator app (dd-lab).
# Build:  pyinstaller packaging/lab.spec
# Output: dist/dd-lab/  (one directory — ship the whole folder)
#
# Heavy optional extras (transformers/torch, volatility3, onnxruntime) are NOT
# bundled; install them into the environment before building if you need them.
# `dissect.target` (image-based acquisition, `--extra image`) is pure Python
# and modest in size — it IS bundled automatically if it was installed
# (`uv sync --package dd-lab --extra image`) before this build.

import os

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None
_root = os.path.dirname(SPECPATH)  # repo root

datas = [(os.path.join(_root, "packages/lab/dd_lab/ui"), "dd_lab/ui")]
datas += collect_data_files("ccl_chromium_reader")

hiddenimports = (
    collect_submodules("dd_lab")
    + collect_submodules("dd_collector")
    + collect_submodules("ccl_chromium_reader")
    + ["duckdb", "uvicorn", "uvicorn.logging", "uvicorn.protocols",
       "uvicorn.protocols.http.auto", "uvicorn.protocols.websockets.auto",
       "uvicorn.lifespan.on", "pydantic_ai.models.openai", "webview"]
)

a = Analysis(
    [os.path.join(SPECPATH, "lab_entry.py")],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    excludes=["torch", "transformers", "onnxruntime", "volatility3", "gliner"],
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True,
    name="dd-lab", console=True, upx=True,
)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, upx=True, name="dd-lab")
