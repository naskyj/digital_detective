# Packaging

Two build targets. Build each **on the OS you will run it on** (PyInstaller does
not cross-compile) — for real use that means building on Windows.

## Prerequisites

```bash
uv sync --all-packages --extra chromium        # + --extra desktop for the lab window
uv pip install pyinstaller
```

## 1. Collector — `dist/dd-collect.exe` (one file)

Minimal: `dd_core` + `dd_collector` + `psutil` + stdlib Tkinter. No LLM, no
DuckDB, no parsers.

```bash
uv run pyinstaller packaging/collector.spec
```

Run it:

- double-click → GUI
- `dd-collect.exe --list` → enumerate Electron apps
- `dd-collect.exe --app Discord --output E:\evidence --memory --operator "Det. Smith"`

### Bundling memory-*acquisition* tools (optional)

RAM is volatile — it can only be captured live on the examined machine, and
Windows needs a kernel driver to read physical memory. These tools do that
capture; they are **not** analysis tools (Volatility runs later, on the
investigator machine — see step 2). They are separate from Volatility and
serve the opposite end of the workflow.

Fetch them into `packaging/tools/` with the helper script — **run it on this
build machine, never on the evidence laptop** — then build:

```bash
python scripts/fetch_collector_tools.py        # WinPmem + ProcDump → packaging/tools/
uv run pyinstaller packaging/collector.spec    # embeds them next to dd-collect.exe
```

| Tool | Captures → | Source | License |
|---|---|---|---|
| `winpmem.exe` | full physical RAM → `physmem.raw` | github.com/Velocidex/WinPmem | Apache-2.0 (vendorable) |
| `procdump.exe` | one process → `pid_<n>.dmp` | download.sysinternals.com | Sysinternals EULA (don't commit) |
| `DumpIt.exe` | full RAM (WinPmem fallback) | Magnet/Comae (manual download) | — |

`packaging/tools/` is git-ignored so build binaries never bloat the repo. WinPmem
is Apache-2.0 — if you want a fully offline checkout, `git add -f
packaging/tools/winpmem.exe` after fetching. Keep ProcDump/DumpIt out (their
licenses restrict redistribution); each build box re-runs the script.

Without any of these, `--memory` still collects all disk artifacts and records
that memory capture was skipped. For a disk-only collection you don't need them.

## 2. Investigator app — `dist/dd-lab/` (one directory)

The full thing: collector + parsers + DuckDB warehouse + classifier + agent +
report + web UI in a pywebview window.

```bash
uv run pyinstaller packaging/lab.spec
```

Ship the whole `dist/dd-lab/` folder. Launch `dd-lab/dd-lab.exe` — it starts the
local server and opens the window (falls back to the default browser at
`http://127.0.0.1:8765`).

### Heavy optional extras

`transformers` / `torch` (zero-shot classifier), `onnxruntime` (trained model)
and `volatility3` (memory *analysis* — reads the dumps the collector captured;
the counterpart to winpmem/procdump above) are **excluded** from the bundle to
keep it small. If you need them baked in, remove the matching name from
`excludes` in `packaging/lab.spec`, `uv sync … --extra zeroshot --extra memory`,
and rebuild — expect the bundle to grow by 1–3 GB.

> Without `volatility3`, the lab still runs **string carving** on any dump
> (tokens, chat text, URLs — no dependency). Volatility adds the structured OS
> view (processes, connections) and only for a full physical-memory image.

For most deployments, keep them out and let the machine that needs them install
them into a normal Python environment instead.

## Notes

- Antivirus / SmartScreen may flag a fresh unsigned PyInstaller exe. Code-sign
  the binaries for distribution.
- The lab app writes cases and settings under the per-user config dir
  (`%APPDATA%\DigitalDetective` on Windows); the exe location doesn't matter.
- `uv run pyinstaller` uses the workspace venv, so all three packages resolve.
