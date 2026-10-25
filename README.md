# Digital Detective

Local, agent-assisted digital-forensics toolkit for **Electron desktop
applications** (Discord, Slack, Signal, Notion, Postman, VS Code, …).

Acquires evidence from a computer — Chromium disk cache, Local/Session Storage,
IndexedDB, cookies, JSON preferences, and process memory — then, on the
investigator's workstation, verifies it against a signed manifest, parses every
store into a queryable warehouse, classifies each artifact
(chat / credentials / payment / location / user / …), analyses memory dumps, and
produces a court-style HTML forensic report with an evidence timeline and a
full chain of custody. Everything runs on one machine; no cloud, no network
required. An optional local or hosted LLM can orchestrate the investigation.

## Quick start

Needs [`uv`](https://docs.astral.sh/uv/) and **Python 3.12 or 3.13** (not 3.14 —
a C-extension dependency has no 3.14 wheel yet; the repo pins 3.13 via
`.python-version`).

```bash
uv python install 3.13
uv sync --all-packages --extra chromium
uv run dd-lab                 # the investigator app (add --extra desktop for a native window)
uv run dd-collect --list      # the portable collector
```

Full walkthrough: [docs/HOW_TO_RUN.md](docs/HOW_TO_RUN.md).

## Layout

```
packages/core        shared: evidence schema, hashing, signing, manifest +
                     verification, chain-of-custody log, case layout
packages/collector   dd-collect — portable, read-only acquisition + signing
packages/lab         dd-lab — parsers · DuckDB warehouse · classifier ·
                     memory analysis · agent · report · web UI + API
model/               optional NLP classifier training track
packaging/           PyInstaller specs for the two executables
docs/                the documentation set
```
