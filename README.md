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

## Documentation

Start with **[docs/README.md](docs/README.md)** — it indexes the doc set:

| | |
|---|---|
| [docs/OVERVIEW.md](docs/OVERVIEW.md) | what it is, the shape of the system |
| [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) | one investigation traced end to end |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | every component, data structure, schema |
| [docs/HOW_TO_RUN.md](docs/HOW_TO_RUN.md) | install, collect, run a case, read the report |
| [docs/FIELD_GUIDE.md](docs/FIELD_GUIDE.md) | examine a Windows PC from a Mac — concrete steps |
| [docs/LLM_SETUP.md](docs/LLM_SETUP.md) | configure the agent's model (Ollama / API) |
| [docs/MODEL_TRAINING.md](docs/MODEL_TRAINING.md) | the optional NLP classifier training track |
| [docs/CLOUD_TO_LOCAL.md](docs/CLOUD_TO_LOCAL.md) | maps the earlier Google-Cloud build to this one |
| [PLAN.md](PLAN.md) | the original build plan and technology rationale |

## Quick start

```bash
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
