"""Application settings, persisted to a per-user config file.

Holds the cases root (where collected/imported cases live) and the LLM
configuration. Model *selection* is all that lives here; installing Ollama or
providing an API key is an environment/setup concern.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from pydantic import BaseModel

from dd_lab.agent.config import AgentConfig


def _config_dir() -> Path:
    # Explicit override wins on every platform — this is what tests use to
    # isolate from the real per-user config; on macOS in particular the
    # platform branch below ignores XDG_CONFIG_HOME entirely, so setting that
    # alone silently does nothing there and a test run ends up overwriting the
    # developer's real settings.json (found by actually running the app).
    if override := os.environ.get("DD_CONFIG_DIR"):
        d = Path(override) / "DigitalDetective"
    elif os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData/Roaming"))
        d = base / "DigitalDetective"
    elif os.uname().sysname == "Darwin":
        base = Path.home() / "Library/Application Support"
        d = base / "DigitalDetective"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        d = base / "DigitalDetective"
    d.mkdir(parents=True, exist_ok=True)
    return d


SETTINGS_PATH = _config_dir() / "settings.json"
DEFAULT_CASES_ROOT = _config_dir() / "cases"


class AppSettings(BaseModel):
    cases_root: str = str(DEFAULT_CASES_ROOT)
    llm_provider: str = "ollama"
    llm_model: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""
    use_agent: bool = False  # False -> deterministic pipeline
    classifier_model_dir: str = ""  # trained ONNX model dir, when available
    use_zeroshot: bool = False  # pretrained zero-shot voter (needs `zeroshot` extra)

    def build_classifier(self):
        from dd_lab.classify import build_classifier

        return build_classifier(self.classifier_model_dir or None, self.use_zeroshot)

    def agent_config(self) -> AgentConfig:
        cfg = AgentConfig.from_env()
        cfg.provider = self.llm_provider or cfg.provider
        if self.llm_model:
            cfg.model = self.llm_model
        cfg.base_url = self.llm_base_url or cfg.base_url
        cfg.api_key = self.llm_api_key or cfg.api_key
        return cfg

    def save(self) -> None:
        redacted = self.model_dump()
        SETTINGS_PATH.write_text(json.dumps(redacted, indent=2), encoding="utf-8")
        Path(self.cases_root).mkdir(parents=True, exist_ok=True)

    @classmethod
    def load(cls) -> AppSettings:
        if SETTINGS_PATH.is_file():
            s = cls.model_validate_json(SETTINGS_PATH.read_text("utf-8"))
        else:
            s = cls()
            s.save()
        Path(s.cases_root).mkdir(parents=True, exist_ok=True)
        return s


def describe_llm(cfg: AgentConfig) -> dict:
    return {k: v for k, v in asdict(cfg).items() if k != "api_key"} | {
        "api_key_set": bool(cfg.api_key)
    }
