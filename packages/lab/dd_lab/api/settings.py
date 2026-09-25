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

from dd_lab.agent.config import AgentConfig, env_key_for


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
    classifier_model_dir: str = ""  # trained category ONNX model dir, when available
    relevance_model_dir: str = ""  # trained binary relevant/not_relevant ONNX model dir - a
    # DIFFERENT model/slot from classifier_model_dir above (see
    # dd_lab/classify/ensemble.py's module docstring and model.py's
    # OnnxRelevanceModel) - this one predicts relevance directly, feeding
    # the primary relevant/relevance_score output, not category.
    use_zeroshot: bool = False  # pretrained zero-shot voter (needs `zeroshot` extra)
    # The adversarial judge (agent/judge.py) reviews every finding independently
    # of the primary agent above. Left blank, it falls back to DD_JUDGE_LLM_*
    # env vars, then to the primary agent's own config, then to Ollama — so a
    # judge always at least attempts to run with no configuration at all. Set
    # these to point it at a genuinely different model/provider than the
    # primary agent, which is the whole point of an *independent* review.
    use_judge: bool = True
    judge_llm_provider: str = ""
    judge_llm_model: str = ""
    judge_llm_base_url: str = ""
    judge_llm_api_key: str = ""

    def build_classifier(self):
        from dd_lab.classify import build_classifier

        # Keyword args, deliberately - build_classifier() takes 3 positional-
        # capable params now (model_dir, relevance_model_dir, zeroshot); a
        # positional call here would silently bind use_zeroshot to the
        # WRONG parameter after relevance_model_dir was added.
        return build_classifier(
            model_dir=self.classifier_model_dir or None,
            relevance_model_dir=self.relevance_model_dir or None,
            zeroshot=self.use_zeroshot,
        )

    def agent_config(self) -> AgentConfig:
        cfg = AgentConfig.from_env()
        cfg.provider = self.llm_provider or cfg.provider
        if self.llm_model:
            cfg.model = self.llm_model
        cfg.base_url = self.llm_base_url or cfg.base_url
        # Settings first, then this provider's OWN environment variable.
        # Re-resolved here because `provider` was just overridden above, and
        # from_env() could only see whatever DD_LLM_PROVIDER said.
        cfg.api_key = self.llm_api_key or cfg.api_key or env_key_for(cfg.provider)
        return cfg

    def judge_agent_config(self) -> AgentConfig:
        cfg = AgentConfig.from_env(prefix="DD_JUDGE_LLM")
        cfg.provider = self.judge_llm_provider or cfg.provider
        if self.judge_llm_model:
            cfg.model = self.judge_llm_model
        cfg.base_url = self.judge_llm_base_url or cfg.base_url
        cfg.api_key = (
            self.judge_llm_api_key or cfg.api_key
            or env_key_for(cfg.provider, prefix="DD_JUDGE_LLM")
            or env_key_for(cfg.provider)
        )
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
