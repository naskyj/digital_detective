"""AppSettings.judge_agent_config() — the judge is configurable from Settings
alone, no environment variables or CLI needed. Mirrors agent_config()'s
override pattern (Settings field > env var > default)."""

from __future__ import annotations

from dd_lab.api.settings import AppSettings


def test_judge_config_overrides_env_fallback(monkeypatch):
    monkeypatch.delenv("DD_JUDGE_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("DD_LLM_PROVIDER", raising=False)

    s = AppSettings(judge_llm_provider="anthropic", judge_llm_model="claude-sonnet-5")
    cfg = s.judge_agent_config()

    assert cfg.provider == "anthropic"
    assert cfg.model == "claude-sonnet-5"


def test_judge_config_falls_back_to_env_when_settings_blank(monkeypatch):
    monkeypatch.setenv("DD_JUDGE_LLM_PROVIDER", "openai")
    monkeypatch.setenv("DD_JUDGE_LLM_MODEL", "gpt-4o-mini")

    s = AppSettings()  # judge_llm_* left blank
    cfg = s.judge_agent_config()

    assert cfg.provider == "openai"
    assert cfg.model == "gpt-4o-mini"


def test_judge_config_falls_back_to_ollama_when_nothing_set(monkeypatch):
    for var in ("DD_JUDGE_LLM_PROVIDER", "DD_JUDGE_LLM_MODEL", "DD_LLM_PROVIDER", "DD_LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)

    cfg = AppSettings().judge_agent_config()

    assert cfg.provider == "ollama"


def test_judge_and_primary_can_be_configured_independently(monkeypatch):
    for var in ("DD_JUDGE_LLM_PROVIDER", "DD_JUDGE_LLM_MODEL", "DD_LLM_PROVIDER", "DD_LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)

    s = AppSettings(
        llm_provider="ollama", llm_model="llama3.1:8b",
        judge_llm_provider="anthropic", judge_llm_model="claude-sonnet-5",
    )

    assert s.agent_config().provider == "ollama"
    assert s.judge_agent_config().provider == "anthropic"
