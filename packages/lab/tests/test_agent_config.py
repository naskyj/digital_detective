"""AgentConfig.from_env prefix fallback — lets the judge be configured
independently of the primary agent (DD_JUDGE_LLM_* falling back to DD_LLM_*)."""

from __future__ import annotations

from dd_lab.agent.config import AgentConfig


def test_primary_reads_dd_llm_vars(monkeypatch):
    monkeypatch.setenv("DD_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("DD_LLM_MODEL", "claude-sonnet-5")
    cfg = AgentConfig.from_env()
    assert cfg.provider == "anthropic"
    assert cfg.model == "claude-sonnet-5"


def test_judge_prefix_overrides_primary(monkeypatch):
    monkeypatch.setenv("DD_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("DD_LLM_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("DD_JUDGE_LLM_PROVIDER", "openai")
    monkeypatch.setenv("DD_JUDGE_LLM_MODEL", "gpt-4o-mini")

    cfg = AgentConfig.from_env(prefix="DD_JUDGE_LLM")
    assert cfg.provider == "openai"
    assert cfg.model == "gpt-4o-mini"


def test_judge_falls_back_to_primary_when_unset(monkeypatch):
    monkeypatch.delenv("DD_JUDGE_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("DD_JUDGE_LLM_MODEL", raising=False)
    monkeypatch.setenv("DD_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("DD_LLM_MODEL", "claude-sonnet-5")

    cfg = AgentConfig.from_env(prefix="DD_JUDGE_LLM")
    assert cfg.provider == "anthropic"
    assert cfg.model == "claude-sonnet-5"


def test_judge_falls_back_to_ollama_when_nothing_set(monkeypatch):
    for var in ("DD_JUDGE_LLM_PROVIDER", "DD_JUDGE_LLM_MODEL", "DD_LLM_PROVIDER", "DD_LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)

    cfg = AgentConfig.from_env(prefix="DD_JUDGE_LLM")
    assert cfg.provider == "ollama"


def test_judge_api_key_falls_back_through_provider_env_vars(monkeypatch):
    for var in ("DD_JUDGE_LLM_API_KEY", "DD_LLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")

    cfg = AgentConfig.from_env(prefix="DD_JUDGE_LLM")
    assert cfg.api_key == "sk-test-123"
