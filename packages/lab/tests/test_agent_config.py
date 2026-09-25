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


def test_judge_reads_its_own_providers_key(monkeypatch):
    """This previously asserted the opposite: that a judge defaulting to
    OLLAMA still picked up ANTHROPIC_API_KEY. That flat cross-provider
    fallback was the bug - it is what let an OpenAI key be sent to Anthropic -
    so the test was pinning it. A local ollama judge needs no key at all;
    an anthropic one reads Anthropic's own variable."""
    for var in ("DD_JUDGE_LLM_API_KEY", "DD_LLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")

    local = AgentConfig.from_env(prefix="DD_JUDGE_LLM")
    assert local.provider == "ollama"
    assert local.api_key is None

    monkeypatch.setenv("DD_JUDGE_LLM_PROVIDER", "anthropic")
    remote = AgentConfig.from_env(prefix="DD_JUDGE_LLM")
    assert remote.api_key == "sk-test-123"


# ------------------------------------------------- one provider, one key
#
# The fallback used to be a single flat chain tried for every provider:
# ANTHROPIC_API_KEY, then OPENAI_API_KEY, then GEMINI_API_KEY, then
# GOOGLE_API_KEY. On a machine with an OpenAI key exported and no Anthropic
# one, an anthropic-provider agent picked up the OpenAI key and sent it to
# Anthropic, which replied "401 authentication_error: API key is invalid" -
# indistinguishable from a genuinely bad Anthropic key. Two days were spent
# re-checking a key that was never being used.


def test_another_providers_key_is_never_substituted(monkeypatch):
    monkeypatch.delenv("DD_LLM_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-an-openai-key")
    monkeypatch.setenv("DD_LLM_PROVIDER", "anthropic")

    cfg = AgentConfig.from_env()

    assert cfg.provider == "anthropic"
    assert cfg.api_key is None, "an OpenAI key must never be sent to Anthropic"


def test_the_matching_providers_key_is_used(monkeypatch):
    monkeypatch.delenv("DD_LLM_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-real")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-an-openai-key")
    monkeypatch.setenv("DD_LLM_PROVIDER", "anthropic")

    assert AgentConfig.from_env().api_key == "sk-ant-real"


def test_google_accepts_either_of_its_two_variables(monkeypatch):
    monkeypatch.delenv("DD_LLM_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "goog-key")
    monkeypatch.setenv("DD_LLM_PROVIDER", "google")

    assert AgentConfig.from_env().api_key == "goog-key"


def test_the_explicit_variable_still_works_for_any_provider(monkeypatch):
    """DD_LLM_API_KEY is explicit, so there is nothing to guess and no
    cross-provider risk."""
    monkeypatch.setenv("DD_LLM_API_KEY", "explicit-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("DD_LLM_PROVIDER", "anthropic")

    assert AgentConfig.from_env().api_key == "explicit-key"


def test_settings_provider_override_still_finds_its_own_env_key(monkeypatch, tmp_path):
    """AppSettings overrides `provider` AFTER from_env() ran, so the key has
    to be resolvable again once the effective provider is known - otherwise
    choosing Anthropic in Settings with ANTHROPIC_API_KEY exported silently
    yields no key at all."""
    from dd_lab.api.settings import AppSettings

    monkeypatch.delenv("DD_LLM_API_KEY", raising=False)
    monkeypatch.delenv("DD_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-an-openai-key")

    cfg = AppSettings(llm_provider="anthropic").agent_config()

    assert cfg.provider == "anthropic"
    assert cfg.api_key == "sk-ant-from-env"


def test_a_key_set_in_settings_beats_the_environment(monkeypatch):
    from dd_lab.api.settings import AppSettings

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
    cfg = AppSettings(llm_provider="anthropic", llm_api_key="sk-ant-from-settings").agent_config()

    assert cfg.api_key == "sk-ant-from-settings"
