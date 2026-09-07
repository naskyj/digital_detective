"""LLM configuration for the agent.

Model selection is pure configuration: pick a provider + model name (+ endpoint /
key). Installing Ollama and pulling a model is an install-time / setup task, not
something this module does — see docs/LLM_SETUP.md.

Providers:
  ollama     local, OpenAI-compatible endpoint (default)
  openai     OpenAI or any OpenAI-compatible service (set base_url)
  anthropic  Claude API                (needs the `anthropic` extra)
  google     Gemini API                (needs the `google` extra)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"

_DEFAULT_MODELS = {
    "ollama": DEFAULT_OLLAMA_MODEL,
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-5",
    "google": "gemini-2.5-flash",
}


@dataclass
class AgentConfig:
    provider: str = "ollama"
    model: str = DEFAULT_OLLAMA_MODEL
    base_url: str | None = None
    api_key: str | None = None
    max_steps: int = 40
    temperature: float = 0.0

    @classmethod
    def from_env(cls) -> AgentConfig:
        provider = os.environ.get("DD_LLM_PROVIDER", "ollama")
        return cls(
            provider=provider,
            model=os.environ.get("DD_LLM_MODEL", _DEFAULT_MODELS.get(provider, DEFAULT_OLLAMA_MODEL)),
            base_url=os.environ.get("DD_LLM_BASE_URL"),
            api_key=(
                os.environ.get("DD_LLM_API_KEY")
                or os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
            ),
        )


def build_model(cfg: AgentConfig):
    """Return a pydantic_ai model object for the configured provider."""
    if cfg.provider == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        return AnthropicModel(cfg.model, provider=AnthropicProvider(api_key=cfg.api_key))

    if cfg.provider == "google":
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider

        return GoogleModel(cfg.model, provider=GoogleProvider(api_key=cfg.api_key))

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    if cfg.provider == "ollama":
        provider = OpenAIProvider(base_url=cfg.base_url or DEFAULT_OLLAMA_URL, api_key="ollama")
    else:  # openai / generic OpenAI-compatible
        provider = OpenAIProvider(base_url=cfg.base_url, api_key=cfg.api_key)
    return OpenAIChatModel(cfg.model, provider=provider)


def test_connection(cfg: AgentConfig) -> dict:
    """Best-effort connectivity probe used by the UI's 'Test' button and CLI."""
    from pydantic_ai import Agent

    try:
        agent = Agent(build_model(cfg))
        reply = agent.run_sync("Reply with the single word: ready")
        return {"ok": True, "provider": cfg.provider, "model": cfg.model,
                "sample": str(reply.output)[:120]}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "provider": cfg.provider, "model": cfg.model, "error": str(e)[:300]}
