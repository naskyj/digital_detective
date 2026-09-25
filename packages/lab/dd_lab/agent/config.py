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


#: Which environment variable holds a key for which provider. A key is only
#: ever read from the variable belonging to the provider actually in use.
#:
#: The previous fallback was one flat chain - ANTHROPIC_API_KEY, then
#: OPENAI_API_KEY, then GEMINI_API_KEY, then GOOGLE_API_KEY - tried in order
#: for every provider. On a machine with an OpenAI key in the environment and
#: no Anthropic one, an anthropic-provider agent therefore picked up the
#: OpenAI key and sent it to Anthropic, which answered
#: "401 authentication_error: API key is invalid". That is indistinguishable
#: from a genuinely bad Anthropic key, and it cost two days of debugging a
#: key that was never being used: the real fault was that no Anthropic key
#: existed at all and the code silently substituted somebody else's.
#:
#: `DD_LLM_API_KEY` (and `DD_JUDGE_LLM_API_KEY`) still work for any provider -
#: those are explicit, so there is nothing to guess.
_PROVIDER_ENV_KEYS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "ollama": (),  # local, no key
}


def env_key_for(provider: str, prefix: str = "DD_LLM") -> str | None:
    """The API key in the environment for THIS provider, or None.

    Exists because AppSettings overrides `provider` after `from_env()` has
    already run, so the key has to be resolvable again once the effective
    provider is known - otherwise configuring "anthropic" in Settings while
    ANTHROPIC_API_KEY is exported leaves the key unread.
    """
    import os as _os

    if explicit := _os.environ.get(f"{prefix}_API_KEY"):
        return explicit
    for name in _PROVIDER_ENV_KEYS.get(provider, ()):
        if val := _os.environ.get(name):
            return val
    return None


@dataclass
class AgentConfig:
    provider: str = "ollama"
    model: str = DEFAULT_OLLAMA_MODEL
    base_url: str | None = None
    api_key: str | None = None
    max_steps: int = 40
    temperature: float = 0.0

    @classmethod
    def from_env(cls, prefix: str = "DD_LLM") -> AgentConfig:
        """`prefix` lets a second role (e.g. the judge, `DD_JUDGE_LLM_*`) be
        configured independently — falling back to the primary agent's
        `DD_LLM_*` vars for anything not set under its own prefix, and to
        Ollama if neither is set, so a judge "just works" with no extra setup
        but can be pointed at a different provider/model to avoid sharing the
        primary model's blind spots."""
        primary = "DD_LLM"

        def get(name: str, *fallback_names: str) -> str | None:
            prefixes = (prefix,) if prefix == primary else (prefix, primary)
            for p in prefixes:
                if val := os.environ.get(f"{p}_{name}"):
                    return val
            for name2 in fallback_names:
                if val := os.environ.get(name2):
                    return val
            return None

        provider = get("PROVIDER") or "ollama"
        return cls(
            provider=provider,
            model=get("MODEL") or _DEFAULT_MODELS.get(provider, DEFAULT_OLLAMA_MODEL),
            base_url=get("BASE_URL"),
            # Only this provider's own variable - see _PROVIDER_ENV_KEYS.
            api_key=get("API_KEY", *_PROVIDER_ENV_KEYS.get(provider, ())),
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
