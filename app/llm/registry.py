"""Provider selection.

Challenges call `get_provider()`. Tests call `set_provider()` to install a
`FakeProvider` and get deterministic exploits.
"""

from __future__ import annotations

from app.config import settings
from app.llm.base import LLMProvider, ProviderError

_override: LLMProvider | None = None
_cached: LLMProvider | None = None


def set_provider(provider: LLMProvider | None) -> None:
    """Install a provider override (used by tests). Pass None to clear."""
    global _override
    _override = provider


def get_provider() -> LLMProvider:
    global _cached

    if _override is not None:
        return _override

    if _cached is not None:
        return _cached

    name = settings.provider.lower().strip()

    if name == "ollama":
        from app.llm.ollama import OllamaProvider

        _cached = OllamaProvider()
    elif name == "anthropic":
        from app.llm.cloud import AnthropicProvider

        _cached = AnthropicProvider()
    elif name == "openai":
        from app.llm.cloud import OpenAIProvider

        _cached = OpenAIProvider()
    elif name == "fake":
        from app.llm.fake import FakeProvider

        _cached = FakeProvider()
    else:
        raise ProviderError(
            f"Unknown DVAA_PROVIDER '{settings.provider}'. "
            "Expected one of: ollama, anthropic, openai, fake."
        )

    return _cached


def describe_provider() -> str:
    """Human-readable provider label for the UI footer."""
    p = get_provider()
    model = getattr(p, "model", "")
    return f"{p.name}{f' ({model})' if model else ''}"


# --------------------------------------------------------------------------
# Model selection
#
# Which model is answering changes what a learner sees more than any other
# setting: a 3B model loses the thread of a tool-calling loop and the challenge
# looks broken, while a stronger one makes the defended levels behave like
# production. That is a knob to put in front of the player, not in a rebuild.
# --------------------------------------------------------------------------

MODEL_SETTING_KEY = "active_model"


def available_models() -> list[str]:
    """Models the player may switch to, or [] if the provider has no choice."""
    provider = get_provider()
    lister = getattr(provider, "installed_models", None)
    return lister() if callable(lister) else []


def current_model() -> str:
    return getattr(get_provider(), "model", "")


def set_model(name: str) -> bool:
    """Point the provider at another model. False if it is not installed."""
    provider = get_provider()
    if not hasattr(provider, "model") or name not in available_models():
        return False
    provider.model = name
    _persist_model(name)
    return True


def _persist_model(name: str) -> None:
    """Remember the choice, so it survives a restart of the container."""
    from app.db import Setting, get_session

    with get_session() as s:
        row = s.get(Setting, MODEL_SETTING_KEY) or Setting(key=MODEL_SETTING_KEY)
        row.value = name
        s.add(row)
        s.commit()


def restore_model() -> None:
    """Re-apply a previously chosen model at startup, if it is still installed."""
    from app.db import Setting, get_session

    with get_session() as s:
        row = s.get(Setting, MODEL_SETTING_KEY)
    if row and row.value:
        provider = get_provider()
        if hasattr(provider, "model") and row.value in available_models():
            provider.model = row.value
