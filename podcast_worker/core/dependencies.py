"""Shared dependencies and helpers for routers."""

from typing import Optional

from podcast_worker.core import config


def resolve_llm_key(req, provider_field: str = "provider") -> Optional[str]:
    """Resolve provider API keys from server configuration only.

    Legacy request models still accept key fields for compatibility, but the
    service must not trust or forward client-submitted provider secrets.
    """
    provider = getattr(req, provider_field, None)

    if provider == "openrouter":
        return config.OPENROUTER_API_KEY
    if provider == "openai":
        return config.OPENAI_API_KEY
    return None
