"""Shared dependencies and helpers for routers."""

import os
from typing import Optional


def resolve_llm_key(req, provider_field: str = "provider") -> Optional[str]:
    """Resolve the LLM API key, handling OpenRouter special case."""
    provider = getattr(req, provider_field, None)
    api_key = getattr(req, "api_key", None)
    openrouter_key = getattr(req, "openrouter_key", None)

    if provider == "openrouter":
        return openrouter_key or os.environ.get("OPENROUTER_API_KEY", "")
    return api_key
