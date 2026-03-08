from __future__ import annotations

import os
from threading import Lock
from typing import Dict

from langchain_core.rate_limiters import InMemoryRateLimiter

_lock = Lock()
_limiters: Dict[str, InMemoryRateLimiter] = {}

# Default requests-per-minute for each provider (free-tier safe values)
_PROVIDER_DEFAULTS_RPM: Dict[str, float] = {
    "hf":          10.0,   # HuggingFace Inference API free tier
    "groq":        30.0,   # Groq free tier (per-model limit)
    "openrouter":  20.0,
    "gemini":      15.0,
}


def get_shared_llm_rate_limiter(
    name: str = "gemini_llm",
    *,
    requests_per_second: float = 0.25,
    max_bucket_size: int = 1,
) -> InMemoryRateLimiter:
    with _lock:
        limiter = _limiters.get(name)
        if limiter is None:
            limiter = InMemoryRateLimiter(
                requests_per_second=requests_per_second,
                max_bucket_size=max_bucket_size,
            )
            _limiters[name] = limiter
        return limiter


def get_provider_rate_limiter(provider: str) -> InMemoryRateLimiter:
    """Return a shared rate limiter for *provider*, configured from env vars.

    Reads ``{PROVIDER}_REQUESTS_PER_MINUTE`` (e.g. ``HF_REQUESTS_PER_MINUTE``).
    Falls back to the hard-coded default for that provider.
    """
    env_var = f"{provider.upper()}_REQUESTS_PER_MINUTE"
    default_rpm = _PROVIDER_DEFAULTS_RPM.get(provider.lower(), 15.0)
    rpm = float(os.getenv(env_var, default_rpm))
    return get_shared_llm_rate_limiter(
        name=f"{provider.lower()}_llm",
        requests_per_second=rpm / 60.0,
        max_bucket_size=1,
    )

