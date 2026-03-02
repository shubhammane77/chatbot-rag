from __future__ import annotations

from threading import Lock
from typing import Dict

from langchain_core.rate_limiters import InMemoryRateLimiter

_lock = Lock()
_limiters: Dict[str, InMemoryRateLimiter] = {}


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

