"""Simple in-memory rate limiting for API requests."""

from __future__ import annotations

import time
from collections import defaultdict
from threading import Lock

from fastapi import HTTPException, Request

from app.api.auth import resolve_api_key
from config.loader import load_app_config

_buckets: dict[str, tuple[float, int]] = defaultdict(lambda: (time.monotonic(), 0))
_bucket_lock = Lock()


def _rate_limit_key(request: Request) -> str:
    api_key = resolve_api_key()
    if api_key:
        return f"api_key:{api_key}"
    client = request.client
    host = client.host if client else "unknown"
    return f"ip:{host}"


def enforce_rate_limit(request: Request) -> None:
    """Reject requests that exceed the configured per-minute limit."""
    limit = load_app_config().security.rate_limit_per_minute
    if limit <= 0:
        return

    key = _rate_limit_key(request)
    now = time.monotonic()
    with _bucket_lock:
        window_start, count = _buckets[key]
        if now - window_start >= 60:
            _buckets[key] = (now, 1)
            return
        if count >= limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        _buckets[key] = (window_start, count + 1)
