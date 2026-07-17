"""API authentication helpers."""

from __future__ import annotations

import logging
import os
import secrets

from fastapi import Header, HTTPException

from config.loader import load_app_config

logger = logging.getLogger(__name__)
_api_key_warning_logged = False


def resolve_api_key() -> str | None:
    """Return configured API key from env or config, or None when disabled."""
    configured = os.environ.get("API_KEY")
    if configured:
        return configured
    return load_app_config().security.api_key or None


def verify_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """Require a valid X-API-Key header when API key auth is configured."""
    global _api_key_warning_logged
    configured = resolve_api_key()
    if not configured:
        if not _api_key_warning_logged:
            logger.warning("API key authentication is disabled; set API_KEY to enable")
            _api_key_warning_logged = True
        return

    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    if not secrets.compare_digest(x_api_key, configured):
        raise HTTPException(status_code=401, detail="Invalid API key")
