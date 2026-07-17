"""Streamlit authentication helpers."""

from __future__ import annotations

import hashlib
import secrets


def hash_password(password: str) -> str:
    """Return a SHA-256 hex digest for storing in config."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_password(password: str, expected_hash: str) -> bool:
    """Constant-time password verification against a stored hash."""
    return secrets.compare_digest(hash_password(password), expected_hash)
