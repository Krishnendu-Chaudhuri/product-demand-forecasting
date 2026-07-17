"""Backward-compatible shim for FastAPI app."""

from app.api.main import app

__all__ = ["app"]
