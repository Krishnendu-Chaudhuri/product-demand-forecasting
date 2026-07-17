"""Logging utilities."""

from __future__ import annotations

import logging


def configure_logging(level: str, fmt: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=fmt)
