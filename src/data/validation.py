"""Data validation utilities."""

from __future__ import annotations

import logging

import pandas as pd
from pandera.errors import SchemaError, SchemaErrors

from data.schema import RAW_SCHEMA

logger = logging.getLogger(__name__)


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Validate raw dataframe against schema; fail fast on errors."""
    try:
        validated = RAW_SCHEMA.validate(df, lazy=True)
        logger.info("Schema validation passed")
        return validated
    except (SchemaError, SchemaErrors) as exc:
        logger.error("Schema validation failed: %s", exc)
        raise ValueError(f"Data validation failed: {exc}") from exc
