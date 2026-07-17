"""Load raw demand data from CSV."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from data.validation import validate_raw

logger = logging.getLogger(__name__)

WEEK_COLUMN = "week"
WEEK_DT_COLUMN = "week_dt"


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load CSV, parse week dates, and validate schema."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    logger.info("Loading data from %s", file_path)
    df = pd.read_csv(file_path)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    validate_raw(df)

    df[WEEK_DT_COLUMN] = pd.to_datetime(df[WEEK_COLUMN], format="%d/%m/%y", errors="coerce")
    invalid_dates = df[WEEK_DT_COLUMN].isna().sum()
    if invalid_dates:
        raise ValueError(f"Failed to parse {invalid_dates} week values as dates")

    df = df.sort_values(["store_id", "sku_id", WEEK_DT_COLUMN]).reset_index(drop=True)
    logger.info(
        "Parsed dates; date range %s to %s",
        df[WEEK_DT_COLUMN].min().date(),
        df[WEEK_DT_COLUMN].max().date(),
    )
    return df
