"""Data cleaning and optional aggregation."""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

Granularity = Literal["store_sku", "store_only"]


def impute_total_price(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing total_price with column median."""
    result = df.copy()
    missing = result["total_price"].isna().sum()
    if missing:
        median_value = result["total_price"].median()
        result["total_price"] = result["total_price"].fillna(median_value)
        logger.info("Imputed %d missing total_price values with median %.4f", missing, median_value)
    return result


def apply_granularity(df: pd.DataFrame, granularity: Granularity) -> pd.DataFrame:
    """Keep SKU-level rows or aggregate to store-week totals."""
    if granularity == "store_sku":
        logger.info("Using store_sku granularity (%d rows)", len(df))
        return df

    logger.info("Aggregating to store_only granularity")
    aggregated = (
        df.groupby(["week_dt", "week", "store_id"], as_index=False)
        .agg(
            record_ID=("record_ID", "min"),
            sku_id=("sku_id", "first"),
            total_price=("total_price", "sum"),
            base_price=("base_price", "mean"),
            is_featured_sku=("is_featured_sku", "max"),
            is_display_sku=("is_display_sku", "max"),
            units_sold=("units_sold", "sum"),
        )
        .sort_values(["store_id", "week_dt"])
        .reset_index(drop=True)
    )
    logger.info("Aggregated to %d store-week rows", len(aggregated))
    return aggregated


def clean(df: pd.DataFrame, granularity: Granularity = "store_sku") -> pd.DataFrame:
    """Run imputation and optional aggregation."""
    cleaned = impute_total_price(df)
    return apply_granularity(cleaned, granularity)
