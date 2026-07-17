"""Lag and rolling feature engineering."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

from features.config import FeatureConfig

logger = logging.getLogger(__name__)

Granularity = Literal["store_sku", "store_only"]

COLUMN_ALIASES: dict[str, list[str]] = {
    "total_price": ["selling_price"],
    "base_price": ["mrp"],
    "is_featured_sku": ["featured"],
    "is_display_sku": ["display"],
}


def reconcile_meta_columns(meta: pd.DataFrame) -> pd.DataFrame:
    """Rename aliases, derive missing engineered columns, and default booleans."""
    result = meta.copy()

    for target, aliases in COLUMN_ALIASES.items():
        if target not in result.columns:
            for alias in aliases:
                if alias in result.columns:
                    result = result.rename(columns={alias: target})
                    break

    if (
        "price_diff" not in result.columns
        and "total_price" in result.columns
        and "base_price" in result.columns
    ):
        result["price_diff"] = result["total_price"] - result["base_price"]

    if "is_featured_sku" not in result.columns:
        result["is_featured_sku"] = 0
    if "is_display_sku" not in result.columns:
        result["is_display_sku"] = 0

    return result


def _group_columns(granularity: Granularity) -> list[str]:
    if granularity == "store_only":
        return ["store_id"]
    return ["store_id", "sku_id"]


def build_features(
    df: pd.DataFrame,
    cfg: FeatureConfig,
    granularity: Granularity = "store_sku",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build lag/rolling features per time series."""
    group_cols = _group_columns(granularity)
    required = group_cols + ["week_dt", "units_sold"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for feature engineering: {missing}")

    featured = df.sort_values(group_cols + ["week_dt"]).copy()
    grouped = featured.groupby(group_cols, observed=True)["units_sold"]

    for lag in cfg.lags:
        featured[f"lag_{lag}"] = grouped.shift(lag)

    for window in cfg.rolling_windows:
        featured[f"roll_mean_{window}"] = grouped.transform(
            lambda series: series.shift(1).rolling(window, min_periods=window).mean()
        )

    for derived in cfg.derived_features:
        if derived == "price_diff":
            featured["price_diff"] = featured["total_price"] - featured["base_price"]
        else:
            raise ValueError(f"Unsupported derived feature: {derived}")

    lag_cols = [f"lag_{lag}" for lag in cfg.lags]
    roll_cols = [f"roll_mean_{window}" for window in cfg.rolling_windows]
    static_cols = list(cfg.static_columns)
    derived_cols = list(cfg.derived_features)

    required_static = set(static_cols)
    for derived in cfg.derived_features:
        if derived == "price_diff":
            required_static.update({"total_price", "base_price"})
    missing_static = [col for col in required_static if col not in featured.columns]
    if missing_static:
        raise ValueError(f"Missing required columns for static/derived features: {missing_static}")

    feature_cols = lag_cols + roll_cols + static_cols + derived_cols

    featured = featured.dropna(subset=feature_cols).reset_index(drop=True)
    logger.info(
        "Built %d features for %d rows (%s granularity)",
        len(feature_cols),
        len(featured),
        granularity,
    )

    x = featured[feature_cols]
    y = featured["units_sold"]
    meta = featured[group_cols + ["week", "week_dt", "units_sold"] + feature_cols].copy()
    return x, y, meta


def chronological_split(
    x: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    test_size_pct: float = 0.15,
    cutoff_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Split features chronologically by week_dt."""
    if not (0.0 < test_size_pct < 1.0):
        raise ValueError("test_size_pct must be between 0 and 1")

    order = meta["week_dt"].argsort(kind="mergesort")
    x_sorted = x.iloc[order].reset_index(drop=True)
    y_sorted = y.iloc[order].reset_index(drop=True)
    meta_sorted = meta.iloc[order].reset_index(drop=True)

    if cutoff_date:
        cutoff = pd.Timestamp(cutoff_date)
        train_mask = meta_sorted["week_dt"] < cutoff
        test_mask = ~train_mask
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            raise ValueError(f"cutoff_date {cutoff_date} leaves empty train or test set")
    else:
        test_size = max(1, int(len(x_sorted) * test_size_pct))
        train_mask = np.zeros(len(x_sorted), dtype=bool)
        train_mask[: len(x_sorted) - test_size] = True
        test_mask = ~train_mask

    x_train = x_sorted.loc[train_mask].reset_index(drop=True)
    x_test = x_sorted.loc[test_mask].reset_index(drop=True)
    y_train = y_sorted.loc[train_mask].reset_index(drop=True)
    y_test = y_sorted.loc[test_mask].reset_index(drop=True)
    meta_train = meta_sorted.loc[train_mask].reset_index(drop=True)
    meta_test = meta_sorted.loc[test_mask].reset_index(drop=True)

    logger.info(
        "Chronological split: train=%d rows (max date %s), test=%d rows (min date %s)",
        len(x_train),
        meta_train["week_dt"].max().date() if len(meta_train) else "n/a",
        len(x_test),
        meta_test["week_dt"].min().date() if len(meta_test) else "n/a",
    )
    return x_train, x_test, y_train, y_test, meta_train, meta_test
