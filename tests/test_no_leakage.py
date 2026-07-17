"""No-leakage checks for time-series pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from data.loader import load_raw
from features.config import load_feature_config
from features.engineering import build_features, chronological_split

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_CSV = FIXTURES / "sample_data.csv"
FEATURE_CONFIG = Path(__file__).parent.parent / "configs" / "features" / "v2.yaml"


def test_train_dates_do_not_overlap_test_period() -> None:
    df = load_raw(SAMPLE_CSV)
    cfg = load_feature_config(FEATURE_CONFIG)
    x, y, meta = build_features(df, cfg, granularity="store_sku")
    _, _, _, _, meta_train, meta_test = chronological_split(x, y, meta, test_size_pct=0.25)

    assert meta_train["week_dt"].max() <= meta_test["week_dt"].min()
    assert len(meta_train) + len(meta_test) == len(meta)


def test_lag_features_do_not_use_future_target() -> None:
    df = load_raw(SAMPLE_CSV)
    cfg = load_feature_config(FEATURE_CONFIG)
    _, _, meta = build_features(df, cfg, granularity="store_sku")

    for lag in cfg.lags:
        for (_, _), group in meta.groupby(["store_id", "sku_id"], sort=False):
            future = group["units_sold"].shift(-lag)
            valid = future.notna()
            if not valid.any():
                continue
            assert not np.allclose(group.loc[valid, f"lag_{lag}"].values, future.loc[valid].values)


def test_no_feature_equals_current_target() -> None:
    df = load_raw(SAMPLE_CSV)
    cfg = load_feature_config(FEATURE_CONFIG)
    _, _, meta = build_features(df, cfg, granularity="store_sku")

    for feature in cfg.feature_names:
        assert not np.allclose(meta[feature].values, meta["units_sold"].values)
