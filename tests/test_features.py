"""Feature engineering correctness tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.cleaning import clean
from data.loader import load_raw
from features.config import load_feature_config
from features.engineering import build_features, chronological_split, reconcile_meta_columns

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_CSV = FIXTURES / "sample_data.csv"
FEATURE_CONFIG = Path(__file__).parent.parent / "configs" / "features" / "v2.yaml"


def test_lag_values_match_prior_periods_within_series() -> None:
    df = load_raw(SAMPLE_CSV)
    cfg = load_feature_config(FEATURE_CONFIG)
    _, _, meta = build_features(df, cfg, granularity="store_sku")

    for _, row in meta.iterrows():
        store_id = row["store_id"]
        sku_id = row["sku_id"]
        week_dt = row["week_dt"]
        history = df[
            (df["store_id"] == store_id)
            & (df["sku_id"] == sku_id)
            & (df["week_dt"] < week_dt)
        ].sort_values("week_dt")

        for lag in cfg.lags:
            if len(history) >= lag:
                expected = history.iloc[-lag]["units_sold"]
                assert row[f"lag_{lag}"] == expected


def test_first_rows_per_series_dropped_for_missing_lags() -> None:
    df = load_raw(SAMPLE_CSV)
    cfg = load_feature_config(FEATURE_CONFIG)
    _, _, meta = build_features(df, cfg, granularity="store_sku")

    max_lag = max(cfg.lags)
    for (store_id, sku_id), group in df.groupby(["store_id", "sku_id"]):
        series_weeks = group.sort_values("week_dt")["week_dt"].tolist()
        if len(series_weeks) <= max_lag:
            assert meta[(meta["store_id"] == store_id) & (meta["sku_id"] == sku_id)].empty
        else:
            earliest_kept = meta[
                (meta["store_id"] == store_id) & (meta["sku_id"] == sku_id)
            ]["week_dt"].min()
            assert earliest_kept == series_weeks[max_lag]


def test_store_sku_granularity_preserves_row_count_order_of_magnitude() -> None:
    project_root = Path(__file__).parent.parent
    data_csv = project_root / "assets" / "data.csv"
    if not data_csv.exists():
        return

    df = clean(load_raw(data_csv), granularity="store_sku")
    cfg = load_feature_config(FEATURE_CONFIG)
    _, _, meta = build_features(df, cfg, granularity="store_sku")
    assert len(meta) > 100_000


def test_chronological_split_has_no_date_overlap() -> None:
    df = load_raw(SAMPLE_CSV)
    cfg = load_feature_config(FEATURE_CONFIG)
    x, y, meta = build_features(df, cfg, granularity="store_sku")
    _, _, _, _, meta_train, meta_test = chronological_split(x, y, meta, test_size_pct=0.25)

    assert meta_train["week_dt"].max() <= meta_test["week_dt"].min()


def test_v2_includes_price_and_promo_features() -> None:
    df = load_raw(SAMPLE_CSV)
    cfg = load_feature_config(FEATURE_CONFIG)
    _, _, meta = build_features(df, cfg, granularity="store_sku")

    for col in ["total_price", "base_price", "is_featured_sku", "is_display_sku", "price_diff"]:
        assert col in meta.columns

    assert (meta["price_diff"] == meta["total_price"] - meta["base_price"]).all()


def test_v2_feature_names_match_config() -> None:
    cfg = load_feature_config(FEATURE_CONFIG)
    assert len(cfg.feature_names) == 9
    assert cfg.feature_names == [
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_4",
        "total_price",
        "base_price",
        "is_featured_sku",
        "is_display_sku",
        "price_diff",
    ]


def test_reconcile_renames_column_aliases() -> None:
    meta = pd.DataFrame(
        {
            "store_id": [1],
            "sku_id": [1],
            "selling_price": [10.0],
            "mrp": [12.0],
            "featured": [1],
            "display": [0],
        }
    )
    result = reconcile_meta_columns(meta)

    assert "total_price" in result.columns
    assert "base_price" in result.columns
    assert "is_featured_sku" in result.columns
    assert "is_display_sku" in result.columns
    assert result["total_price"].iloc[0] == 10.0
    assert result["base_price"].iloc[0] == 12.0


def test_reconcile_derives_price_diff() -> None:
    meta = pd.DataFrame({"total_price": [10.0, 20.0], "base_price": [12.0, 18.0]})
    result = reconcile_meta_columns(meta)

    assert "price_diff" in result.columns
    assert (result["price_diff"] == result["total_price"] - result["base_price"]).all()


def test_reconcile_defaults_missing_boolean_columns() -> None:
    meta = pd.DataFrame({"total_price": [10.0], "base_price": [12.0]})
    result = reconcile_meta_columns(meta)

    assert result["is_featured_sku"].iloc[0] == 0
    assert result["is_display_sku"].iloc[0] == 0


def test_reconcile_noop_when_columns_already_present() -> None:
    df = load_raw(SAMPLE_CSV)
    cfg = load_feature_config(FEATURE_CONFIG)
    _, _, meta = build_features(df, cfg, granularity="store_sku")
    result = reconcile_meta_columns(meta)

    for col in [
        "total_price",
        "base_price",
        "is_featured_sku",
        "is_display_sku",
        "price_diff",
    ]:
        assert col in result.columns
    assert (result["price_diff"] == result["total_price"] - result["base_price"]).all()
