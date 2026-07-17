"""Schema and loader validation tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.loader import load_raw
from data.validation import validate_raw

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_CSV = FIXTURES / "sample_data.csv"
PROJECT_ROOT = Path(__file__).parent.parent
DATA_CSV = PROJECT_ROOT / "assets" / "data.csv"


def test_validate_sample_fixture_passes() -> None:
    df = pd.read_csv(SAMPLE_CSV)
    validated = validate_raw(df)
    assert len(validated) == len(df)


def test_validate_rejects_negative_units_sold() -> None:
    df = pd.read_csv(SAMPLE_CSV)
    df.loc[0, "units_sold"] = -1
    with pytest.raises(ValueError, match="validation failed"):
        validate_raw(df)


def test_validate_rejects_invalid_feature_flag() -> None:
    df = pd.read_csv(SAMPLE_CSV)
    df.loc[0, "is_featured_sku"] = 2
    with pytest.raises(ValueError, match="validation failed"):
        validate_raw(df)


def test_load_raw_parses_dates_and_sorts() -> None:
    df = load_raw(SAMPLE_CSV)
    assert "week_dt" in df.columns
    assert df["week_dt"].isna().sum() == 0
    assert df.equals(
        df.sort_values(["store_id", "sku_id", "week_dt"]).reset_index(drop=True)
    )


def test_load_raw_real_data_csv() -> None:
    if not DATA_CSV.exists():
        pytest.skip("assets/data.csv not available")
    df = load_raw(DATA_CSV)
    assert len(df) == 150_150
    assert df["week_dt"].min() < df["week_dt"].max()


def test_load_raw_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_raw(FIXTURES / "missing.csv")
