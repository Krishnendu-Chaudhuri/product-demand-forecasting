"""Streamlit demand forecasting UI."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
for _p in (_SRC, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import logging  # noqa: E402
import os  # noqa: E402
import secrets  # noqa: E402
from typing import cast  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import streamlit as st  # noqa: E402
from app.streamlit.auth import verify_password  # noqa: E402
from config.loader import load_app_config  # noqa: E402
from config.settings import AppConfig  # noqa: E402
from data.cleaning import clean  # noqa: E402
from data.loader import load_raw  # noqa: E402
from features.config import FeatureConfig  # noqa: E402
from features.engineering import (  # noqa: E402
    Granularity,
    build_features,
    chronological_split,
    reconcile_meta_columns,
)
from models.predict import load_model, predict  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_ROOT = _ROOT


@st.cache_resource
def get_app_config() -> AppConfig:
    return load_app_config(PROJECT_ROOT)


@st.cache_resource
def load_trained_model(models_dir: str) -> tuple[object, dict]:
    return load_model(models_dir=models_dir)


@st.cache_data
def load_prepared_data(
    data_path: str,
    granularity: str,
    _feature_cfg: FeatureConfig,
) -> pd.DataFrame:
    df = clean(load_raw(data_path), granularity=cast(Granularity, granularity))
    _, _, meta = build_features(
        df,
        _feature_cfg,
        granularity=cast(Granularity, granularity),
    )
    return meta


def build_series_predictions(
    meta: pd.DataFrame,
    model: object,
    feature_names: list[str],
    store_id: int,
    sku_id: int,
) -> pd.DataFrame:
    series = meta[(meta["store_id"] == store_id) & (meta["sku_id"] == sku_id)].copy()
    if series.empty:
        return series

    x = series[feature_names]
    series["prediction"] = predict(model, x)
    return series.sort_values("week_dt")


def render_forecast_plot(series: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(series["week_dt"], series["units_sold"], label="Actual Sales", marker="o")
    ax.plot(series["week_dt"], series["prediction"], label="Predictions", marker="x")
    ax.set_xlabel("Week")
    ax.set_ylabel("Units Sold")
    ax.set_title("Predicted vs Actual Sales")
    ax.legend()
    fig.autofmt_xdate()
    st.pyplot(fig)
    plt.close(fig)


def _check_auth(app_cfg: AppConfig) -> None:
    if not app_cfg.ui.auth_enabled:
        return

    if st.session_state.get("authenticated"):
        return

    users = app_cfg.ui.users
    if users:
        username = st.sidebar.text_input("Username")
        entered = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            expected_hash = users.get(username)
            if expected_hash and verify_password(entered, expected_hash):
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.rerun()
            else:
                st.sidebar.error("Invalid username or password")
        st.stop()
        return

    password = os.environ.get("STREAMLIT_AUTH_PASSWORD") or app_cfg.ui.auth_password
    if not password:
        st.warning("Auth is enabled but no password is configured.")
        st.stop()

    logger.warning(
        "Using deprecated single-password Streamlit auth; configure ui.users with hashed passwords"
    )
    entered = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if secrets.compare_digest(entered, password):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.sidebar.error("Invalid password")
    st.stop()


def main() -> None:
    st.set_page_config(page_title="Demand Forecast", layout="wide")
    st.title("Product Demand Forecasting")

    app_cfg = get_app_config()
    _check_auth(app_cfg)
    data_path = str(app_cfg.resolve_data_path(PROJECT_ROOT))

    try:
        model, metadata = load_trained_model(app_cfg.registry.models_dir)
    except FileNotFoundError:
        st.error("No trained model found. Run `python main.py train` first.")
        st.stop()

    granularity = metadata.get("granularity", app_cfg.data.granularity)
    feature_names = metadata.get("feature_names")
    feature_cfg = app_cfg.features
    if feature_config := metadata.get("feature_config"):
        feature_cfg = FeatureConfig.model_validate(feature_config)
    if not feature_names:
        feature_names = feature_cfg.feature_names

    meta = load_prepared_data(data_path, granularity, feature_cfg)
    meta = reconcile_meta_columns(meta)

    logger.debug("meta.columns: %s", list(meta.columns))
    logger.debug("feature_names: %s", feature_names)

    missing = [c for c in feature_names if c not in meta.columns]
    logger.debug("missing columns: %s", missing)

    if missing:
        st.error(f"Missing columns: {missing}")
        st.write("Available columns:", list(meta.columns))
        st.stop()

    _, _, _, _, _, meta_test = chronological_split(
        meta[feature_names],
        meta["units_sold"],
        meta,
        test_size_pct=app_cfg.split.test_size_pct,
        cutoff_date=app_cfg.split.cutoff_date,
    )

    st.sidebar.header("Series Selection")
    store_ids = sorted(meta["store_id"].unique().tolist())
    store_id = st.sidebar.selectbox("Store ID", store_ids)

    sku_ids = sorted(meta.loc[meta["store_id"] == store_id, "sku_id"].unique().tolist())
    sku_id = st.sidebar.selectbox("SKU ID", sku_ids)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model Info**")
    st.sidebar.write(f"Version: `{metadata.get('version', 'unknown')}`")
    st.sidebar.write(f"Type: `{metadata.get('model_type', 'unknown')}`")
    metrics = metadata.get("metrics", {})
    if metrics:
        st.sidebar.write(
            f"Holdout R²: `{metrics.get('r2', 'n/a'):.4f}`"
            if isinstance(metrics.get("r2"), (int, float))
            else "Holdout R²: n/a"
        )

    series = build_series_predictions(meta_test, model, feature_names, store_id, sku_id)
    history = meta[(meta["store_id"] == store_id) & (meta["sku_id"] == sku_id)].sort_values(
        "week_dt", ascending=False
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Recent History")
        st.dataframe(
            history[["week", "week_dt", "units_sold"] + feature_names].head(8),
            use_container_width=True,
        )

        if not series.empty:
            latest = series.iloc[-1]
            st.metric("Latest Actual", f"{latest['units_sold']:.0f}")
            st.metric("Latest Prediction", f"{latest['prediction']:.1f}")

    with col2:
        st.subheader("Forecast vs Actual (Test Holdout)")
        if series.empty:
            st.info("No test holdout rows for the selected store/SKU.")
        else:
            render_forecast_plot(series)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
