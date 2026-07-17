"""Model prediction tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from config.settings import ModelConfig
from data.loader import load_raw
from features.config import load_feature_config
from features.engineering import build_features
from models.predict import predict
from models.registry import save_artifact
from models.train import create_model

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_CSV = FIXTURES / "sample_data.csv"
FEATURE_CONFIG = Path(__file__).parent.parent / "configs" / "features" / "v2.yaml"


def test_predict_output_shape_and_dtype(tmp_path: Path) -> None:
    df = load_raw(SAMPLE_CSV)
    cfg = load_feature_config(FEATURE_CONFIG)
    x, y, _ = build_features(df, cfg, granularity="store_sku")

    model_cfg = ModelConfig(type="random_forest")
    model = create_model(model_cfg)
    model.fit(x, y)

    preds = predict(model, x)
    assert preds.shape == (len(x),)
    assert preds.dtype == float
    assert not np.isnan(preds).any()


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    x = pd.DataFrame({"lag_1": [1, 2, 3], "lag_2": [4, 5, 6]})
    y = np.array([10, 11, 12])
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(x, y)

    metadata = {"feature_names": list(x.columns), "model_type": "random_forest"}
    save_artifact(model, metadata, models_dir=str(tmp_path), version="test_v1")

    from models.predict import load_model

    loaded_model, loaded_meta = load_model(version="test_v1", models_dir=str(tmp_path))
    preds = predict(loaded_model, x)
    assert loaded_meta["version"] == "test_v1"
    assert preds.shape == (len(x),)
