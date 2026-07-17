"""FastAPI prediction endpoint tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestRegressor

from app.api import main as api_main
from app.api import rate_limit as rate_limit_module
from app.api.main import app
from models.registry import save_artifact

PREDICT_PAYLOAD = {
    "features": {
        "lag_1": 3.0,
        "lag_2": 6.0,
        "lag_3": 9.0,
        "lag_4": 12.0,
        "total_price": 12.0,
        "base_price": 12.0,
        "is_featured_sku": 0.0,
        "is_display_sku": 1.0,
        "price_diff": 0.0,
    }
}


def _save_v2_model(models_dir: str, version: str) -> None:
    x = pd.DataFrame(
        {
            "lag_1": [1.0, 2.0, 3.0],
            "lag_2": [4.0, 5.0, 6.0],
            "lag_3": [7.0, 8.0, 9.0],
            "lag_4": [10.0, 11.0, 12.0],
            "total_price": [10.0, 11.0, 12.0],
            "base_price": [12.0, 12.0, 12.0],
            "is_featured_sku": [0.0, 1.0, 0.0],
            "is_display_sku": [0.0, 0.0, 1.0],
            "price_diff": [-2.0, -1.0, 0.0],
        }
    )
    y = np.array([10.0, 11.0, 12.0])
    model = RandomForestRegressor(n_estimators=10, max_depth=4, random_state=0)
    model.fit(x, y)
    save_artifact(
        model,
        {"feature_names": list(x.columns), "model_type": "random_forest"},
        models_dir=models_dir,
        version=version,
    )


def test_predict_endpoint(tmp_path, monkeypatch) -> None:
    models_dir = str(tmp_path / "models")
    _save_v2_model(models_dir, "api_test_v1")

    monkeypatch.setattr("app.api.main._get_models_dir", lambda: models_dir)
    monkeypatch.setattr("app.api.auth.resolve_api_key", lambda: None)
    client = TestClient(app)

    response = client.post("/predict", json=PREDICT_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "api_test_v1"
    assert body["model_type"] == "random_forest"
    assert isinstance(body["prediction"], float)


def test_predict_missing_features_returns_422(tmp_path, monkeypatch) -> None:
    models_dir = str(tmp_path / "models")
    x = pd.DataFrame({"lag_1": [1.0, 2.0]})
    y = np.array([10.0, 11.0])
    model = RandomForestRegressor(n_estimators=5, random_state=0)
    model.fit(x, y)
    save_artifact(
        model,
        {"feature_names": ["lag_1"], "model_type": "random_forest"},
        models_dir=models_dir,
        version="api_test_v2",
    )

    monkeypatch.setattr("app.api.main._get_models_dir", lambda: models_dir)
    monkeypatch.setattr("app.api.auth.resolve_api_key", lambda: None)
    client = TestClient(app)
    response = client.post("/predict", json={"features": {}})
    assert response.status_code == 422


def test_predict_requires_api_key_when_configured(tmp_path, monkeypatch) -> None:
    models_dir = str(tmp_path / "models")
    _save_v2_model(models_dir, "api_test_auth")

    monkeypatch.setattr("app.api.main._get_models_dir", lambda: models_dir)
    monkeypatch.setattr("app.api.auth.resolve_api_key", lambda: "secret-key")
    client = TestClient(app)

    response = client.post("/predict", json=PREDICT_PAYLOAD)
    assert response.status_code == 401
    assert response.json()["detail"] == "Missing X-API-Key header"


def test_predict_invalid_api_key_returns_401(tmp_path, monkeypatch) -> None:
    models_dir = str(tmp_path / "models")
    _save_v2_model(models_dir, "api_test_auth2")

    monkeypatch.setattr("app.api.main._get_models_dir", lambda: models_dir)
    monkeypatch.setattr("app.api.auth.resolve_api_key", lambda: "secret-key")
    client = TestClient(app)

    response = client.post(
        "/predict",
        json=PREDICT_PAYLOAD,
        headers={"X-API-Key": "wrong-key"},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API key"


def test_predict_with_valid_api_key(tmp_path, monkeypatch) -> None:
    models_dir = str(tmp_path / "models")
    _save_v2_model(models_dir, "api_test_auth3")

    monkeypatch.setattr("app.api.main._get_models_dir", lambda: models_dir)
    monkeypatch.setattr("app.api.auth.resolve_api_key", lambda: "secret-key")
    client = TestClient(app)

    response = client.post(
        "/predict",
        json=PREDICT_PAYLOAD,
        headers={"X-API-Key": "secret-key"},
    )
    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], float)


def test_predict_caches_model_in_memory(tmp_path, monkeypatch) -> None:
    import joblib

    models_dir = str(tmp_path / "models")
    _save_v2_model(models_dir, "api_test_cache")
    load_calls = {"count": 0}
    original_load = joblib.load

    def counting_load(path: str) -> object:
        load_calls["count"] += 1
        return original_load(path)

    monkeypatch.setattr("app.api.main._get_models_dir", lambda: models_dir)
    monkeypatch.setattr("app.api.auth.resolve_api_key", lambda: None)
    monkeypatch.setattr(joblib, "load", counting_load)
    api_main._get_cached_artifact.cache_clear()

    client = TestClient(app)
    for _ in range(2):
        response = client.post("/predict", json=PREDICT_PAYLOAD)
        assert response.status_code == 200

    assert load_calls["count"] == 1
    api_main._get_cached_artifact.cache_clear()


def test_predict_rate_limit_returns_429(tmp_path, monkeypatch) -> None:
    from config.settings import AppConfig, SecurityConfig

    models_dir = str(tmp_path / "models")
    _save_v2_model(models_dir, "api_test_rate")

    monkeypatch.setattr("app.api.main._get_models_dir", lambda: models_dir)
    monkeypatch.setattr("app.api.auth.resolve_api_key", lambda: None)
    monkeypatch.setattr(
        "app.api.rate_limit.load_app_config",
        lambda: AppConfig(security=SecurityConfig(rate_limit_per_minute=1)),
    )
    rate_limit_module._buckets.clear()

    client = TestClient(app)
    first = client.post("/predict", json=PREDICT_PAYLOAD)
    second = client.post("/predict", json=PREDICT_PAYLOAD)
    assert first.status_code == 200
    assert second.status_code == 429
    rate_limit_module._buckets.clear()
