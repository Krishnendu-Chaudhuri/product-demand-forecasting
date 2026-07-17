"""FastAPI application for model inference."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request

from app.api.auth import verify_api_key
from app.api.rate_limit import enforce_rate_limit
from app.api.schemas import PredictRequest, PredictResponse
from config.loader import load_app_config
from models.predict import predict
from models.registry import load_artifact

logger = logging.getLogger(__name__)

app = FastAPI(title="Demand Forecast API", version="0.1.0")
LATEST_VERSION_KEY = "__latest__"


@lru_cache(maxsize=1)
def _get_models_dir() -> str:
    return load_app_config().registry.models_dir


@lru_cache(maxsize=8)
def _get_cached_artifact(models_dir: str, version_key: str) -> dict[str, Any]:
    version = None if version_key == LATEST_VERSION_KEY else version_key
    return load_artifact(version=version, models_dir=models_dir)


def _load_model(version: str | None) -> dict[str, Any]:
    version_key = version or LATEST_VERSION_KEY
    try:
        return _get_cached_artifact(_get_models_dir(), version_key)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(
    body: PredictRequest,
    request: Request,
    model_version: str | None = Query(default=None, description="Optional model version"),
    _: None = Depends(verify_api_key),
) -> PredictResponse:
    enforce_rate_limit(request)
    artifact = _load_model(model_version)
    model = artifact["model"]
    metadata = artifact["metadata"]
    feature_names: list[str] = metadata.get("feature_names", [])
    if not feature_names:
        raise HTTPException(status_code=500, detail="Model metadata missing feature_names")

    missing = [name for name in feature_names if name not in body.features]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required features: {missing}",
        )

    row = {name: body.features[name] for name in feature_names}
    x = pd.DataFrame([row])
    try:
        prediction = float(predict(model, x)[0])
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    logger.info(
        json.dumps(
            {
                "event": "prediction",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_version": metadata["version"],
                "feature_summary": {name: body.features[name] for name in feature_names},
                "prediction": prediction,
            }
        )
    )

    return PredictResponse(
        prediction=prediction,
        model_version=metadata["version"],
        model_type=metadata.get("model_type", "unknown"),
    )
