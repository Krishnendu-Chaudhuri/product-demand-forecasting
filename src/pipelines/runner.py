"""Training and evaluation pipeline logic."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from config.settings import AppConfig
from data.cleaning import clean
from data.loader import load_raw
from features.engineering import build_features, chronological_split
from models.metrics import compute_metrics
from models.predict import predict
from models.registry import load_artifact, save_artifact
from models.train import train_model

logger = logging.getLogger(__name__)


def _metadata_data_path(data_path: Path) -> str:
    cwd = Path.cwd()
    if data_path.is_relative_to(cwd):
        return str(data_path.relative_to(cwd))
    return data_path.name


def run_train(app_cfg: AppConfig, tune: bool = False) -> Path:
    logger = logging.getLogger(__name__)
    data_path = app_cfg.resolve_data_path(Path.cwd())
    feature_cfg = app_cfg.features

    df = clean(load_raw(data_path), granularity=app_cfg.data.granularity)
    x, y, meta = build_features(df, feature_cfg, granularity=app_cfg.data.granularity)
    x_train, x_test, y_train, y_test, _, _ = chronological_split(
        x,
        y,
        meta,
        test_size_pct=app_cfg.split.test_size_pct,
        cutoff_date=app_cfg.split.cutoff_date,
    )

    model = train_model(x_train, y_train, app_cfg, tune=tune or app_cfg.tuning.enabled)
    y_pred = predict(model, x_test)
    metrics = compute_metrics(y_test.to_numpy(), y_pred)
    logger.info("Evaluation metrics: %s", metrics)

    metadata: dict[str, Any] = {
        "model_type": app_cfg.model.type,
        "granularity": app_cfg.data.granularity,
        "feature_config": feature_cfg.model_dump(),
        "feature_names": feature_cfg.feature_names,
        "metrics": metrics,
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "data_path": _metadata_data_path(data_path),
        "target_stats": {
            "mean": float(y_train.mean()),
            "std": float(y_train.std()),
        },
        "tuned": tune,
    }
    artifact_path = save_artifact(
        model,
        metadata,
        models_dir=app_cfg.registry.models_dir,
        keep_last_n=app_cfg.registry.keep_last_n,
    )
    return artifact_path


def run_evaluate(app_cfg: AppConfig, model_version: str | None = None) -> dict[str, float]:
    logger = logging.getLogger(__name__)
    data_path = app_cfg.resolve_data_path(Path.cwd())
    feature_cfg = app_cfg.features

    artifact = load_artifact(version=model_version, models_dir=app_cfg.registry.models_dir)
    model = artifact["model"]
    metadata = artifact["metadata"]

    granularity = metadata.get("granularity", app_cfg.data.granularity)
    df = clean(load_raw(data_path), granularity=granularity)
    x, y, meta = build_features(df, feature_cfg, granularity=granularity)
    _, x_test, _, y_test, _, _ = chronological_split(
        x,
        y,
        meta,
        test_size_pct=app_cfg.split.test_size_pct,
        cutoff_date=app_cfg.split.cutoff_date,
    )

    feature_names = metadata.get("feature_names", feature_cfg.feature_names)
    x_test = x_test[feature_names]
    y_pred = predict(model, x_test)
    metrics = compute_metrics(y_test.to_numpy(), y_pred)
    logger.info("Evaluation metrics for version %s: %s", metadata["version"], metrics)
    return metrics
