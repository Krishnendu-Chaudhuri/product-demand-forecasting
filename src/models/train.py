"""Model training utilities."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from config.settings import AppConfig, ModelConfig

logger = logging.getLogger(__name__)

ModelFactory = Callable[[ModelConfig], RegressorMixin]
_MODEL_REGISTRY: dict[str, ModelFactory] = {}


def register_model(name: str, factory: ModelFactory) -> None:
    """Register a model factory under a config type name."""
    _MODEL_REGISTRY[name] = factory


def _resolve_max_features(value: str | None) -> str | int | float | None:
    if value is None:
        return None
    if value in {"sqrt", "log2"}:
        return value
    return value


def _create_random_forest(model_cfg: ModelConfig) -> RegressorMixin:
    params = model_cfg.hyperparams.random_forest.copy()
    max_features = params.get("max_features")
    if isinstance(max_features, str):
        params["max_features"] = _resolve_max_features(max_features)
    params["random_state"] = model_cfg.random_state
    return RandomForestRegressor(**params)


def _create_xgboost(model_cfg: ModelConfig) -> RegressorMixin:
    params = model_cfg.hyperparams.xgboost.copy()
    params["random_state"] = model_cfg.random_state
    params.setdefault("objective", "reg:squarederror")
    return xgb.XGBRegressor(**params)


register_model("random_forest", _create_random_forest)
register_model("xgboost", _create_xgboost)


def create_model(model_cfg: ModelConfig) -> RegressorMixin:
    """Instantiate a regressor from configuration."""
    factory = _MODEL_REGISTRY.get(model_cfg.type)
    if factory is None:
        raise ValueError(f"Unsupported model type: {model_cfg.type}")
    return factory(model_cfg)


def _normalize_tuning_grid(grid: dict[str, Any]) -> dict[str, list[Any]]:
    return {key: value if isinstance(value, list) else [value] for key, value in grid.items()}


def train_model(
    x_train: Any,
    y_train: Any,
    app_cfg: AppConfig,
    tune: bool = False,
) -> RegressorMixin:
    """Train a model with optional hyperparameter tuning."""
    model = create_model(app_cfg.model)

    if not tune:
        logger.info("Training %s without tuning", app_cfg.model.type)
        model.fit(x_train, np.asarray(y_train).ravel())
        return model

    if app_cfg.model.type != "random_forest":
        logger.warning("Tuning is only supported for random_forest; training with defaults")
        model.fit(x_train, np.asarray(y_train).ravel())
        return model

    param_grid = _normalize_tuning_grid(app_cfg.tuning.grid)
    base = RandomForestRegressor(random_state=app_cfg.model.random_state)
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_grid,
        n_iter=app_cfg.tuning.n_iter,
        cv=TimeSeriesSplit(n_splits=app_cfg.tuning.cv),
        random_state=app_cfg.model.random_state,
        n_jobs=-1,
    )
    logger.info("Running RandomizedSearchCV with %d iterations", app_cfg.tuning.n_iter)
    search.fit(x_train, np.asarray(y_train).ravel())
    logger.info("Best tuning params: %s", search.best_params_)
    return search.best_estimator_
