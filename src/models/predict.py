"""Model prediction utilities."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin

from models.registry import load_artifact

logger = logging.getLogger(__name__)


def predict(model: RegressorMixin, x: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Generate predictions from a trained model."""
    predictions = model.predict(x)
    result = np.asarray(predictions, dtype=float).ravel()
    if np.isnan(result).any():
        raise ValueError("Model produced NaN predictions")
    return result


def load_model(
    version: str | None = None,
    models_dir: str = "models",
) -> tuple[RegressorMixin, dict[str, Any]]:
    """Load a saved model and metadata."""
    artifact = load_artifact(version=version, models_dir=models_dir)
    logger.info("Loaded model version %s", artifact["metadata"]["version"])
    return artifact["model"], artifact["metadata"]
