"""Evaluation metrics for demand forecasting."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-8,
) -> dict[str, float]:
    """Compute regression metrics including MAE, RMSE, MAPE, and R2."""
    y_true_arr = np.asarray(y_true, dtype=float).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=float).ravel()

    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    mape = float(
        np.mean(np.abs((y_true_arr - y_pred_arr) / np.maximum(np.abs(y_true_arr), epsilon)))
        * 100.0
    )
    r2 = float(r2_score(y_true_arr, y_pred_arr))

    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}
