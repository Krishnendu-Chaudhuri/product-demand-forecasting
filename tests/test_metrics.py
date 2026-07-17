"""Metrics unit tests."""

from __future__ import annotations

import numpy as np

from models.metrics import compute_metrics


def test_compute_metrics_known_values() -> None:
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])

    metrics = compute_metrics(y_true, y_pred)

    assert np.isclose(metrics["mae"], 7.0 / 3.0)
    assert np.isclose(metrics["rmse"], np.sqrt((4 + 4 + 9) / 3))
    assert metrics["r2"] < 1.0
    assert metrics["mape"] > 0.0
