"""Model registry plugin tests."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from config.settings import ModelConfig
from models.train import create_model, register_model


class DummyRegressor(BaseEstimator, RegressorMixin):
    def fit(self, x: object, y: object) -> DummyRegressor:
        self.y_mean_ = float(np.mean(y))
        return self

    def predict(self, x: object) -> np.ndarray:
        rows = len(x)  # type: ignore[arg-type]
        return np.full(rows, self.y_mean_)


def test_register_custom_model_type() -> None:
    register_model("dummy", lambda _cfg: DummyRegressor())
    model = create_model(ModelConfig.model_construct(type="dummy"))
    y = np.array([1.0, 3.0, 5.0])
    model.fit([[1.0], [2.0], [3.0]], y)
    preds = model.predict([[4.0], [5.0]])
    assert preds.tolist() == [3.0, 3.0]
