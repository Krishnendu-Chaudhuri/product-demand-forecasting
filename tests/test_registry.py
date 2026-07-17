"""Registry retention policy tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from models.registry import list_versions, save_artifact


def _save_dummy(version: str, models_dir: str, keep_last_n: int | None = None) -> None:
    x = pd.DataFrame({"lag_1": [1.0, 2.0]})
    y = np.array([10.0, 11.0])
    model = RandomForestRegressor(n_estimators=5, random_state=0)
    model.fit(x, y)
    metadata = {"feature_names": ["lag_1"], "model_type": "random_forest"}
    save_artifact(
        model,
        metadata,
        models_dir=models_dir,
        version=version,
        keep_last_n=keep_last_n,
    )


def test_prune_keeps_last_n_versions(tmp_path) -> None:
    models_dir = str(tmp_path / "models")
    versions = [f"v{i:02d}" for i in range(7)]

    for idx, version in enumerate(versions):
        keep = 3 if idx == len(versions) - 1 else None
        _save_dummy(version, models_dir, keep_last_n=keep)

    remaining = list_versions(models_dir)
    assert remaining == ["v06", "v05", "v04"]
