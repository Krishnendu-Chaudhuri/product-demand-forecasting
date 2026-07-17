"""Capture or verify baseline prediction from the production model."""

from __future__ import annotations

import pandas as pd

from models.predict import predict
from models.registry import load_artifact

VERSION = "20260714_181954"
MODELS_DIR = "models"

SAMPLE_ROW = {
    "lag_1": 10.0,
    "lag_2": 8.0,
    "lag_3": 6.0,
    "lag_4": 4.0,
    "total_price": 198.0,
    "base_price": 200.0,
    "is_featured_sku": 0.0,
    "is_display_sku": 0.0,
    "price_diff": -2.0,
}


def main() -> None:
    artifact = load_artifact(version=VERSION, models_dir=MODELS_DIR)
    model = artifact["model"]
    feature_names: list[str] = artifact["metadata"]["feature_names"]
    x = pd.DataFrame([{name: SAMPLE_ROW[name] for name in feature_names}])
    pred = float(predict(model, x)[0])
    print(f"version={VERSION}")
    print(f"baseline_prediction={pred}")


if __name__ == "__main__":
    main()
