"""Versioned feature configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class FeatureConfig(BaseModel):
    version: str = "v2"
    lags: list[int] = Field(default_factory=lambda: [1, 2, 3, 4])
    rolling_windows: list[int] = Field(default_factory=list)
    static_columns: list[str] = Field(
        default_factory=lambda: [
            "total_price",
            "base_price",
            "is_featured_sku",
            "is_display_sku",
        ]
    )
    derived_features: list[str] = Field(default_factory=lambda: ["price_diff"])

    @property
    def feature_names(self) -> list[str]:
        names = [f"lag_{lag}" for lag in self.lags]
        for window in self.rolling_windows:
            names.append(f"roll_mean_{window}")
        names.extend(self.static_columns)
        names.extend(self.derived_features)
        return names


def load_feature_config(path: str | Path) -> FeatureConfig:
    """Load feature config from YAML."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Feature config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}

    return FeatureConfig.model_validate(raw)
