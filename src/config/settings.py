"""Application settings and configuration models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from features.config import FeatureConfig


class DataConfig(BaseModel):
    path: str = "assets/data.csv"
    granularity: Literal["store_sku", "store_only"] = "store_sku"


class SplitConfig(BaseModel):
    test_size_pct: float = Field(default=0.15, ge=0.01, le=0.5)
    cutoff_date: str | None = None


class ModelHyperparams(BaseModel):
    random_forest: dict[str, Any] = Field(default_factory=dict)
    xgboost: dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    type: Literal["random_forest", "xgboost"] = "random_forest"
    random_state: int = 0
    hyperparams: ModelHyperparams = Field(default_factory=ModelHyperparams)


class TuningConfig(BaseModel):
    enabled: bool = False
    n_iter: int = 10
    cv: int = 3
    grid: dict[str, Any] = Field(default_factory=dict)


class RegistryConfig(BaseModel):
    models_dir: str = "models"
    keep_last_n: int = Field(default=5, ge=1)


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class UiConfig(BaseModel):
    auth_enabled: bool = False
    auth_password: str | None = None
    users: dict[str, str] = Field(default_factory=dict)


class SecurityConfig(BaseModel):
    api_key: str | None = None
    rate_limit_per_minute: int = Field(default=0, ge=0)


class AppConfig(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    registry: RegistryConfig = Field(default_factory=RegistryConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    ui: UiConfig = Field(default_factory=UiConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> AppConfig:
        return cls.model_validate(cfg)

    def resolve_data_path(self, project_root: Path | None = None) -> Path:
        root = project_root or Path.cwd()
        path = Path(self.data.path)
        if path.is_absolute():
            return path
        return root / path
