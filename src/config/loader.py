"""Hydra configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from config.settings import AppConfig


def hydra_config_to_app(cfg: DictConfig) -> AppConfig:
    """Convert a composed Hydra DictConfig to AppConfig."""
    container = OmegaConf.to_container(cfg, resolve=True)
    return AppConfig.from_dict(cast(dict[str, Any], container))


def load_app_config(project_root: Path | None = None) -> AppConfig:
    """Load and compose the main Hydra config with defaults."""
    root = project_root or Path.cwd()
    config_dir = root / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config")
    return hydra_config_to_app(cfg)
