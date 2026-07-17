"""Hydra CLI for training and evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from config.loader import hydra_config_to_app
from pipelines.runner import run_evaluate, run_train
from utils.logging import configure_logging

_COMMAND = "train"
_CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    configure_logging(cfg.logging.level, cfg.logging.format)
    app_cfg = hydra_config_to_app(cfg)

    tune = bool(cfg.tuning.enabled)
    model_version: str | None = cfg.get("model_version")
    if model_version is not None:
        model_version = str(model_version)
    command = cfg.get("command", _COMMAND)

    if command == "evaluate":
        run_evaluate(app_cfg, model_version=model_version)
    else:
        run_train(app_cfg, tune=tune)


def _prepare_hydra_argv() -> None:
    global _COMMAND

    if "--tune" in sys.argv:
        sys.argv.remove("--tune")
        sys.argv.append("tuning.enabled=true")

    for arg in list(sys.argv[1:]):
        if arg.startswith("model_version="):
            version = arg.split("=", 1)[1]
            sys.argv.remove(arg)
            sys.argv.append(f'+model_version="{version}"')

    for arg in list(sys.argv[1:]):
        if arg in {"train", "evaluate"}:
            _COMMAND = arg
            sys.argv.remove(arg)
            break

    if _COMMAND == "evaluate":
        sys.argv.append("+command=evaluate")
