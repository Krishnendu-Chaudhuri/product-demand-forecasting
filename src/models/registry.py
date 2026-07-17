"""Model artifact persistence.

For remote backup, see scripts/upload_artifacts.py.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from sklearn.base import RegressorMixin

logger = logging.getLogger(__name__)


def _models_root(models_dir: str) -> Path:
    return Path(models_dir)


def generate_version() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def save_artifact(
    model: RegressorMixin,
    metadata: dict[str, Any],
    models_dir: str = "models",
    version: str | None = None,
    keep_last_n: int | None = None,
) -> Path:
    """Save model and metadata to a versioned directory."""
    version_id = version or generate_version()
    target_dir = _models_root(models_dir) / version_id
    target_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        **metadata,
        "version": version_id,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    model_path = target_dir / "model.joblib"
    metadata_path = target_dir / "metadata.json"

    joblib.dump(model, model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Saved model artifact to %s", target_dir)

    if keep_last_n is not None:
        _prune_old_versions(models_dir, keep_last_n)

    return target_dir


def _prune_old_versions(models_dir: str, keep_last_n: int) -> None:
    """Remove oldest model versions beyond the retention limit."""
    versions = list_versions(models_dir)
    if len(versions) <= keep_last_n:
        return

    root = _models_root(models_dir)
    for version_id in versions[keep_last_n:]:
        target_dir = root / version_id
        if target_dir.exists():
            for path in target_dir.iterdir():
                path.unlink()
            target_dir.rmdir()
            logger.info("Pruned old model version %s", version_id)


def list_versions(models_dir: str = "models") -> list[str]:
    root = _models_root(models_dir)
    if not root.exists():
        return []
    versions = sorted(
        [
            path.name
            for path in root.iterdir()
            if path.is_dir() and (path / "model.joblib").exists()
        ],
        reverse=True,
    )
    return versions


def load_artifact(version: str | None = None, models_dir: str = "models") -> dict[str, Any]:
    """Load model artifact and metadata."""
    versions = list_versions(models_dir)
    if not versions:
        raise FileNotFoundError(f"No saved models found in {models_dir}")

    version_id = version or versions[0]
    target_dir = _models_root(models_dir) / version_id
    model_path = target_dir / "model.joblib"
    metadata_path = target_dir / "metadata.json"

    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Model version not found: {version_id}")

    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {"model": model, "metadata": metadata, "path": target_dir}
