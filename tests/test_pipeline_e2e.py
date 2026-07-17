"""End-to-end pipeline integration tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PIPELINE = PROJECT_ROOT / "src" / "pipeline.py"
SAMPLE_CSV = PROJECT_ROOT / "tests" / "fixtures" / "sample_data.csv"


def test_pipeline_train_e2e(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    data_path = SAMPLE_CSV.resolve()

    cmd = [
        sys.executable,
        str(PIPELINE),
        "train",
        f"data.path={data_path}",
        f"registry.models_dir={models_dir}",
        "features=v2",
    ]
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout

    versions = sorted(models_dir.iterdir())
    assert len(versions) == 1
    version_dir = versions[0]
    assert (version_dir / "model.joblib").exists()
    assert (version_dir / "metadata.json").exists()

    metadata = json.loads((version_dir / "metadata.json").read_text(encoding="utf-8"))
    assert "metrics" in metadata
    for key in ("mae", "rmse", "mape", "r2"):
        assert key in metadata["metrics"]
    assert metadata["feature_config"]["version"] == "v2"
    assert not Path(metadata["data_path"]).is_absolute()
