"""Export a trained model artifact to ONNX format."""

from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.base import RegressorMixin

from models.registry import load_artifact


def _export_sklearn(model: RegressorMixin, feature_names: list[str], output_path: Path) -> None:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_type = [("input", FloatTensorType([None, len(feature_names)]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    output_path.write_bytes(onnx_model.SerializeToString())


def _export_xgboost(model: RegressorMixin, feature_names: list[str], output_path: Path) -> None:
    sample = np.zeros((1, len(feature_names)), dtype=np.float32)
    model.save_model(str(output_path.with_suffix(".json")))
    try:
        model.get_booster().save_model(str(output_path))
    except Exception:
        import onnxmltools
        from onnxmltools.convert import convert_xgboost

        onnx_model = convert_xgboost(model.get_booster(), initial_types=[("input", sample)])
        onnxmltools.utils.save_model(onnx_model, str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a model version to ONNX")
    parser.add_argument("--version", help="Model version to export")
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()

    artifact = load_artifact(version=args.version, models_dir=args.models_dir)
    model = artifact["model"]
    metadata = artifact["metadata"]
    model_type = metadata.get("model_type", "random_forest")
    feature_names: list[str] = metadata.get("feature_names", [])
    if not feature_names:
        raise SystemExit("Model metadata missing feature_names")

    output_path = Path(artifact["path"]) / "model.onnx"
    if model_type == "xgboost":
        _export_xgboost(model, feature_names, output_path)
    else:
        _export_sklearn(model, feature_names, output_path)

    print(f"exported {output_path}")


if __name__ == "__main__":
    main()
