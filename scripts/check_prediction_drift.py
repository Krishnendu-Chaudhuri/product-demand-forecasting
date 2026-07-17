"""Compare recent prediction logs against training target distribution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _load_predictions(lines: list[str]) -> list[float]:
    predictions: list[float] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("event") == "prediction":
            predictions.append(float(payload["prediction"]))
    return predictions


def _load_target_stats(metadata_path: Path) -> tuple[float, float]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    stats = metadata.get("target_stats", {})
    return float(stats["mean"]), float(stats["std"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Check prediction drift against training stats")
    parser.add_argument(
        "--metadata",
        default="models",
        help="Path to metadata.json or models directory containing versions",
    )
    parser.add_argument(
        "--version",
        help="Model version when --metadata points to models/",
    )
    parser.add_argument(
        "--log-file",
        help="Log file with JSON prediction events; defaults to stdin",
    )
    parser.add_argument("--sigma-threshold", type=float, default=2.0)
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    if metadata_path.is_dir():
        versions = sorted(
            [path for path in metadata_path.iterdir() if path.is_dir()],
            reverse=True,
        )
        if args.version:
            metadata_path = metadata_path / args.version / "metadata.json"
        elif versions:
            metadata_path = versions[0] / "metadata.json"
        else:
            raise SystemExit("No model versions found")

    target_mean, target_std = _load_target_stats(metadata_path)
    if args.log_file:
        lines = Path(args.log_file).read_text(encoding="utf-8").splitlines()
    else:
        lines = sys.stdin.read().splitlines()

    predictions = _load_predictions(lines)
    if not predictions:
        raise SystemExit("No prediction events found in log input")

    pred_mean = float(np.mean(predictions))
    pred_std = float(np.std(predictions))
    mean_shift = abs(pred_mean - target_mean)
    drift_detected = target_std > 0 and mean_shift > args.sigma_threshold * target_std

    print(f"target_mean={target_mean:.4f} target_std={target_std:.4f}")
    print(f"prediction_mean={pred_mean:.4f} prediction_std={pred_std:.4f}")
    if drift_detected:
        print("DRIFT_DETECTED: prediction mean shifted beyond threshold")
        raise SystemExit(1)
    print("No significant drift detected")


if __name__ == "__main__":
    main()
