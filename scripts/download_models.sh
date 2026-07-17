#!/usr/bin/env bash
# Regenerates the production model (v2 features, bounded Random Forest).
set -euo pipefail
cd "$(dirname "$0")/.."
python main.py train
