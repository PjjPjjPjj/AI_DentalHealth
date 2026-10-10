#!/usr/bin/env bash
# Train plaque classification baseline
set -e
cd "$(dirname "$0")/.."
python -m src.train.train_plaque --config configs/plaque.yaml "$@"
python -m src.eval.evaluate_plaque --weights outputs/checkpoints/plaque/best.pt
