#!/usr/bin/env bash
# Train caries detection baseline (first train, validates the env)
set -e
cd "$(dirname "$0")/.."
python -m src.train.train_caries --config configs/caries.yaml "$@"
python -m src.eval.evaluate_caries --weights outputs/checkpoints/caries/best.pt --export-examples
