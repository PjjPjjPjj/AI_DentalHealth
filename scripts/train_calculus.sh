#!/usr/bin/env bash
# Train calculus segmentation baseline
set -e
cd "$(dirname "$0")/.."
python -m src.train.train_calculus --config configs/calculus.yaml "$@"
python -m src.eval.evaluate_calculus --weights outputs/checkpoints/calculus/best.pt --export-examples
