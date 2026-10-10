#!/usr/bin/env bash
# Train tooth instance segmentation baseline
set -e
cd "$(dirname "$0")/.."
python -m src.train.train_tooth --config configs/tooth.yaml "$@"
python -m src.eval.evaluate_tooth --weights outputs/checkpoints/tooth/best.pt
