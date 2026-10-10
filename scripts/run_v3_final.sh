#!/bin/bash
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$(dirname "$0")/.."
python3 scripts/final_test_v3.py
