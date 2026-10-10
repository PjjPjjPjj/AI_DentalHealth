#!/bin/bash
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$(dirname "$0")/.."
echo "[chain3] step 8: calibrate on val"
python3 scripts/calibrate_v3.py
echo "[chain3] step 9: final test"
python3 scripts/final_test_v3.py
echo "[chain3] ALL DONE"
