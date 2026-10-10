#!/bin/bash
# V3 chain continuation: verifier -> imgclf -> calibrate -> final test
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$(dirname "$0")/.."

echo "[chain2] step 6: train verifier"
python3 scripts/train_v3_verifier.py

echo "[chain2] step 7: train image classifier"
python3 scripts/train_v3_imgclf.py

echo "[chain2] step 8: calibrate on val"
python3 scripts/calibrate_v3.py

echo "[chain2] step 9: final test"
python3 scripts/final_test_v3.py

echo "[chain2] ALL DONE"
