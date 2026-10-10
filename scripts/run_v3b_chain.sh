#!/bin/bash
# V3b chain: stage1b -> crops -> stage2b -> verifier -> imgclf b -> calibrate -> final
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$(dirname "$0")/.."

echo "[v3b] 1: train stage1b"
python3 scripts/train_v3b_stage1.py

echo "[v3b] 2: build crops (from v3b stage1)"
python3 scripts/build_v3_crops.py

echo "[v3b] 3: train stage2b"
python3 scripts/train_v3_stage2.py

echo "[v3b] 4: build verifier dataset"
python3 scripts/build_v3_verifier.py

echo "[v3b] 5: train verifier"
python3 scripts/train_v3_verifier.py

echo "[v3b] 6: train image classifier (448, domain-weighted)"
python3 scripts/train_v3_imgclf.py

echo "[v3b] 7: calibrate on val"
python3 scripts/calibrate_v3.py

echo "[v3b] 8: final test"
python3 scripts/final_test_v3.py

echo "[v3b] ALL DONE"
