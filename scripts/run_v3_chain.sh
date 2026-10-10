#!/bin/bash
# V3 full pipeline chain: waits for stage1, then runs all remaining steps.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$(dirname "$0")/.."

echo "[chain] waiting for stage1 to finish ..."
while pgrep -f train_v3_stage1.py > /dev/null; do sleep 60; done
# make sure best.pt exists (training done or crashed)
until [ -f outputs/calculus_v3_stage1/weights/best.pt ]; do
  echo "[chain] stage1 best.pt missing; aborting"; exit 1
done
echo "[chain] stage1 done"

echo "[chain] step 2: stage1 val eval"
python3 scripts/v3_stage1_eval.py

echo "[chain] step 3: build crops"
python3 scripts/build_v3_crops.py

echo "[chain] step 4: train stage2"
python3 scripts/train_v3_stage2.py

echo "[chain] step 5: build verifier dataset"
python3 scripts/build_v3_verifier.py

echo "[chain] step 6: train verifier"
python3 scripts/train_v3_verifier.py

echo "[chain] step 7: train image classifier"
python3 scripts/train_v3_imgclf.py

echo "[chain] step 8: calibrate on val"
python3 scripts/calibrate_v3.py

echo "[chain] step 9: final test"
python3 scripts/final_test_v3.py

echo "[chain] ALL DONE"
