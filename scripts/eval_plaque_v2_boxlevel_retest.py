"""Box-level re-test on the ORIGINAL Mendeley test set with the deployed stack:
v2_sano + Platt calibration (deployed threshold 0.746 calibrated == raw 0.70).

Reports clean (n=56) and extended (n=420) negative sets, full threshold profile,
and per-FDI breakdown at the deployed point.
Output: outputs/plaque_bin_v2_sano/boxlevel_retest_calibrated.{json,md}
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

ROOT = Path(f"{_ROOT}")
DATA1 = ROOT / "datasets/processed/plaque_bin_v1"
PLAQUE_W = ROOT / "outputs/plaque_bin_v2_sano/weights/best.pt"
CALIB = ROOT / "outputs/plaque_bin_v2_sano/calibration_platt.npz"
DEPLOY_THR = 0.746  # calibrated scale (== raw 0.70)
OUT = ROOT / "outputs/plaque_bin_v2_sano"


def cm(y, p, t):
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1); prec = tp / max(tp + fp, 1)
    return {"n_pos": tp + fn, "n_neg": tn + fp, "thr": t, "sens": round(sens, 4),
            "spec": round(spec, 4), "prec": round(prec, 4), "fp": fp, "fn": fn,
            "f1": round(2 * prec * sens / max(prec + sens, 1e-9), 4)}


def main() -> None:
    sys.path.insert(0, str(ROOT))
    from src.inference.models import PlaqueModel
    from sklearn.metrics import roc_auc_score, average_precision_score

    pm = PlaqueModel(PLAQUE_W, calibration=CALIB)
    rows = pd.read_csv(DATA1 / "plaque_crops.csv")
    uniq = rows[rows.is_dup == 0]
    test = uniq[uniq.split == "test"].reset_index(drop=True)
    test_clean = test[test.is_aug == 0]

    p = np.array(pm._proba_array([str(DATA1 / c) for c in test.crop]))
    y = test.label.values.astype(int)

    res = {
        "model": str(PLAQUE_W), "calibration": str(CALIB), "deploy_thr_calibrated": DEPLOY_THR,
        "auroc_test_clean": round(float(roc_auc_score(
            test_clean.label, [p[i] for i in test_clean.index])), 4),
        "ap_test_clean": round(float(average_precision_score(
            test_clean.label, [p[i] for i in test_clean.index])), 4),
        "deploy_clean": cm(y[test_clean.index], p[test_clean.index], DEPLOY_THR),
        "deploy_extended": cm(y, p, DEPLOY_THR),
        "profile_extended": {str(t): cm(y, p, t) for t in
                             (0.3, 0.4, 0.5, 0.6, 0.7, 0.746, 0.8, 0.9, 0.95)},
    }
    # per-FDI at deploy thr
    per_fdi = defaultdict(lambda: {"pos": 0, "neg": 0, "fp": 0, "fn": 0})
    for i in range(len(test)):
        r = test.iloc[i]
        k = str(r.fdi); yy = y[i]; pred = int(p[i] >= DEPLOY_THR)
        per_fdi[k]["pos" if yy else "neg"] += 1
        if yy == 0 and pred == 1: per_fdi[k]["fp"] += 1
        if yy == 1 and pred == 0: per_fdi[k]["fn"] += 1
    res["per_fdi_extended"] = {k: v for k, v in sorted(per_fdi.items())}
    res["records"] = test[["crop", "label", "is_aug", "fdi"]].assign(p=np.round(p, 4)).to_dict("records")

    (OUT / "boxlevel_retest_calibrated.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    d, e = res["deploy_clean"], res["deploy_extended"]
    m = ["# 原始 test 框级复测（v2_sano + Platt 校准，部署 thr=0.746 校准刻度 ≡ raw 0.70）", "",
         f"- AUROC {res['auroc_test_clean']} / AP {res['ap_test_clean']}（clean 框，与迭代1一致——接线无回退）",
         f"- clean (n_neg={d['n_neg']}): Sens {d['sens']} / Spec {d['spec']} / FP {d['fp']}",
         f"- extended (n_neg={e['n_neg']}): Sens {e['sens']} / Spec {e['spec']} / FP {e['fp']} / F1 {e['f1']}", "",
         "| thr(校准) | Sens | Spec | FP |",
         "|---|---|---|---|"]
    for t, v in res["profile_extended"].items():
        m.append(f"| {t} | {v['sens']} | {v['spec']} | {v['fp']} |")
    (OUT / "boxlevel_retest_calibrated.md").write_text("\n".join(m) + "\n")
    print("\n".join(m))


if __name__ == "__main__":
    sys.exit(main())
