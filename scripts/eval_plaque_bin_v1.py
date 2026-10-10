"""Evaluate plaque_bin_v1 (box-level binary classification).

Protocol (negatives are scarce -> layered sets):
  - clean  : unique base crops only          (is_dup=0, is_aug=0)
  - extended: clean + augmented-variant negative crops (is_dup=0, any is_aug)
  - AUROC on clean val/test; threshold frozen on extended val (max F1, also Youden J)
  - final metrics at frozen thr on val & test, clean primary / extended secondary

Output: outputs/plaque_bin_v1/eval_plaque_bin.json + eval_plaque_bin.md
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

ROOT = Path(f"{_ROOT}")
DATA = ROOT / "datasets/processed/plaque_bin_v1"
_CAND = [ROOT / "runs/classify/runs/classify/outputs/plaque_bin_v1/weights/best.pt",
         ROOT / "runs/classify/outputs/plaque_bin_v1/weights/best.pt"]
RUN = next(p for p in _CAND if p.exists())
OUT = ROOT / "outputs/plaque_bin_v1"


def main() -> None:
    from ultralytics import YOLO

    rows = []
    with open(DATA / "plaque_crops.csv", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    uniq = [r for r in rows if r["is_dup"] == "0"]
    val_rows = [r for r in uniq if r["split"] == "val"]
    test_rows = [r for r in uniq if r["split"] == "test"]
    val_clean = [r for r in val_rows if r["is_aug"] == "0"]
    test_clean = [r for r in test_rows if r["is_aug"] == "0"]

    model = YOLO(str(RUN))
    plaque_idx = [i for i, n in model.names.items() if n == "plaque"][0]

    def predict_probs(rows_):
        probs = []
        paths = [str(DATA / r["crop"]) for r in rows_]
        B = 128
        for i in range(0, len(paths), B):
            res = model.predict(paths[i:i + B], imgsz=224, verbose=False, device=0)
            for rr in res:
                probs.append(float(rr.probs.data[plaque_idx]))
        return np.array(probs)

    p_val = predict_probs(val_rows)
    p_test = predict_probs(test_rows)
    for r, p in zip(val_rows, p_val):
        r["p"] = p
    for r, p in zip(test_rows, p_test):
        r["p"] = p

    def y_true(rs):
        return np.array([int(r["label"]) for r in rs])

    # ---- threshold sweep on extended val ----
    yv = y_true(val_rows)
    pv = p_val
    grid = np.round(np.arange(0.02, 0.981, 0.01), 2)
    sweep = []
    for t in grid:
        pred = (pv >= t).astype(int)
        tp = int(((pred == 1) & (yv == 1)).sum())
        fp = int(((pred == 1) & (yv == 0)).sum())
        fn = int(((pred == 0) & (yv == 1)).sum())
        tn = int(((pred == 0) & (yv == 0)).sum())
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        prec = tp / max(tp + fp, 1)
        f1 = 2 * prec * sens / max(prec + sens, 1e-9)
        sweep.append({"thr": float(t), "sens": round(sens, 4), "spec": round(spec, 4),
                      "prec": round(prec, 4), "f1": round(f1, 4), "youden": round(sens + spec - 1, 4)})
    best_f1 = max(sweep, key=lambda s: s["f1"])
    best_j = max(sweep, key=lambda s: s["youden"])
    # screening-first operating point: max spec subject to sens >= 0.95 (extended val)
    feas = [s for s in sweep if s["sens"] >= 0.95]
    best_screen = max(feas, key=lambda s: (s["spec"], s["f1"])) if feas else best_f1
    thr = best_screen["thr"]

    def metrics(rs, t):
        y = y_true(rs)
        p = np.array([r["p"] for r in rs])
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        prec = tp / max(tp + fp, 1)
        f1 = 2 * prec * sens / max(prec + sens, 1e-9)
        return {"n_pos": tp + fn, "n_neg": tn + fp, "thr": t,
                "sens": round(sens, 4), "spec": round(spec, 4), "prec": round(prec, 4),
                "f1": round(f1, 4), "acc": round((tp + tn) / max(len(y), 1), 4)}

    from sklearn.metrics import roc_auc_score, average_precision_score
    auroc_val = float(roc_auc_score(y_true(val_clean), [r["p"] for r in val_clean]))
    auroc_test = float(roc_auc_score(y_true(test_clean), [r["p"] for r in test_clean]))
    ap_val = float(average_precision_score(y_true(val_clean), [r["p"] for r in val_clean]))
    ap_test = float(average_precision_score(y_true(test_clean), [r["p"] for r in test_clean]))

    # negative prob distribution (extended val/test)
    def neg_stats(rs):
        p = np.array([r["p"] for r in rs if r["label"] == "0"])
        if len(p) == 0:
            return {}
        return {"n": len(p), "median": round(float(np.median(p)), 3),
                "p25": round(float(np.percentile(p, 25)), 3), "p75": round(float(np.percentile(p, 75)), 3),
                "frac_lt_0.5": round(float((p < 0.5).mean()), 3), "frac_lt_0.9": round(float((p < 0.9).mean()), 3)}

    # per-FDI on test clean
    per_fdi = defaultdict(lambda: {"pos": 0, "neg": 0, "tp": 0, "tn": 0})
    for r in test_clean:
        k = r["fdi"]
        y = int(r["label"])
        pred = int(r["p"] >= thr)
        per_fdi[k]["pos" if y else "neg"] += 1
        if y == 1 and pred == 1:
            per_fdi[k]["tp"] += 1
        if y == 0 and pred == 0:
            per_fdi[k]["tn"] += 1

    rep = {
        "model": str(RUN),
        "frozen_thr_from": "extended val, max spec s.t. sens>=0.95 (screening-first)",
        "thr_screen": best_screen, "thr_f1": best_f1, "thr_youden": best_j,
        "sweep_extended_val": sweep,
        "auroc_val_clean": round(auroc_val, 4),
        "auroc_test_clean": round(auroc_test, 4),
        "ap_val_clean": round(ap_val, 4),
        "ap_test_clean": round(ap_test, 4),
        "neg_prob_stats": {"val_extended": neg_stats(val_rows), "test_extended": neg_stats(test_rows)},
        "val_clean_at_thr": metrics(val_clean, thr),
        "val_extended_at_thr": metrics(val_rows, thr),
        "test_clean_at_thr": metrics(test_clean, thr),
        "test_extended_at_thr": metrics(test_rows, thr),
        "per_fdi_test_clean": {k: v for k, v in sorted(per_fdi.items())},
        "counts": {"val_rows": len(val_rows), "val_clean": len(val_clean),
                   "test_rows": len(test_rows), "test_clean": len(test_clean)},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "eval_plaque_bin.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))

    m = ["# plaque_bin_v1 评估（框级二分类 YOLO11s-cls@224）", "",
         f"- 工作点: thr={thr}（extended val 上满足 Sens≥0.95 的最大 Spec；"
         f"F1 最优点 {best_f1['thr']}，Youden 最优点 {best_j['thr']}）",
         f"- AUROC: val {auroc_val:.4f} / test {auroc_test:.4f}；AP: val {ap_val:.4f} / test {ap_test:.4f}（干净 base 框）",
         f"- val(clean, n_pos={rep['val_clean_at_thr']['n_pos']}, n_neg={rep['val_clean_at_thr']['n_neg']}): "
         f"Sens {rep['val_clean_at_thr']['sens']}, Spec {rep['val_clean_at_thr']['spec']}, "
         f"P {rep['val_clean_at_thr']['prec']}, F1 {rep['val_clean_at_thr']['f1']}",
         f"- test(clean, n_pos={rep['test_clean_at_thr']['n_pos']}, n_neg={rep['test_clean_at_thr']['n_neg']}): "
         f"Sens {rep['test_clean_at_thr']['sens']}, Spec {rep['test_clean_at_thr']['spec']}, "
         f"P {rep['test_clean_at_thr']['prec']}, F1 {rep['test_clean_at_thr']['f1']}",
         f"- test(extended, n_neg={rep['test_extended_at_thr']['n_neg']}): "
         f"Sens {rep['test_extended_at_thr']['sens']}, Spec {rep['test_extended_at_thr']['spec']}",
         f"- 阴性框概率分布(extended): val {rep['neg_prob_stats']['val_extended']} / "
         f"test {rep['neg_prob_stats']['test_extended']}",
         f"- 注意: 阴性框极少（test clean {rep['test_clean_at_thr']['n_neg']} 个），"
         f"Spec 置信区间宽；阴性多为阳性牙内的子区域，低 Spec 部分源于标签语义。"]
    (OUT / "eval_plaque_bin.md").write_text("\n".join(m) + "\n")
    print("\n".join(m))


if __name__ == "__main__":
    sys.exit(main())
