"""Evaluate plaque_bin_v1 vs v2_sano under identical protocols.

For each model:
  1. official protocol (eval_plaque_bin_v1.py): thr frozen on extended val
     (max spec s.t. sens>=0.95) -> clean/extended val+test metrics
  2. SANO negative eval set (1,148 healthy tooth crops from 46 unseen images):
     Spec at the same frozen thr + P distribution + image-cluster bootstrap CI
  3. val top1 cross-check on the rebalanced val (sanity vs training curves)

Output: outputs/plaque_bin_v2_sano/eval_v1_vs_v2.json + .md
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

ROOT = Path(f"{_ROOT}")
DATA1 = ROOT / "datasets/processed/plaque_bin_v1"
DATA2 = ROOT / "datasets/processed/plaque_bin_v2"
SANO = ROOT / "datasets/processed/plaque_sano"
MODELS = {
    "v1": ROOT / "outputs/plaque_bin_v1/weights/best.pt",
    "v2_sano": ROOT / "runs/classify/runs/classify/outputs/plaque_bin_v2_sano/weights/best.pt",
}
OUT = ROOT / "outputs/plaque_bin_v2_sano"


def predict(model, paths, plaque_idx, bs=128):
    probs = []
    for i in range(0, len(paths), bs):
        res = model.predict(paths[i:i + bs], imgsz=224, verbose=False, device=0)
        for r in res:
            probs.append(float(r.probs.data[plaque_idx]))
    return np.array(probs)


def cls_metrics(y, p, t):
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    return {"n_pos": tp + fn, "n_neg": tn + fp, "thr": t, "sens": round(sens, 4),
            "spec": round(spec, 4), "prec": round(prec, 4),
            "f1": round(2 * prec * sens / max(prec + sens, 1e-9), 4)}


def cluster_bootstrap_spec(df, p, t, n_boot=2000, seed=42):
    """Spec CI with image-level cluster resampling (negatives only)."""
    rng = np.random.RandomState(seed)
    d = df[df.label == 0].reset_index(drop=True)
    neg_p = p[d.index.values]
    imgs = d.image.values
    clusters = defaultdict(list)
    for i, im in enumerate(imgs):
        clusters[im].append(i)
    keys = list(clusters)
    specs = []
    for _ in range(n_boot):
        pick = rng.choice(keys, len(keys), replace=True)
        idx = np.concatenate([clusters[k] for k in pick])
        tn = int((neg_p[idx] < t).sum())
        specs.append(tn / len(idx))
    return {"spec_mean": round(float(np.mean(specs)), 4),
            "ci95": [round(float(np.percentile(specs, 2.5)), 4),
                     round(float(np.percentile(specs, 97.5)), 4)]}


def main():
    from ultralytics import YOLO
    from sklearn.metrics import roc_auc_score, average_precision_score

    rows = pd.read_csv(DATA1 / "plaque_crops.csv")
    uniq = rows[rows.is_dup == 0]
    val_rows = uniq[uniq.split == "val"].reset_index(drop=True)
    test_rows = uniq[uniq.split == "test"].reset_index(drop=True)
    # keep original positions within val_rows/test_rows (index alignment for pv/pt)
    val_clean = val_rows[val_rows.is_aug == 0]
    test_clean = test_rows[test_rows.is_aug == 0]

    sano = pd.read_csv(SANO / "sano_crops.csv")
    sano_ev = sano[(sano.subset == "eval") & (sano.status == "kept")].reset_index(drop=True)
    sano_ev["label"] = 0  # SANO = healthy per source label + audit

    rep = {}
    for name, w in MODELS.items():
        model = YOLO(str(w))
        pidx = [i for i, n in model.names.items() if n == "plaque"][0]

        pv = predict(model, [str(DATA1 / c) for c in val_rows.crop], pidx)
        pt = predict(model, [str(DATA1 / c) for c in test_rows.crop], pidx)
        ps = predict(model, [str(SANO / c) for c in sano_ev.crop], pidx)

        # freeze thr on extended val: max spec s.t. sens>=0.95
        yv = val_rows.label.values.astype(int)
        grid = np.round(np.arange(0.02, 0.981, 0.01), 2)
        feas = []
        for t in grid:
            pred = (pv >= t).astype(int)
            sens = ((pred == 1) & (yv == 1)).sum() / max((yv == 1).sum(), 1)
            spec = ((pred == 0) & (yv == 0)).sum() / max((yv == 0).sum(), 1)
            feas.append((float(t), sens, spec))
        ok = [f for f in feas if f[1] >= 0.95]
        thr = max(ok, key=lambda f: (f[2], ))[0] if ok else 0.92

        r = {"weights": str(w), "frozen_thr": thr,
             "auroc_val_clean": round(float(roc_auc_score(val_clean.label, [
                 pv[i] for i in val_clean.index])), 4),
             "auroc_test_clean": round(float(roc_auc_score(test_clean.label, [
                 pt[i] for i in test_clean.index])), 4),
             "ap_test_clean": round(float(average_precision_score(test_clean.label, [
                 pt[i] for i in test_clean.index])), 4),
             "val_extended_at_thr": cls_metrics(yv, pv, thr),
             "test_clean_at_thr": cls_metrics(
                 test_clean.label.values.astype(int), pt[test_clean.index], thr),
             "test_extended_at_thr": cls_metrics(
                 test_rows.label.values.astype(int), pt, thr),
             "val_rebalanced_top1": round(float((
                 (pv >= 0.5).astype(int) == yv).mean()), 4),
             "sano_eval": cls_metrics(
                 sano_ev.label.values.astype(int), ps, thr),
             "sano_prob_stats": {
                 "median": round(float(np.median(ps)), 3),
                 "p25": round(float(np.percentile(ps, 25)), 3),
                 "p75": round(float(np.percentile(ps, 75)), 3),
                 "frac_ge_thr": round(float((ps >= thr).mean()), 3)},
             "sano_spec_bootstrap95": cluster_bootstrap_spec(sano_ev, ps, thr),
             "sano_fpr_per_image": {
                 "n_images": int(sano_ev.image.nunique()),
                 "images_with_any_fp": int(pd.Series(ps >= thr).groupby(
                     sano_ev.image.values).any().sum())},
             }
        rep[name] = r

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "eval_v1_vs_v2.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))

    v1, v2 = rep["v1"], rep["v2_sano"]
    m = ["# plaque_bin_v1 vs v2_sano（SANO 健康阴性微调）", "",
         f"- v1 thr={v1['frozen_thr']}  v2 thr={v2['frozen_thr']}（各自 extended val 冻结，Sens≥0.95 下最大 Spec）", "",
         "## 原协议（Mendeley 官方 test，不重划）",
         f"| 指标 | v1 | v2_sano |",
         f"|---|---|---|",
         f"| AUROC test clean | {v1['auroc_test_clean']} | {v2['auroc_test_clean']} |",
         f"| AP test clean | {v1['ap_test_clean']} | {v2['ap_test_clean']} |",
         f"| test clean Sens | {v1['test_clean_at_thr']['sens']} | {v2['test_clean_at_thr']['sens']} |",
         f"| test clean Spec (n_neg={v1['test_clean_at_thr']['n_neg']}) | {v1['test_clean_at_thr']['spec']} | {v2['test_clean_at_thr']['spec']} |",
         f"| test extended Spec (n_neg={v1['test_extended_at_thr']['n_neg']}) | {v1['test_extended_at_thr']['spec']} | {v2['test_extended_at_thr']['spec']} |",
         f"| test extended F1 | {v1['test_extended_at_thr']['f1']} | {v2['test_extended_at_thr']['f1']} |", "",
         "## SANO 健康阴性评估集（1,148 crop / 46 图，全阴性）",
         f"| 指标 | v1 | v2_sano |",
         f"|---|---|---|",
         f"| Spec @frozen thr | {v1['sano_eval']['spec']} | {v2['sano_eval']['spec']} |",
         f"| Spec 95% CI (image-cluster bootstrap) | {v1['sano_spec_bootstrap95']['ci95']} | {v2['sano_spec_bootstrap95']['ci95']} |",
         f"| P(plaque) 中位 | {v1['sano_prob_stats']['median']} | {v2['sano_prob_stats']['median']} |",
         f"| 误报图占比（任一 crop 误报） | {v1['sano_fpr_per_image']['images_with_any_fp']}/{v1['sano_fpr_per_image']['n_images']} | {v2['sano_fpr_per_image']['images_with_any_fp']}/{v2['sano_fpr_per_image']['n_images']} |", "",
         f"- val(重平衡) top1 对照：v1 {v1['val_rebalanced_top1']} / v2 {v2['val_rebalanced_top1']}（口径一致性检查）",
         "- SANO 标签为目录级弱标签（SANO=健康），已全量目检 225 图 + crop 抽检；"
         "eval 46 图与训练 144 图零近重复（dHash≤5 已剔除 12 对）"]
    (OUT / "eval_v1_vs_v2.md").write_text("\n".join(m) + "\n")
    print("\n".join(m))


if __name__ == "__main__":
    main()
