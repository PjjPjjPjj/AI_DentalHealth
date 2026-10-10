"""Image-level plaque screening evaluation (max-P aggregation over tooth crops).

Aggregation: image_score = max P(plaque) over tooth-seg crops (deployment
protocol: iop_toothseg @896 -> bbox crops -> PlaqueModel v2 with Platt
calibration). Also records mean / top3-mean for reference.

Sets:
  - Mendeley val base (162 imgs, GT labels = any plaque box)  -> threshold sweep
  - Mendeley test base (207 imgs, GT labels)                  -> primary test
  - SANO eval (46 imgs, all healthy by source label)          -> OOD-neg Spec

Output: outputs/plaque_image_level/eval_image_level.{json,md}
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
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
SANO = ROOT / "datasets/processed/plaque_sano"
SANO_RAW = ROOT / "datasets/raw/Plaque/SANO/images"
TOOTH_W = ROOT / "outputs/iop_toothseg/weights/best.pt"
PLAQUE_W = ROOT / "outputs/plaque_bin_v2_sano/weights/best.pt"
CALIB = ROOT / "outputs/plaque_bin_v2_sano/calibration_platt.npz"
PARTS = [ROOT / "datasets/raw/Plaque/mendeley-dataset-materials_Part_1",
         ROOT / "datasets/raw/Plaque/mendeley-dataset-materials_Part_2"]
OUT = ROOT / "outputs/plaque_image_level"


def find_image(image_id: str):
    patient = image_id.split("_")[0].lower()
    for part in PARTS:
        d = part / "data" / "images" / patient
        if d.is_dir():
            for f in d.iterdir():
                if f.stem.lower() == image_id.lower():
                    return f
    return None


def cm_image(y, s, t):
    pred = (s >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    return {"n_pos": tp + fn, "n_neg": tn + fp, "thr": t, "sens": round(sens, 4),
            "spec": round(spec, 4), "prec": round(prec, 4),
            "f1": round(2 * prec * sens / max(prec + sens, 1e-9), 4)}


def main() -> None:
    sys.path.insert(0, str(ROOT))
    from src.inference.models import ToothModel, PlaqueModel
    from sklearn.metrics import roc_auc_score

    tm = ToothModel(TOOTH_W)
    pm = PlaqueModel(PLAQUE_W, calibration=CALIB)

    rows = pd.read_csv(DATA1 / "plaque_crops.csv")
    base = rows[(rows.is_dup == 0) & (rows.is_aug == 0)]
    img_label = base.groupby("image_id").label.max().to_dict()

    sets = {}
    for split in ("val", "test"):
        ids = sorted(base[base.split == split].image_id.unique())
        sets[f"mendeley_{split}"] = [(i, int(img_label[i])) for i in ids]
    sano = pd.read_csv(SANO / "sano_crops.csv")
    sano_ev = sano[(sano.subset == "eval") & (sano.status == "kept")]
    sets["sano_eval"] = [(im, 0) for im in sorted(sano_ev.image.unique())]

    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for set_name, items in sets.items():
        recs = []
        for image_id, y in items:
            p = find_image(image_id) if set_name.startswith("mendeley") \
                else SANO_RAW / Path(image_id).name.replace(" ", "_")
            img = None if p is None else __import__("cv2").imread(str(p))
            if img is None:
                recs.append({"image_id": image_id, "y": y, "score_max": None, "n_teeth": 0,
                             "status": "no_image_or_teeth"})
                continue
            H, W = img.shape[:2]
            teeth = tm.predict(img)
            crops = []
            for t in teeth:
                x0, y0, x1, y1 = [int(max(0.0, v)) for v in t["bbox"]]
                x1, y1 = min(W, x1), min(H, y1)
                if x1 - x0 < 8 or y1 - y0 < 8:
                    continue
                crops.append(img[y0:y1, x0:x1].copy())
            if not crops:
                recs.append({"image_id": image_id, "y": y, "score_max": 0.0, "n_teeth": 0,
                             "status": "no_teeth"})
                continue
            probs = pm.predict_proba_batch(crops) if hasattr(pm, "predict_proba_batch") \
                else pm._proba_array(crops)
            a = np.array(sorted(probs, reverse=True))
            recs.append({"image_id": image_id, "y": y, "n_teeth": len(crops),
                         "score_max": round(float(a[0]), 4),
                         "score_mean": round(float(probs and np.mean(probs)), 4),
                         "score_top3": round(float(a[:3].mean()), 4), "status": "ok"})
        df = pd.DataFrame([r for r in recs if r["status"] == "ok"])
        if df.empty:
            raise RuntimeError(f"{set_name}: no scorable images — check paths")
        y = df.y.values.astype(int)
        s = df.score_max.values
        res = {"n_images": len(df), "n_excluded": len(recs) - len(df),
               "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
               "auroc": round(float(roc_auc_score(y, s)), 4) if len(set(y)) > 1 else None}
        # threshold sweep on val only
        if set_name == "mendeley_val":
            grid = np.round(np.arange(0.05, 0.99, 0.01), 2)
            sweep = []
            for t in grid:
                pred = (s >= t).astype(int)
                se = ((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)
                sp = ((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)
                sweep.append({"thr": float(t), "sens": round(se, 4), "spec": round(sp, 4)})
            feas = [w for w in sweep if w["sens"] >= 0.95]
            best = max(feas, key=lambda w: w["spec"]) if feas else max(sweep, key=lambda w: w["f1"])
            res["frozen_thr"] = best["thr"]
            res["sweep"] = sweep
        results[set_name] = {**res, "records": recs}

    tv = results["mendeley_val"]["frozen_thr"]
    summary = {}
    all_recs = []
    for k, r in results.items():
        for x in r["records"]:
            all_recs.append({"set": k, **x})
        y = np.array([x["y"] for x in r["records"] if x["status"] == "ok"])
        s = np.array([x["score_max"] for x in r["records"] if x["status"] == "ok"])
        summary[k] = {k2: v for k2, v in r.items() if k2 != "records"}
        summary[k]["at_frozen_thr"] = cm_image(y, s, tv)
        summary[k]["profile"] = {str(t): cm_image(y, s, t) for t in (0.5, 0.6, 0.7, 0.8, 0.9)}

    pd.DataFrame(all_recs).to_csv(OUT / "image_scores.csv", index=False)

    # quantile-freeze: thr = P5 of val POSITIVE image scores (sens>=0.95 by rule),
    # then report SANO Spec per aggregator (Mendeley val/test have no plaque-free
    # images -> image-level Spec is only measurable on external sets)
    agg_cmp = {}
    sc = pd.DataFrame(all_recs)
    for agg in ("score_max", "score_mean", "score_top3"):
        vp = sc[(sc.set == "mendeley_val") & (sc.status == "ok")]
        tp = sc[(sc.set == "mendeley_test") & (sc.status == "ok")]
        sa = sc[(sc.set == "sano_eval") & (sc.status == "ok")]
        thr_q = float(np.quantile(vp[vp.y == 1][agg].values, 0.05))
        agg_cmp[agg] = {
            "thr_val_pos_p5": round(thr_q, 4),
            "sens_val": round(float((vp[vp.y == 1][agg] >= thr_q).mean()), 4),
            "sens_test": round(float((tp[tp.y == 1][agg] >= thr_q).mean()), 4),
            "spec_sano": round(float((sa[agg] < thr_q).mean()), 4),
        }

    (OUT / "eval_image_level.json").write_text(json.dumps(
        {"frozen_thr": tv, "aggregation": "max P over tooth crops (calibrated)",
         "quantile_freeze_aggregators": agg_cmp,
         "note": "Mendeley val/test base images are ALL plaque-positive at image level "
                 "(by dataset design); image-level Spec is only measurable on external sets",
         "sets": summary}, indent=2, ensure_ascii=False))

    mv, mt, sa = summary["mendeley_val"], summary["mendeley_test"], summary["sano_eval"]
    m = ["# 牙菌斑图级筛查评估（v2_sano 校准分数）", "",
         "**数据事实**：Mendeley val/test 的 base 图在图级全部为阳性（每图≥1 个菌斑框，数据集设计如此），"
         "因此图级特异度只能在外部健康集（SANO 46 图）上度量；Mendeley 侧只报告图级敏感度。", "",
         "## 聚合器对比（thr = Mendeley val 阳性图分数 P5，按规则 Sens≥95%）",
         f"| 聚合器 | thr | val Sens | test Sens | SANO Spec |",
         f"|---|---|---|---|---|"]
    for agg, v in agg_cmp.items():
        m.append(f"| {agg} | {v['thr_val_pos_p5']} | {v['sens_val']} | {v['sens_test']} | **{v['spec_sano']}** |")
    m += ["",
          f"## max-P 在固定阈值下的 SANO Spec",
          f"| thr | 0.5 | 0.7 | 0.9 |",
          f"|---|---|---|---|",
          f"| SANO Spec | {sa['profile']['0.5']['spec']} | {sa['profile']['0.7']['spec']} | {sa['profile']['0.9']['spec']} |", "",
          "- 推荐：图分 = max 校准 P；部署图级 thr 取业务可接受的 Sens 档位（P5 规则≈0.95）",
          "- 无牙检出图计 0 分（no_teeth 计入 FN/tn）；每图分数明细见 image_scores.csv"]
    (OUT / "eval_image_level.md").write_text("\n".join(m) + "\n")
    print("\n".join(m))
    (OUT / "eval_image_level.md").write_text("\n".join(m) + "\n")
    print("\n".join(m))


if __name__ == "__main__":
    main()
