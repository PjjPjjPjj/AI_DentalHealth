"""Single-tooth-level plaque accuracy on Mendeley test (base) + SANO healthy.

Mirrors verify3_plaque_integration.py pairing logic, but on TEST base images
with the deployed v2_sano model + Platt calibration. Also reports SANO
single-tooth specificity (external healthy set) -- the gap left by verify3
which only evaluated val (n=282 pairs) and never ran test single-tooth level.

Outputs: outputs/plaque_test_toothlevel/{results.json, summary.md}
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

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, f"{_ROOT}")
from src.inference.models import PlaqueModel, ToothModel  # noqa: E402

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

ROOT = Path(f"{_ROOT}")
DATA = ROOT / "datasets/processed/plaque_bin_v1"
RAW_DIRS = [ROOT / "datasets/raw/Plaque/mendeley-dataset-materials_Part_1/data",
            ROOT / "datasets/raw/Plaque/mendeley-dataset-materials_Part_2/data"]
SANO_EVAL = ROOT / "datasets/processed/plaque_sano/eval"
OUT = ROOT / "outputs/plaque_test_toothlevel"

DEPLOY_THR_CALIB = 0.746   # calibrated scale (== raw 0.70)
DEPLOY_THR_RAW = 0.70


def find_image(image_id: str) -> Path | None:
    for d in RAW_DIRS:
        for img in (d / "images").glob(f"*/{image_id}.jpg"):
            return img
    return None


def find_label(image_id: str) -> Path | None:
    for d in RAW_DIRS:
        cands = list((d / "labels").glob(f"*/{image_id}.txt"))
        if cands:
            return cands[0]
    return None


def thr_metrics(y, p, thr):
    y = np.asarray(y); p = np.asarray(p)
    pred = p >= thr
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    prec = tp / max(1, tp + fp)
    f1 = 2 * prec * sens / max(1e-9, prec + sens)
    return dict(sens=round(sens, 4), spec=round(spec, 4),
                prec=round(prec, 4), f1=round(f1, 4),
                tp=tp, fp=fp, fn=fn, tn=tn)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pm = PlaqueModel(
        weights=str(ROOT / "outputs/plaque_bin_v2_sano/weights/best.pt"),
        imgsz=224, windows=[[0, 0, 1, 1]],
        calibration=str(ROOT / "outputs/plaque_bin_v2_sano/calibration_platt.npz"))
    tm = ToothModel(weights=str(ROOT / "outputs/iop_toothseg/weights/best.pt"),
                    conf=0.30, iou=0.70, imgsz=896)

    rows = pd.read_csv(DATA / "plaque_crops.csv")
    test = rows[(rows.split == "test") & (rows.is_aug == 0)].copy()
    test = test[~test.image_id.str.contains("dup")]

    # ---- test base: tooth-seg crop pairing (verify3 logic) ----
    y_raw, p_calib, p_raw, ious_all, n_teeth_total = [], [], [], [], 0
    n_img = n_match = n_gt = 0
    unmatched_reasons = defaultdict(int)
    for iid, grp in test.groupby("image_id"):
        img_path = find_image(iid)
        lbl = find_label(iid)
        if img_path is None or lbl is None:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        teeth = tm.predict(img)
        if not teeth:
            for _ in grp.itertuples():
                unmatched_reasons["no_teeth_detected"] += 1
            continue
        n_img += 1
        n_teeth_total += len(teeth)
        tb = np.array([t["bbox"] for t in teeth], float)
        areas = (tb[:, 2] - tb[:, 0]) * (tb[:, 3] - tb[:, 1])
        crops_full, keep_idx = [], []
        for ti in range(len(teeth)):
            x0, y0, x1, y1 = [int(max(0, v)) for v in tb[ti]]
            x1, y1 = min(W, x1), min(H, y1)
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            crops_full.append(img[y0:y1, x0:x1])
            keep_idx.append(ti)
        if not crops_full:
            continue
        probs_calib = pm._proba_array(crops_full)
        probs_raw = pm._proba_array_raw(crops_full)
        lines = [l.split() for l in lbl.read_text().splitlines() if l.strip()]
        occ = defaultdict(int)
        csv_occ = defaultdict(int)
        grp = grp.sort_values("crop")
        for r in grp.itertuples():
            n_gt += 1
            li = occ[(int(r.tooth_id), int(r.label))]
            hits = [l for l in lines if len(l) == 6
                    and int(l[5]) == int(r.tooth_id) and int(l[0]) == int(r.label)]
            if li >= len(hits):
                unmatched_reasons["gt_line_not_found"] += 1
                continue
            occ[(int(r.tooth_id), int(r.label))] += 1
            f = hits[li]
            cx, cy, w, h = (float(f[1]) * W, float(f[2]) * H,
                            float(f[3]) * W, float(f[4]) * H)
            gb = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
            ix0 = np.maximum(tb[:, 0], gb[0]); iy0 = np.maximum(tb[:, 1], gb[1])
            ix1 = np.minimum(tb[:, 2], gb[2]); iy1 = np.minimum(tb[:, 3], gb[3])
            iw = np.clip(ix1 - ix0, 0, None); ih = np.clip(iy1 - iy0, 0, None)
            inter = iw * ih
            union = areas + w * h - inter
            ious = np.where(union > 0, inter / union, 0)
            j = int(np.argmax(ious))
            if ious[j] < 0.3 or j not in keep_idx:
                unmatched_reasons["max_iou_below_0.3"] += 1
                continue
            ki = keep_idx.index(j)
            y_raw.append(int(r.label))
            p_calib.append(float(probs_calib[ki]))
            p_raw.append(float(probs_raw[ki]))
            ious_all.append(float(ious[j]))
            n_match += 1

    y_arr = np.asarray(y_raw)
    pc_arr = np.asarray(p_calib)
    pr_arr = np.asarray(p_raw)
    iou_arr = np.asarray(ious_all)

    has_both = len(set(y_arr.tolist())) > 1

    # pairing-quality stratification: high IoU (>=0.5) vs low (0.3-0.5)
    def _auc_subset(mask):
        ys, ps = y_arr[mask], pc_arr[mask]
        if len(set(ys.tolist())) < 2:
            return None
        return round(float(roc_auc_score(ys, ps)), 4)
    hi_mask = iou_arr >= 0.5
    lo_mask = (iou_arr >= 0.3) & (iou_arr < 0.5)
    strat = {
        "high_iou_ge0p5": {"n": int(hi_mask.sum()),
                           "pos": int((y_arr[hi_mask] == 1).sum()),
                           "neg": int((y_arr[hi_mask] == 0).sum()),
                           "auroc": _auc_subset(hi_mask)},
        "low_iou_0p3_0p5": {"n": int(lo_mask.sum()),
                            "pos": int((y_arr[lo_mask] == 1).sum()),
                            "neg": int((y_arr[lo_mask] == 0).sum()),
                            "auroc": _auc_subset(lo_mask)},
    }

    test_result = {
        "set": "mendeley_test_base",
        "n_images": n_img,
        "n_gt_boxes": n_gt,
        "n_teeth_segmented": n_teeth_total,
        "mean_teeth_per_image": round(n_teeth_total / max(1, n_img), 1),
        "n_matched_pairs": n_match,
        "match_rate": round(n_match / max(1, n_gt), 4),
        "unmatched": dict(unmatched_reasons),
        "matched_iou_median": round(float(np.median(iou_arr)), 4) if len(iou_arr) else None,
        "matched_iou_mean": round(float(np.mean(iou_arr)), 4) if len(iou_arr) else None,
        "label_dist": {"pos": int((y_arr == 1).sum()), "neg": int((y_arr == 0).sum())},
        "auroc_calibrated": round(float(roc_auc_score(y_arr, pc_arr)), 4) if has_both else None,
        "auroc_raw": round(float(roc_auc_score(y_arr, pr_arr)), 4) if has_both else None,
        "ap_calibrated": round(float(average_precision_score(y_arr, pc_arr)), 4) if has_both else None,
        "ap_raw": round(float(average_precision_score(y_arr, pr_arr)), 4) if has_both else None,
        "at_deploy_thr_calib_0p746": thr_metrics(y_arr, pc_arr, DEPLOY_THR_CALIB) if has_both else None,
        "at_deploy_thr_raw_0p70": thr_metrics(y_arr, pr_arr, DEPLOY_THR_RAW) if has_both else None,
        "iou_stratified_auroc": strat,
    }

    # ---- SANO healthy: single-tooth specificity ----
    # SANO eval dir already contains pre-cropped single-tooth images (1148
    # crops, ~400-500px), NOT full images. Run the classifier directly on
    # each crop (these ARE the single-tooth units); no tooth segmentation.
    sano_y, sano_pc, sano_pr = [], [], []
    sano_crops_paths = sorted(SANO_EVAL.glob("*.jpg")) + sorted(SANO_EVAL.glob("*.JPG"))
    sano_imgs = []
    for cp in sano_crops_paths:
        img = cv2.imread(str(cp))
        if img is None:
            continue
        sano_imgs.append(img)
    if sano_imgs:
        # batch in chunks to bound memory
        BATCH = 128
        for i in range(0, len(sano_imgs), BATCH):
            chunk = sano_imgs[i:i + BATCH]
            pc = pm._proba_array(chunk)
            pr = pm._proba_array_raw(chunk)
            for v in pc:
                sano_y.append(0); sano_pc.append(float(v))
            for v in pr:
                sano_pr.append(float(v))

    sy = np.asarray(sano_y)
    spc = np.asarray(sano_pc)
    spr = np.asarray(sano_pr)
    sano_result = {
        "set": "sano_healthy_eval",
        "n_crops": int(len(sy)),
        "all_negative": True,
        "single_tooth_spec_calib_0p746": thr_metrics(sy, spc, DEPLOY_THR_CALIB),
        "single_tooth_spec_raw_0p70": thr_metrics(sy, spr, DEPLOY_THR_RAW),
        "prob_calib_median": round(float(np.median(spc)), 4) if len(spc) else None,
        "prob_calib_p95": round(float(np.percentile(spc, 95)), 4) if len(spc) else None,
        "prob_raw_median": round(float(np.median(spr)), 4) if len(spr) else None,
        "n_fp_calib": int((spc >= DEPLOY_THR_CALIB).sum()),
        "n_fp_raw": int((spr >= DEPLOY_THR_RAW).sum()),
    }

    out = {"test_tooth_level": test_result, "sano_tooth_level": sano_result,
           "model": "plaque_bin_v2_sano + Platt calibration",
           "tooth_model": "iop_toothseg 11m-seg @896 conf0.30 iou0.70",
           "pairing": "GT box -> max-IoU segmented tooth, IoU>=0.3 (verify3 protocol)",
           "note": "test base only (is_aug==0); SANO eval = external healthy single-tooth specificity"}
    (OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
