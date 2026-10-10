"""Verify 3: plaque integration loss quantification (Mendeley val).

(a) direct:   plaque model on GT region crops       (reference ~AUROC 0.868)
(b) pipeline: tooth crops from iop_toothseg on the original val images; each GT
    region is matched to the max-IoU pipeline tooth (IoU>=0.3) and scored by
    that tooth's crop probability (full crop and multi-window variants).

Reports AUROC (a) vs (b) + n matched pairs.
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
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, f"{_ROOT}")
from src.inference.models import PlaqueModel, ToothModel  # noqa: E402

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

ROOT = Path(f"{_ROOT}")
DATA = ROOT / "datasets/processed/plaque_bin_v1"
RAW_DIRS = [ROOT / "datasets/raw/Plaque/mendeley-dataset-materials_Part_1/data",
            ROOT / "datasets/raw/Plaque/mendeley-dataset-materials_Part_2/data"]
OUT = ROOT / "outputs/pipeline_v2_validation"


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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pm = PlaqueModel(weights=str(ROOT / "outputs/plaque_bin_v1/weights/best.pt"),
                     imgsz=224, windows=[[0, 0, 1, 1]])
    pm_win = PlaqueModel(weights=str(ROOT / "outputs/plaque_bin_v1/weights/best.pt"),
                         imgsz=224,
                         windows=[[0, 0, 1, 1], [0.15, 0.10, 0.85, 0.90]])
    tm = ToothModel(weights=str(ROOT / "outputs/iop_toothseg/weights/best.pt"),
                    conf=0.3, iou=0.7, imgsz=896)

    rows = pd.read_csv(DATA / "plaque_crops.csv")
    val = rows[(rows.split == "val") & (rows.is_aug == 0)]
    val = val[~val.image_id.str.contains("dup")]

    # (a) direct GT crops
    import cv2
    ya, pa = [], []
    for r in val.itertuples():
        img = cv2.imread(str(DATA / r.crop))
        if img is None:
            continue
        ya.append(int(r.label))
        pa.append(pm.predict_proba(img))
    auroc_a = float(roc_auc_score(ya, pa))

    # (b) pipeline tooth crops
    yb_full, pb_full, yb_win, pb_win = [], [], [], []
    n_img = n_match = 0
    for iid, grp in val.groupby("image_id"):
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
            continue
        n_img += 1
        tb = np.array([t["bbox"] for t in teeth])
        areas = (tb[:, 2] - tb[:, 0]) * (tb[:, 3] - tb[:, 1])
        crops_full = []
        crops_win = []
        keep_idx = []
        for ti, t in enumerate(teeth):
            x0, y0, x1, y1 = [int(max(0, v)) for v in tb[ti]]
            x1, y1 = min(W, x1), min(H, y1)
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            crops_full.append(img[y0:y1, x0:x1])
            keep_idx.append(ti)
        if not crops_full:
            continue
        probs_full = pm._proba_array(crops_full)
        probs_win = [pm_win.predict_proba_windows(c) for c in crops_full]
        for r in grp.itertuples():
            # GT region box in original image: recompute from label line order
            pass
        # GT regions from raw label file
        lines = [l.split() for l in lbl.read_text().splitlines() if l.strip()]
        # index GT lines by (tid, occurrence) to match csv crop order
        from collections import defaultdict
        occ = defaultdict(int)
        csv_occ = defaultdict(int)
        grp = grp.sort_values("crop")
        for r in grp.itertuples():
            k = (int(r.tooth_id), csv_occ[(int(r.tooth_id), r.label)])
            csv_occ[(int(r.tooth_id), r.label)] += 1
            # find matching label line: same tooth id and same flag, next unused
            li = occ[(int(r.tooth_id), int(r.label))]
            hits = [l for l in lines if len(l) == 6 and int(l[5]) == int(r.tooth_id)
                    and int(l[0]) == int(r.label)]
            if li >= len(hits):
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
                continue
            ki = keep_idx.index(j)
            yb_full.append(int(r.label)); pb_full.append(probs_full[ki])
            yb_win.append(int(r.label)); pb_win.append(probs_win[ki])
            n_match += 1
    out = {
        "direct_gt_crops": {"n": len(ya), "auroc": round(auroc_a, 4)},
        "pipeline_tooth_crops_full": {
            "n": len(yb_full),
            "auroc": round(float(roc_auc_score(yb_full, pb_full)), 4) if len(set(yb_full)) > 1 else None},
        "pipeline_tooth_crops_multiwin": {
            "n": len(yb_win),
            "auroc": round(float(roc_auc_score(yb_win, pb_win)), 4) if len(set(yb_win)) > 1 else None},
        "val_images_used": n_img, "matched_pairs": n_match,
    }
    (OUT / "plaque_integration_loss.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
