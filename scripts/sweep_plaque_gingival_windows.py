"""Sweep gingival-margin sub-window configs for plaque tooth-crop classification.

verify3 tested only [0,0,1,1] (full) vs [0,0,1,1]+[0.15,0.10,0.85,0.90] (center),
finding full-only best (AUROC 0.9164 on val). But that center window still covers
most of the crown. This sweep tests gingival-edge-biased windows (bottom 1/3,
top 1/3, four-edge 1/4) and max-aggregations, since plaque concentrates at the
gingival margin but image orientation varies (maxillary vs mandibular views).

Pairing protocol: identical to verify3 (GT box -> max-IoU segmented tooth,
IoU>=0.3). Runs on both val and test base images.
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
OUT = ROOT / "outputs/plaque_gingival_window_sweep"

# window configs to sweep. Each entry: (name, [list of fractional windows])
# predict_proba_windows takes max P over the listed windows.
WINDOW_CONFIGS = [
    ("full",              [[0.0, 0.0, 1.0, 1.0]]),
    ("bottom_third",      [[0.0, 0.67, 1.0, 1.0]]),
    ("top_third",         [[0.0, 0.0, 1.0, 0.33]]),
    ("top+bot_third",     [[0.0, 0.0, 1.0, 0.33], [0.0, 0.67, 1.0, 1.0]]),
    ("four_edge_quarter", [[0.0, 0.0, 1.0, 0.25], [0.0, 0.75, 1.0, 1.0],
                           [0.0, 0.0, 0.25, 1.0], [0.75, 0.0, 1.0, 1.0]]),
    ("center_60",         [[0.2, 0.2, 0.8, 0.8]]),
    ("full+bot_third",    [[0.0, 0.0, 1.0, 1.0], [0.0, 0.67, 1.0, 1.0]]),
    ("full+four_edge",    [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 0.25],
                           [0.0, 0.75, 1.0, 1.0], [0.0, 0.0, 0.25, 1.0],
                           [0.75, 0.0, 1.0, 1.0]]),
    ("full+topbot_third", [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 0.33],
                           [0.0, 0.67, 1.0, 1.0]]),
]


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


def collect_pairs(tm, rows, split: str):
    """Return list of (label, crop_img) for GT-box <-> tooth matches (IoU>=0.3)."""
    base = rows[(rows.split == split) & (rows.is_aug == 0)].copy()
    base = base[~base.image_id.str.contains("dup")]
    pairs = []
    for iid, grp in base.groupby("image_id"):
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
        lines = [l.split() for l in lbl.read_text().splitlines() if l.strip()]
        occ = defaultdict(int)
        grp = grp.sort_values("crop")
        for r in grp.itertuples():
            li = occ[(int(r.tooth_id), int(r.label))]
            hits = [l for l in lines if len(l) == 6
                    and int(l[5]) == int(r.tooth_id) and int(l[0]) == int(r.label)]
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
            pairs.append((int(r.label), crops_full[ki]))
    return pairs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pm = PlaqueModel(
        weights=str(ROOT / "outputs/plaque_bin_v2_sano/weights/best.pt"),
        imgsz=224, windows=[[0, 0, 1, 1]],
        calibration=str(ROOT / "outputs/plaque_bin_v2_sano/calibration_platt.npz"))
    tm = ToothModel(weights=str(ROOT / "outputs/iop_toothseg/weights/best.pt"),
                    conf=0.30, iou=0.70, imgsz=896)

    rows = pd.read_csv(DATA / "plaque_crops.csv")
    results = {}
    for split in ("val", "test"):
        pairs = collect_pairs(tm, rows, split)
        y = np.array([p[0] for p in pairs])
        crops = [p[1] for p in pairs]
        n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
        print(f"[{split}] {len(pairs)} pairs ({n_pos} pos / {n_neg} neg)")

        split_res = {"n_pairs": len(pairs), "n_pos": n_pos, "n_neg": n_neg}
        for name, windows in WINDOW_CONFIGS:
            probs = []
            for crop in crops:
                # build a temp PlaqueModel-like call with these windows
                pm.windows = windows
                probs.append(pm.predict_proba_windows(crop))
            pm.windows = [[0, 0, 1, 1]]  # reset
            probs = np.asarray(probs)
            has_both = len(set(y.tolist())) > 1
            auc = round(float(roc_auc_score(y, probs)), 4) if has_both else None
            ap = round(float(average_precision_score(y, probs)), 4) if has_both else None
            split_res[name] = {"auroc": auc, "ap": ap}
            print(f"  {name:22s} AUROC={auc}  AP={ap}")
        results[split] = split_res

    out = {"model": "plaque_bin_v2_sano + Platt", "pairing": "verify3 (IoU>=0.3)",
           "configs": [c[0] for c in WINDOW_CONFIGS], "results": results}
    (OUT / "sweep_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n=> {OUT / 'sweep_results.json'}")


if __name__ == "__main__":
    main()
