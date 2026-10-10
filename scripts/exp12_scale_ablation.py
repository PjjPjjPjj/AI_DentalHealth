"""EXP12: GT-centered crop SCALE ablation (VAL only, no Stage1, no training).

Fixed: model/ckpt frozen, c2=0.05, stage2 640, crop center = GT bbox center.
Variable: crop side {96,128,160,192,224,256}.

Per side per bucket (sample excluded if GT inclusion<0.95; n reported):
  1 n  2 Dice (current-unified semantic view: union vs GT instance)
  3 Recall@IoU0.5 (best instance)  4 matched Dice  5 mean IoU
  6 lesion_area/crop_area median P25 P75
  7 GT bbox (w/h/short-side) after resize to 640, median
Correlation: Dice vs lesion/crop-ratio ; Dice vs GT short-side px @640.
Outputs: outputs/calculus_yolo11m_twostage/exp12_scale_ablation.csv
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

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.inference.twostage import TARGET, TwoStageSegmenter
from src.utils.paths import PROCESSED_DIR

DATA = PROCESSED_DIR / "calculus_seg"
RUN = Path("outputs/calculus_yolo11m_twostage")
BUCKET_EDGES = (0.00552, 0.01565)
BUCKETS = ("small", "medium", "large")
SIDES = (96, 128, 160, 192, 224, 256)
C2 = 0.05


def bucket_of(ratio: float) -> str:
    if ratio < BUCKET_EDGES[0]:
        return "small"
    if ratio < BUCKET_EDGES[1]:
        return "medium"
    return "large"


def main() -> None:
    seg = TwoStageSegmenter("outputs/calculus_yolo11m_full/weights/best.pt",
                            "outputs/calculus_yolo11m_crop/weights/best.pt",
                            c1=0.05, c2=C2, device=0)
    man = pd.read_csv(DATA / "split_manifest.csv")
    rows = man[(man.split == "val") & man.has_calculus]

    per = {s: {b: {"dice": [], "sio": [], "hit": [], "mdice": [],
                   "ratio": [], "bw": [], "bh": [], "bs": []} for b in BUCKETS}
           for s in SIDES}
    for _, r in rows.iterrows():
        img = cv2.imread(r["image_path"])
        H, W = img.shape[:2]
        gt = cv2.imread(r["mask_original_path"], cv2.IMREAD_UNCHANGED) > 0
        if not gt.any():
            continue
        n, comp, _, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8), 8)
        for i in range(1, n):
            if (comp == i).sum() < 5:
                continue
            m = (comp == i).astype(np.uint8)
            gx, gy, gw, gh = cv2.boundingRect(m)
            px = int(m.sum())
            b = bucket_of(px / float(H * W))
            cx, cy = gx + gw / 2.0, gy + gh / 2.0
            for side in SIDES:
                x0 = int(round(cx - side / 2)); y0 = int(round(cy - side / 2))
                x0 = max(0, min(x0, W - side)); y0 = max(0, min(y0, H - side))
                x1, y1 = x0 + side, y0 + side
                incl = int(m[y0:y1, x0:x1].sum()) / float(px)
                if incl < 0.95:
                    continue
                crop = cv2.resize(img[y0:y1, x0:x1], (TARGET, TARGET),
                                  interpolation=cv2.INTER_AREA)
                insts = seg._stage2_instances(crop)
                tc = cv2.resize(m[y0:y1, x0:x1].astype(np.uint8),
                                (TARGET, TARGET), interpolation=cv2.INTER_AREA)
                tc = (tc > 0.5).astype(np.uint8)
                union = np.zeros((TARGET, TARGET), np.uint8)
                best_io, best_mi = 0.0, None
                for inst in insts:
                    mi = inst["bin640"]
                    union = np.maximum(union, mi)
                    ii = int((tc & mi).sum())
                    uu = int((tc | mi).sum())
                    v = ii / uu if uu else 0.0
                    if v > best_io:
                        best_io = v
                        best_mi = mi
                inter = int((tc & union).sum())
                uni = int((tc | union).sum())
                io = inter / uni if uni else 0.0
                dc = 2 * inter / (int(tc.sum()) + int(union.sum())) \
                    if (int(tc.sum()) + int(union.sum())) else 1.0
                st = per[side][b]
                st["dice"].append(dc)
                st["sio"].append(io)
                st["hit"].append(int(best_io >= 0.5))
                md = 0.0
                if best_mi is not None and best_io >= 0.5:
                    ii = int((tc & best_mi).sum())
                    uu = int((tc | best_mi).sum())
                    md = 2 * ii / (int(tc.sum()) + int(best_mi.sum())) \
                        if (int(tc.sum()) + int(best_mi.sum())) else 1.0
                st["mdice"].append(md)
                st["ratio"].append(px / float(side * side))
                sc = 640.0 / side
                st["bw"].append(int(gw) * sc)
                st["bh"].append(int(gh) * sc)
                st["bs"].append(min(int(gw), int(gh)) * sc)

    rows = []
    for side in SIDES:
        for b in BUCKETS:
            s = per[side][b]
            n = len(s["dice"])
            if not n:
                rows.append({"side": side, "bucket": b, "n": 0})
                continue
            rows.append({
                "side": side, "bucket": b, "n": n,
                "dice": round(float(np.mean(s["dice"])), 4),
                "iou": round(float(np.mean(s["sio"])), 4),
                "rec05": round(float(np.mean(s["hit"])), 4),
                "matched_dice": round(float(np.mean(s["mdice"])), 4),
                "ratio_med": round(float(np.median(s["ratio"])), 5),
                "ratio_p25": round(float(np.percentile(s["ratio"], 25)), 5),
                "ratio_p75": round(float(np.percentile(s["ratio"], 75)), 5),
                "bbox_w640_med": round(float(np.median(s["bw"])), 1),
                "bbox_h640_med": round(float(np.median(s["bh"])), 1),
                "short640_med": round(float(np.median(s["bs"])), 1),
            })
    df = pd.DataFrame(rows)
    df.to_csv(RUN / "exp12_scale_ablation.csv", index=False)

    # correlations per bucket pooled over sides
    corr = []
    for b in BUCKETS:
        d, rr, ss = [], [], []
        for side in SIDES:
            s = per[side][b]
            d += s["dice"]; rr += s["ratio"]; ss += s["bs"]
        d = np.array(d); rr = np.array(rr); ss = np.array(ss)
        k = len(d)
        c1 = spearmanr(d, rr).statistic if k > 2 else float("nan")
        c2 = spearmanr(d, ss).statistic if k > 2 else float("nan")
        corr.append({"bucket": b, "n": k,
                     "spearman_dice_vs_ratio": round(float(c1), 4),
                     "spearman_dice_vs_short640": round(float(c2), 4),
                     "pearson_dice_vs_ratio": round(float(np.corrcoef(d, rr)[0, 1]), 4),
                     "pearson_dice_vs_short640": round(float(np.corrcoef(d, ss)[0, 1]), 4)})
    cdf = pd.DataFrame(corr)
    cdf.to_csv(RUN / "exp12_scale_corr.csv", index=False)

    pd.set_option("display.width", 250)
    print("EXP12: GT-centered crop scale (uniform protocol, inclusion>=0.95)")
    print(df.pivot_table(index="side", columns="bucket",
                         values=["dice", "rec05", "matched_dice",
                                 "ratio_med"]).round(4).to_string())
    print("\nEXP12: GT bbox @640 median sizes (w/h/short)")
    print(df.pivot_table(index="side", columns="bucket",
                         values=["bbox_w640_med", "short640_med"]).round(1).to_string())
    print("\nEXP12 correlations (all sides pooled):")
    print(cdf.to_string(index=False))


if __name__ == "__main__":
    main()
