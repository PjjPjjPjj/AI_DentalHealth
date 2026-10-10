"""EXP11: Stage2 crop-position bias probe (VAL only, no training, same ckpt).

Baseline: 160x160 crop centered on the GT bbox center, resized to 640.
Shifts (single-axis, horizontal & vertical, both signs): 16/32/48 px.

Per configuration (VAL):
  - only samples with GT inclusion>=0.95 are counted (removes crop clipping)
  - stage2 (640, conf c2=0.05) -> raw instances + union, crop space
  - per bucket: Dice (union vs GT crop 640), IoU, Recall@IoU0.5
    (best single instance >=0.5), matched Dice (best instance),
    plus GT centroid normalized position (u,v) inside the crop
Outputs: outputs/calculus_yolo11m_twostage/exp11_position_bias.csv
+ Dice vs shift-distance tables printed.
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

from src.inference.twostage import TARGET, TwoStageSegmenter
from src.utils.paths import PROCESSED_DIR

DATA = PROCESSED_DIR / "calculus_seg"
RUN = Path("outputs/calculus_yolo11m_twostage")
BUCKET_EDGES = (0.00552, 0.01565)
BUCKETS = ("small", "medium", "large")
SIDE = 160
C2 = 0.05

SHIFTS = [(0, 0), (16, 0), (-16, 0), (32, 0), (-32, 0), (48, 0), (-48, 0),
          (0, 16), (0, -16), (0, 32), (0, -32), (0, 48), (0, -48)]


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

    stats = {s: {b: {"dice": [], "iou": [], "hit": [], "mdice": [],
                     "u": [], "v": []} for b in BUCKETS} for s in SHIFTS}
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
            cx_gt, cy_gt = gx + gw / 2.0, gy + gh / 2.0
            b = bucket_of(int(m.sum()) / float(H * W))
            # GT centroid for (u,v)
            ys, xs = np.nonzero(m)
            cxs, cys = float(xs.mean()), float(ys.mean())
            for dx, dy in SHIFTS:
                cx, cy = cx_gt + dx, cy_gt + dy
                x0 = int(round(cx - SIDE / 2)); y0 = int(round(cy - SIDE / 2))
                x0 = max(0, min(x0, W - SIDE)); y0 = max(0, min(y0, H - SIDE))
                x1 = x0 + SIDE; y1 = y0 + SIDE
                inside = int(m[y0:y1, x0:x1].sum())
                incl = inside / float(m.sum())
                if incl < 0.95:
                    continue
                crop = cv2.resize(img[y0:y1, x0:x1], (TARGET, TARGET),
                                  interpolation=cv2.INTER_AREA)
                insts = seg._stage2_instances(crop)
                gtc = cv2.resize(m[y0:y1, x0:x1].astype(np.uint8),
                                 (TARGET, TARGET), interpolation=cv2.INTER_AREA)
                gtc = (gtc > 0.5).astype(np.uint8)
                union = np.zeros((TARGET, TARGET), np.uint8)
                best = None
                best_io = 0.0
                for inst in insts:
                    mi = inst["bin640"]
                    union = np.maximum(union, mi)
                    ii = int((gtc & mi).sum())
                    uu = int((gtc | mi).sum())
                    v = ii / uu if uu else 0.0
                    if v > best_io:
                        best_io = v
                        best = mi
                inter = int((gtc & union).sum())
                uni = int((gtc | union).sum())
                dicc = 2 * inter / (int(gtc.sum()) + int(union.sum())) \
                    if (int(gtc.sum()) + int(union.sum())) else 1.0
                md = 0.0
                if best_io >= 0.5:
                    interb = int((gtc & best).sum())
                    unib = int((gtc | best).sum())
                    md = 2 * interb / (int(gtc.sum()) + int(best.sum())) \
                        if (int(gtc.sum()) + int(best.sum())) else 1.0
                st = stats[(dx, dy)][b]
                st["dice"].append(dicc)
                st["iou"].append(inter / uni if uni else 0.0)
                st["hit"].append(int(best_io >= 0.5))
                st["mdice"].append(md)
                st["u"].append((cxs - x0) / float(SIDE))
                st["v"].append((cys - y0) / float(SIDE))

    rows = []
    for (dx, dy) in SHIFTS:
        for b in BUCKETS:
            st = stats[(dx, dy)][b]
            n = len(st["dice"])
            if not n:
                rows.append({"dx": dx, "dy": dy, "bucket": b, "n": 0})
                continue
            rows.append({"dx": dx, "dy": dy, "bucket": b, "n": n,
                         "dice": round(float(np.mean(st["dice"])), 4),
                         "iou": round(float(np.mean(st["iou"])), 4),
                         "rec05": round(float(np.mean(st["hit"])), 4),
                         "matched_dice": round(float(np.mean(st["mdice"])), 4),
                         "u_mean": round(float(np.mean(st["u"])), 4),
                         "v_mean": round(float(np.mean(st["v"])), 4)})
    df = pd.DataFrame(rows)
    df.to_csv(RUN / "exp11_position_bias.csv", index=False)
    pd.set_option("display.width", 250)

    def tab(metric):
        return df.pivot_table(index=["dx", "dy"], columns="bucket",
                              values=metric).round(4)
    print("EXP11: Dice (union, crop-space, inclusion>=0.95 only)")
    print(tab("dice").to_string())
    print("\nEXP11: Recall@IoU0.5")
    print(tab("rec05").to_string())
    print("\nEXP11: matched Dice")
    print(tab("matched_dice").to_string())
    print("\nEXP11: mean u (horizontal frac) / mean v per bucket at shift=0:",
          df[(df.dx == 0) & (df.dy == 0)][["bucket", "u_mean", "v_mean"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
