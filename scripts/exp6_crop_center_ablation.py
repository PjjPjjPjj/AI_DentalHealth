"""EXP6: stage2 crop-center strategy ablation (VAL only).

Strategies (center of the window):
  bbox : stage1 prediction box xyxy center (current behavior)
  mbbox: center of stage1 prediction mask bounding-box
  mcent: centroid of stage1 prediction mask

Fixed: c1=0.05, c2=0.05, MARGIN=1.8, MIN_WINDOW=160, stage2 imgsz=640,
fusion = EXP5 winner (instance-level mask_nms @IoU0.5, score=s2_conf).
Outputs: outputs/calculus_yolo11m_twostage/exp6_crop_center_ablation.csv
"""
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

from src.inference.fusion import fuse_mask, greedy_nms, rec_bin
from src.inference.twostage import TwoStageSegmenter
from src.utils.paths import PROCESSED_DIR

DATA = PROCESSED_DIR / "calculus_seg"
RUN = Path("outputs/calculus_yolo11m_twostage")
BUCKET_EDGES = (0.00552, 0.01565)
BUCKETS = ("small", "medium", "large")
CENTER_MODES = ("bbox", "mbbox", "mcent")
NMS_IOU, SCORE_MODE = 0.5, "s2"


def bucket_of(ratio: float) -> str:
    if ratio < BUCKET_EDGES[0]:
        return "small"
    if ratio < BUCKET_EDGES[1]:
        return "medium"
    return "large"


def main() -> None:
    seg = TwoStageSegmenter("outputs/calculus_yolo11m_full/weights/best.pt",
                            "outputs/calculus_yolo11m_crop/weights/best.pt",
                            c1=0.05, c2=0.05, device=0, fusion="mask_nms",
                            nms_iou=NMS_IOU, score_mode=SCORE_MODE)
    man = pd.read_csv(DATA / "split_manifest.csv")
    rows = man[man.split == "val"]
    per = {c: {b: {"u": [], "i": []} for b in BUCKETS} for c in CENTER_MODES}
    n_neg = 0
    n_fp = {c: 0 for c in CENTER_MODES}
    for _, r in rows.iterrows():
        img = cv2.imread(r["image_path"])
        H, W = img.shape[:2]
        for centers in CENTER_MODES:
            recs = seg.candidates(r["image_path"], img, centers=centers)
            max_conf = max([cr["s2_conf"] for cr in recs]) if recs else 0.0
            if not bool(r["has_calculus"]):
                n_neg = n_neg or n_neg
                if max_conf >= 0.05:
                    n_fp[centers] += 1
                continue
            gt = cv2.imread(r["mask_original_path"], cv2.IMREAD_UNCHANGED) > 0
            n, comp, _, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8), 8)
            gts = []
            for i in range(1, n):
                if (comp == i).sum() < 5:
                    continue
                m = (comp == i).astype(np.uint8)
                px = int(m.sum())
                gts.append({"mask": m, "bucket": bucket_of(px / float(H * W))})
            mask = fuse_mask(recs, H, W, "mask_nms", NMS_IOU) > 0
            out_insts = [rec_bin(cr, H, W) for cr in greedy_nms(recs, "mask_nms", NMS_IOU, H, W)]
            for g in gts:
                gm = g["mask"]
                inter = int((gm & mask).sum())
                uni = int((gm | mask).sum())
                io = inter / uni if uni else 0.0
                dc = 2 * inter / (int(gm.sum()) + int(mask.sum())) if (int(gm.sum()) + int(mask.sum())) else 1.0
                per[centers][g["bucket"]]["u"].append((io, dc))
                best_io = 0.0
                for om in out_insts:
                    ii = int((gm & om).sum())
                    uu = int((gm | om).sum())
                    v = ii / uu if uu else 0.0
                    if v > best_io:
                        best_io = v
                per[centers][g["bucket"]]["i"].append(best_io)
        # count negatives only once
    n_neg_total = 0
    for _, r in rows.iterrows():
        if not bool(r["has_calculus"]):
            n_neg_total += 1
    rows_l = []
    for c in CENTER_MODES:
        for b in BUCKETS:
            us = per[c][b]["u"]
            if not us:
                continue
            io_u = np.array([v[0] for v in us])
            dc_u = np.array([v[1] for v in us])
            inst = np.array(per[c][b]["i"])
            rows_l.append({"center": c, "bucket": b, "n": len(us),
                           "dice": float(dc_u.mean()),
                           "mean_iou": float(io_u.mean()),
                           "r03": float((io_u >= 0.3).mean()),
                           "r05": float((io_u >= 0.5).mean()),
                           "inst_best_iou": float(inst.mean()),
                           "inst_r05": float((inst >= 0.5).mean())})
    df = pd.DataFrame(rows_l)
    df.to_csv(RUN / "exp6_crop_center_ablation.csv", index=False)
    pv = df.pivot_table(index="center", columns="bucket",
                        values=["dice", "mean_iou", "r03", "r05", "inst_r05"]).round(4)
    print("EXP6: semantic-union view (Dice | IoU | R@0.3 | R@0.5) and INSTANCE view (R@0.5)")
    print(pv.to_string())
    print("\nFP image rate: ", {c: round(n_fp[c] / max(n_neg_total, 1), 3) for c in CENTER_MODES},
          f"(neg={n_neg_total})")


if __name__ == "__main__":
    main()
