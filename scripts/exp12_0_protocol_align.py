"""EXP12-0: evaluation-protocol alignment (VAL only, no training).

Goal: reproduce EXP3's GT-centered crop numbers (small .513 / med .645 /
large .710) under the CURRENT unified protocol:
  - stage2 raw instances kept (no early union inside crop)
  - one-to-one greedy instance matching (IoU>=0.5, one pred <=> one GT)
  - crop center = GT bbox center, side = max(max(bw,bh)*1.8, 160), clamped
  - stage2 640, c1/c2/NMS untouched (NMS not applied here: per-crop only)

Two views are reported so the difference vs EXP3 becomes explicit:
  VIEW A (current unified protocol, per GT instance):
    semantic Dice = Dice(union of stage2 instances, GT instance crop),
    mean IoU, Recall@IoU0.5 (best single instance matched), matched Dice
  VIEW B (EXP3 original semantic view = union mask vs FULL GT annotation
    in the crop, i.e. neighboring GT labels also count):
    dice_full = Dice(union mask, full annotation crop)
Outputs: outputs/calculus_yolo11m_twostage/exp12_0_protocol_align.csv
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

from src.inference.twostage import TARGET, TwoStageSegmenter, MARGIN, MIN_WINDOW
from src.utils.paths import PROCESSED_DIR

DATA = PROCESSED_DIR / "calculus_seg"
RUN = Path("outputs/calculus_yolo11m_twostage")
BUCKET_EDGES = (0.00552, 0.01565)
BUCKETS = ("small", "medium", "large")
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

    st = {b: {"n": 0, "sdice": [], "sio": [], "hit": [], "mdice": [],
              "dice_full": []} for b in BUCKETS}
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
            st[b]["n"] += 1
            side = float(max(gw, gh) * MARGIN)
            side = max(side, MIN_WINDOW)
            side = float(min(side, max(H, W)))
            cx, cy = gx + gw / 2.0, gy + gh / 2.0
            x0 = int(round(cx - side / 2)); y0 = int(round(cy - side / 2))
            x0 = max(0, min(x0, W - int(side))); y0 = max(0, min(y0, H - int(side)))
            x1 = min(x0 + int(side), W); y1 = min(y0 + int(side), H)
            ch, cw = y1 - y0, x1 - x0
            crop = cv2.resize(img[y0:y1, x0:x1], (TARGET, TARGET),
                              interpolation=cv2.INTER_AREA)
            insts = seg._stage2_instances(crop)
            # VIEW B: full annotation crop (EXP3 original)
            tcf = cv2.resize(gt[y0:y1, x0:x1].astype(np.uint8), (TARGET, TARGET),
                             interpolation=cv2.INTER_AREA)
            tcf = (tcf > 0.5).astype(np.uint8)
            union = np.zeros((TARGET, TARGET), np.uint8)
            best_io, best_mi = 0.0, None
            for inst in insts:
                mi = inst["bin640"]
                union = np.maximum(union, mi)
                ii = int((tcf & mi).sum())
                uu = int((tcf | mi).sum())
                v = ii / uu if uu else 0.0
                if v > best_io:
                    best_io = v
                    best_mi = mi
            interb = int((tcf & union).sum())
            unib = int((tcf | union).sum())
            st[b]["dice_full"].append(
                2 * interb / (int(tcf.sum()) + int(union.sum()))
                if (int(tcf.sum()) + int(union.sum())) else 1.0)
            # VIEW A: per-GT-instance basis
            tc = cv2.resize(m[y0:y1, x0:x1].astype(np.uint8), (TARGET, TARGET),
                            interpolation=cv2.INTER_AREA)
            tc = (tc > 0.5).astype(np.uint8)
            inter = int((tc & union).sum())
            uni = int((tc | union).sum())
            io = inter / uni if uni else 0.0
            dc = 2 * inter / (int(tc.sum()) + int(union.sum())) \
                if (int(tc.sum()) + int(union.sum())) else 1.0
            st[b]["sdice"].append(dc)
            st[b]["sio"].append(io)
            st[b]["hit"].append(int(best_io >= 0.5))
            md = 0.0
            if best_mi is not None and best_io >= 0.5:
                interi = int((tc & best_mi).sum())
                unii = int((tc | best_mi).sum())
                md = 2 * interi / (int(tc.sum()) + int(best_mi.sum())) \
                    if (int(tc.sum()) + int(best_mi.sum())) else 1.0
            st[b]["mdice"].append(md)

    rows = []
    for b in BUCKETS:
        s = st[b]
        rows.append({"bucket": b, "n": s["n"],
                     "viewA_semantic_dice": round(float(np.mean(s["sdice"])), 4),
                     "viewA_mean_iou": round(float(np.mean(s["sio"])), 4),
                     "viewA_recall05": round(float(np.mean(s["hit"])), 4),
                     "viewA_matched_dice": round(float(np.mean(s["mdice"])), 4),
                     "viewB_dice_full_gt(EXP3)":
                         round(float(np.mean(s["dice_full"])), 4)})
    df = pd.DataFrame(rows)
    df.to_csv(RUN / "exp12_0_protocol_align.csv", index=False)
    print("EXP12-0 protocol alignment (VAL, GT-centered crop, side=max(1.8*size,160))")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
