"""EXP13: Stage2 confidence c2 ablation (VAL only, no training).

Fixed: c1=0.05, crop side=160 (GT bbox center), stage2 640,
mask-NMS IoU=0.5, score = s2_conf.
Variable: c2 {0.05,0.10,0.15,0.20,0.30,0.40,0.50}.

Inference runs ONCE at c2=0.05 with confs; each higher c2 is replayed by
filtering instance conf >= c2 (equivalent to predict(conf=c2)).
Per GT one crop; intra-crop greedy NMS; strict 1-to-1 matching (a kept
instance may match only its own crop's GT).

Per c2: global Precision/Recall/F1@IOU0.5, per-bucket Recall@0.5 & matched
Dice & semantic Dice & mean IoU, unmatched predictions/image,
predictions/image (pos images, GT-centered protocol; images with GT).
Output: outputs/calculus_yolo11m_twostage/exp13_c2_ablation.csv
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

from src.inference.fusion import greedy_nms
from src.inference.twostage import TARGET, TwoStageSegmenter
from src.utils.paths import PROCESSED_DIR

DATA = PROCESSED_DIR / "calculus_seg"
RUN = Path("outputs/calculus_yolo11m_twostage")
BUCKET_EDGES = (0.00552, 0.01565)
BUCKETS = ("small", "medium", "large")
SIDE = 160
C2_GRID = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
NMS_IOU = 0.5


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
                            nms_iou=NMS_IOU, score_mode="s2")
    man = pd.read_csv(DATA / "split_manifest.csv")
    rows = man[(man.split == "val") & man.has_calculus]

    per = {c: {b: {"gt": 0, "tp": 0, "mdice": [], "sdice": [], "sio": []}
               for b in BUCKETS} for c in C2_GRID}
    pred_tot = {c: 0 for c in C2_GRID}
    unm_tot = {c: 0 for c in C2_GRID}
    for _, r in rows.iterrows():
        img = cv2.imread(r["image_path"])
        H, W = img.shape[:2]
        gt = cv2.imread(r["mask_original_path"], cv2.IMREAD_UNCHANGED) > 0
        if not gt.any():
            continue
        n, comp, _, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8), 8)
        gts = []
        for i in range(1, n):
            if (comp == i).sum() < 5:
                continue
            m = (comp == i).astype(np.uint8)
            gx, gy, gw, gh = cv2.boundingRect(m)
            px = int(m.sum())
            gts.append({"mask": m, "bucket": bucket_of(px / float(H * W))})
        for g in gts:
            mm = g["mask"]
            gx, gy, gw, gh = cv2.boundingRect(mm)
            cx, cy = gx + gw / 2.0, gy + gh / 2.0
            x0 = int(round(cx - SIDE / 2)); y0 = int(round(cy - SIDE / 2))
            x0 = max(0, min(x0, W - SIDE)); y0 = max(0, min(y0, H - SIDE))
            crop = cv2.resize(img[y0:y0 + SIDE, x0:x0 + SIDE], (TARGET, TARGET),
                              interpolation=cv2.INTER_AREA)
            insts = seg._stage2_instances(crop)
            tcs = cv2.resize(mm[y0:y0 + SIDE, x0:x0 + SIDE].astype(np.uint8),
                             (TARGET, TARGET), interpolation=cv2.INTER_AREA)
            tc = (tcs > 0.5).astype(np.uint8)
            for c in C2_GRID:
                sel = [{"x0": 0, "y0": 0, "cw": TARGET, "ch": TARGET,
                        "bin": i["bin640"], "s2_conf": i["s2_conf"],
                        "score": i["s2_conf"]} for i in insts if i["s2_conf"] >= c]
                keep = greedy_nms(sel, "mask_nms", NMS_IOU, 640, 640)
                pred_tot[c] += len(keep)
                st = per[c][g["bucket"]]
                st["gt"] += 1
                union = np.zeros((TARGET, TARGET), np.uint8)
                best_io, best_mi = 0.0, None
                for k in keep:
                    mi = k["bin"]
                    union = np.maximum(union, mi)
                    ii = int((tc & mi).sum())
                    uu = int((tc | mi).sum())
                    v = ii / uu if uu else 0.0
                    if v > best_io:
                        best_io = v
                        best_mi = mi
                if best_io >= 0.5:
                    st["tp"] += 1
                    ii = int((tc & best_mi).sum())
                    uu = int((tc | best_mi).sum())
                    st["mdice"].append(2 * ii / (int(tc.sum()) + int(best_mi.sum())) if (int(tc.sum()) + int(best_mi.sum())) else 1.0)
                    unm_tot[c] += max(0, len(keep) - 1)
                else:
                    unm_tot[c] += len(keep)
                    st["mdice"].append(0.0)
                inter = int((tc & union).sum())
                uni = int((tc | union).sum())
                io = inter / uni if uni else 0.0
                dc = 2 * inter / (int(tc.sum()) + int(union.sum())) if (int(tc.sum()) + int(union.sum())) else 1.0
                st["sio"].append(io)
                st["sdice"].append(dc)

    rows = []
    for c in C2_GRID:
        tp = sum(per[c][b]["tp"] for b in BUCKETS)
        gt_ = sum(per[c][b]["gt"] for b in BUCKETS)
        rec = tp / gt_ if gt_ else 0.0
        n_pred = pred_tot[c]
        prec = tp / n_pred if n_pred else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({
            "c2": c, "gt": gt_, "tp": tp,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4),
            "unmatched_pred_img": round(unm_tot[c] / 142.0, 3),
            "pred_img": round(n_pred / 142.0, 3),
        })
        for b in BUCKETS:
            st = per[c][b]
            rows.append({"c2": c, "bucket": b, "gt_l": st["gt"], "tp_l": st["tp"],
                         "recall_l": round(st["tp"] / st["gt"] if st["gt"] else 0.0, 4),
                         "matched_dice_l": round(float(np.mean(st["mdice"])), 4),
                         "semantic_dice_l": round(float(np.mean(st["sdice"])), 4),
                         "mean_iou_l": round(float(np.mean(st["sio"])), 4)})
    df = pd.DataFrame(rows)
    df.to_csv(RUN / "exp13_c2_ablation.csv", index=False)
    g = df[df.bucket.isna()]
    print("EXP13: global (1-to-1 @IoU0.5) per c2")
    print(g[["c2", "precision", "recall", "f1", "unmatched_pred_img",
             "pred_img"]].to_string(index=False))
    print("\nEXP13 per-bucket (R=Recall@0.5, MD=matched Dice, SD=semantic Dice)")
    bb = df[~df.bucket.isna()]
    print(bb.pivot_table(index="c2", columns="bucket",
                         values=["recall_l", "matched_dice_l",
                                 "semantic_dice_l"]).round(4).to_string())


if __name__ == "__main__":
    main()
