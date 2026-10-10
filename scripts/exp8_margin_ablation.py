"""EXP8: crop MARGIN ablation (VAL only).

Fixed: c1=0.05, c2=0.05, bbox-center crop, MIN_WINDOW=160, stage2 imgsz=640,
score = s2_conf, mask-NMS IoU=0.5 (EXP7 frozen).
Variable: MARGIN {1.2, 1.4, 1.6, 1.8}.

Per size bucket: semantic Dice (union-mask view), instance Precision/Recall/F1
@IoU0.5 (one-to-one greedy matching), matched Dice, matched IoU,
unmatched predictions/image; also lesion_area / crop_area stats (median,
P25, P75) per bucket to check whether margin=1.8 dilutes small lesions.
Outputs: outputs/calculus_yolo11m_twostage/exp8_margin_ablation.csv

Own micro-dev: window size = max(bw, bh) * margin (>= MIN_WINDOW, <= max(H,W)),
centered on the stage1 bbox center; crop -> stage2 instances -> restore.
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

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.inference.fusion import greedy_nms, rec_bin
from src.utils.paths import PROCESSED_DIR
from src.inference.twostage import TwoStageSegmenter

DATA = PROCESSED_DIR / "calculus_seg"
RUN = Path("outputs/calculus_yolo11m_twostage")
BUCKET_EDGES = (0.00552, 0.01565)
BUCKETS = ("small", "medium", "large")
MARGINS = (1.2, 1.4, 1.6, 1.8)
NMS_IOU = 0.5
IOU_THR = 0.5
TARGET = 640
MIN_WINDOW = 160


def bucket_of(ratio: float) -> str:
    if ratio < BUCKET_EDGES[0]:
        return "small"
    if ratio < BUCKET_EDGES[1]:
        return "medium"
    return "large"


def window_for(box, margin: float, H: int, W: int) -> tuple[int, int, int, int]:
    x, y, x2, y2 = box
    cx, cy = (x + x2) / 2.0, (y + y2) / 2.0
    side = max(x2 - x, y2 - y) * float(margin)
    side = max(side, MIN_WINDOW)
    side = float(min(side, max(H, W)))
    x0 = int(round(cx - side / 2)); y0 = int(round(cy - side / 2))
    x0 = max(0, min(x0, W - int(side))); y0 = max(0, min(y0, H - int(side)))
    x1 = min(x0 + int(side), W); y1 = min(y0 + int(side), H)
    return x0, y0, x1, y1


def match_one_to_one(preds, gts, iou_thr=IOU_THR):
    pairs = []
    for pi, P in enumerate(preds):
        for gi, Gm in enumerate(gts):
            inter = int((P & Gm["mask"]).sum())
            union = int((P | Gm["mask"]).sum())
            io = inter / union if union else 0.0
            if io >= iou_thr:
                pairs.append((io, pi, gi))
    pairs.sort(reverse=True)
    used_p, used_g, matched = set(), set(), []
    for io, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi); used_g.add(gi)
        matched.append((pi, gi, io))
    return matched, [pi for pi in range(len(preds)) if pi not in used_p]


def main() -> None:
    seg = TwoStageSegmenter("outputs/calculus_yolo11m_full/weights/best.pt",
                            "outputs/calculus_yolo11m_crop/weights/best.pt",
                            c1=0.05, c2=0.05, device=0, fusion="mask_nms",
                            nms_iou=NMS_IOU, score_mode="s2")
    man = pd.read_csv(DATA / "split_manifest.csv")
    rows = man[man.split == "val"]

    n_img = 0
    per = {m: {b: {"gt": 0, "tp": 0, "s_dice": [], "s_iou": [],
                   "miou": [], "mdice": [], "ratio": []} for b in BUCKETS}
           for m in MARGINS}
    n_pred_img = {m: 0 for m in MARGINS}
    n_unm_img = {m: 0 for m in MARGINS}

    for _, r in rows.iterrows():
        img = cv2.imread(r["image_path"])
        H, W = img.shape[:2]
        boxes, confs = seg._stage1_full(r["image_path"])
        gt = cv2.imread(r["mask_original_path"], cv2.IMREAD_UNCHANGED)
        has = bool(r["has_calculus"])
        gts = []
        if has and gt is not None:
            gtb = gt > 0
            n, comp, _, _ = cv2.connectedComponentsWithStats(gtb.astype(np.uint8), 8)
            for i in range(1, n):
                if (comp == i).sum() < 5:
                    continue
                m = (comp == i).astype(np.uint8)
                px = int(m.sum())
                gts.append({"mask": m, "idx": i,
                            "box": cv2.boundingRect(m),
                            "bucket": bucket_of(px / float(H * W))})
        n_img += 1
        for mgn in MARGINS:
            recs = []
            win_of_box = []
            for b, sc1 in zip(boxes, confs):
                x0, y0, x1, y1 = window_for(b, mgn, H, W)
                ch, cw = y1 - y0, x1 - x0
                crop = cv2.resize(img[y0:y1, x0:x1], (TARGET, TARGET),
                                  interpolation=cv2.INTER_AREA)
                for inst in seg._stage2_instances(crop):
                    mm = cv2.resize(inst["bin640"], (cw, ch),
                                    interpolation=cv2.INTER_NEAREST)
                    recs.append({"x0": x0, "y0": y0, "cw": cw, "ch": ch,
                                 "bin": mm, "s1_conf": float(sc1),
                                 "s2_conf": inst["s2_conf"],
                                 "score": inst["s2_conf"]})
                win_of_box.append((x0, y0, x1, y1))
            kept = greedy_nms(recs, "mask_nms", NMS_IOU, H, W)
            n_pred_img[mgn] += len(kept)
            if not gts:
                n_unm_img[mgn] += len(kept)
                continue
            preds = [rec_bin(c, H, W) for c in kept]
            union = np.zeros((H, W), np.uint8)
            for p in preds:
                union |= p
            matched, unm_pred = match_one_to_one(preds, gts)
            n_unm_img[mgn] += len(unm_pred)
            matched_set = {gi for _, gi, _ in matched}
            for gi, G in enumerate(gts):
                b = G["bucket"]
                per[mgn][b]["gt"] += 1
                gm = G["mask"]
                inter = int((gm & union).sum())
                uni = int((gm | union).sum())
                io = inter / uni if uni else 0.0
                dc = 2 * inter / (int(gm.sum()) + int(union.sum())) \
                    if (int(gm.sum()) + int(union.sum())) else 1.0
                per[mgn][b]["s_dice"].append(dc)
                per[mgn][b]["s_iou"].append(io)
                w = None
                if gi in matched_set:
                    pi = next(pp for pp, gg, _ in matched if gg == gi)
                    cc = kept[pi]
                    w = (cc["x0"], cc["y0"], cc["x0"] + cc["cw"],
                         cc["y0"] + cc["ch"])
                    P = preds[pi]
                    ii = int((gm & P).sum())
                    uu = int((gm | P).sum())
                    v = ii / uu if uu else 0.0
                    dd = 2 * ii / (int(gm.sum()) + int(P.sum())) \
                        if (int(gm.sum()) + int(P.sum())) else 1.0
                    per[mgn][b]["miou"].append(v)
                    per[mgn][b]["mdice"].append(dd)
                    per[mgn][b]["tp"] += 1
                else:
                    gx, gy, gw_, gh_ = G["box"]
                    for ww in win_of_box:
                        if ww[0] <= gx + gw_ / 2 <= ww[2] and ww[1] <= gy + gh_ / 2 <= ww[3]:
                            w = ww
                            break
                if w is not None:
                    x0, y0, x1, y1 = w
                    area = float((x1 - x0) * (y1 - y0))
                    inside = int(gm[y0:y1, x0:x1].sum())
                    per[mgn][b]["ratio"].append(inside / area if area else 0.0)

    rows_l = []
    for mgn in MARGINS:
        for b in BUCKETS:
            p = per[mgn][b]
            gt_n = p["gt"]
            if not gt_n:
                continue
            rows_l.append({
                "margin": mgn, "bucket": b, "n_gt": gt_n, "tp": p["tp"],
                "recall": round(p["tp"] / gt_n, 4),
                "precision_approx": round(p["tp"] / gt_n, 4),
                "semantic_dice": round(float(np.mean(p["s_dice"])), 4),
                "semantic_iou": round(float(np.mean(p["s_iou"])), 4),
                "matched_dice": round(float(np.mean(p["mdice"])), 4),
                "matched_iou": round(float(np.mean(p["miou"])), 4),
                "ratio_med": round(float(np.median(p["ratio"])) if p["ratio"] else 0.0, 4),
                "ratio_p25": round(float(np.percentile(p["ratio"], 25)) if p["ratio"] else 0.0, 4),
                "ratio_p75": round(float(np.percentile(p["ratio"], 75)) if p["ratio"] else 0.0, 4),
            })
    df = pd.DataFrame(rows_l)
    df.to_csv(RUN / "exp8_margin_ablation.csv", index=False)
    print("EXP8 per bucket (n=gt, R=recall, sd=semantic dice, md=matched dice, mi=matched iou)")
    pv = df.pivot_table(index="margin", columns="bucket",
                        values=["recall", "semantic_dice", "matched_dice",
                                "matched_iou", "ratio_med"]).round(4)
    print(pv.to_string())
    print("\nunmatched preds/img / preds/img per margin:",
          {m: (round(n_unm_img[m] / n_img, 3), round(n_pred_img[m] / n_img, 3))
           for m in MARGINS})
    print(f"images={n_img}")


if __name__ == "__main__":
    main()
