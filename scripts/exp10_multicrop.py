"""EXP10: multi-crop inference ablation (VAL only, no training).

Fixed: c1=0.05, NMS IoU=0.5, score=s2_conf, MIN_WINDOW=160, stage2 imgsz=640,
margin logic untouched (side = max(bw,bh)*1.8, min 160, clamped).
Schemes (offset window centers around the stage1-box center):
  A "base"  : (0,0)
  B "5crop" : (0,0),(+32,0),(-32,0),(0,+32),(0,-32)
  C "9crop" : dx,dy in {-32,0,+32} (9 crops)

All crops go independently into stage2; every stage2 instance restored
independently; ONE unified mask-NMS afterwards (no per-crop early union).
Per scheme: per-bucket Recall@IoU0.5, global Prec/F1 (one-to-one greedy
matching), matched Dice, semantic Dice (union vs GT), unmatched preds/image,
preds/image, wall-time/image.
Outputs: outputs/calculus_yolo11m_twostage/exp10_multicrop_ablation.csv
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

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.inference.fusion import greedy_nms, rec_bin
from src.inference.twostage import MIN_WINDOW, TARGET, TwoStageSegmenter, MARGIN
from src.utils.paths import PROCESSED_DIR

DATA = PROCESSED_DIR / "calculus_seg"
RUN = Path("outputs/calculus_yolo11m_twostage")
BUCKET_EDGES = (0.00552, 0.01565)
BUCKETS = ("small", "medium", "large")
NMS_IOU = 0.5
IOU_THR = 0.5
C1, C2 = 0.05, 0.05
OFF = 32
SCHEMES = {
    "A_base": [(0, 0)],
    "B_5crop": [(0, 0), (OFF, 0), (-OFF, 0), (0, OFF), (0, -OFF)],
    "C_9crop": [(dx, dy) for dx in (-OFF, 0, OFF) for dy in (-OFF, 0, OFF)],
}


def bucket_of(ratio: float) -> str:
    if ratio < BUCKET_EDGES[0]:
        return "small"
    if ratio < BUCKET_EDGES[1]:
        return "medium"
    return "large"


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
                            c1=C1, c2=C2, device=0, fusion="mask_nms",
                            nms_iou=NMS_IOU, score_mode="s2")
    man = pd.read_csv(DATA / "split_manifest.csv")
    rows = man[man.split == "val"]
    n_img = 0
    per = {s: {b: {"gt": 0, "tp": 0, "mdice": [], "miou": [], "sdice": [],
                   "sio": []} for b in BUCKETS} for s in SCHEMES}
    pred_img = {s: 0 for s in SCHEMES}
    unm_img = {s: 0 for s in SCHEMES}
    time_img = {s: 0.0 for s in SCHEMES}

    for _, r in rows.iterrows():
        img = cv2.imread(r["image_path"])
        H, W = img.shape[:2]
        t0 = time.time()
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
                gts.append({"mask": m, "bucket": bucket_of(px / float(H * W))})
        for sname, offs in SCHEMES.items():
            t1 = time.time()
            recs = []
            for b, sc1 in zip(boxes, confs):
                cx0, cy0 = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
                side = float(max(b[2] - b[0], b[3] - b[1]) * MARGIN)
                side = max(side, MIN_WINDOW)
                side = float(min(side, max(H, W)))
                for dx, dy in offs:
                    cx, cy = cx0 + dx, cy0 + dy
                    x0 = int(round(cx - side / 2)); y0 = int(round(cy - side / 2))
                    x0 = max(0, min(x0, W - int(side)))
                    y0 = max(0, min(y0, H - int(side)))
                    x1 = min(x0 + int(side), W); y1 = min(y0 + int(side), H)
                    if x1 <= x0 or y1 <= y0:
                        continue
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
            t2 = time.time()
            time_img[sname] += t2 - t1
            keep = greedy_nms(recs, "mask_nms", NMS_IOU, H, W)
            pred_img[sname] += len(keep)
            if not gts:
                unm_img[sname] += len(keep)
                continue
            preds = [rec_bin(c, H, W) for c in keep]
            union = np.zeros((H, W), np.uint8)
            for p in preds:
                union |= p
            matched, unm_pred = match_one_to_one(preds, gts)
            unm_img[sname] += len(unm_pred)
            matched_gi = {gi for _, gi, _ in matched}
            for gi, G in enumerate(gts):
                b = G["bucket"]
                per[sname][b]["gt"] += 1
                gm = G["mask"]
                inter = int((gm & union).sum())
                uni = int((gm | union).sum())
                io = inter / uni if uni else 0.0
                dc = 2 * inter / (int(gm.sum()) + int(union.sum())) \
                    if (int(gm.sum()) + int(union.sum())) else 1.0
                per[sname][b]["sdice"].append(dc)
                per[sname][b]["sio"].append(io)
                if gi in matched_gi:
                    pi = next(pp for pp, gg, _ in matched if gg == gi)
                    P = preds[pi]
                    ii = int((gm & P).sum())
                    uu = int((gm | P).sum())
                    v = ii / uu if uu else 0.0
                    dd = 2 * ii / (int(gm.sum()) + int(P.sum())) \
                        if (int(gm.sum()) + int(P.sum())) else 1.0
                    per[sname][b]["miou"].append(v)
                    per[sname][b]["mdice"].append(dd)
                    per[sname][b]["tp"] += 1
        n_img += 1

    rows_l = []
    for sname in SCHEMES:
        tp_g = sum(per[sname][b]["tp"] for b in BUCKETS)
        gt_g = sum(per[sname][b]["gt"] for b in BUCKETS)
        prec = tp_g / pred_img[sname] if pred_img[sname] else 0.0
        rec_g = tp_g / gt_g if gt_g else 0.0
        for b in BUCKETS:
            p = per[sname][b]
            if not p["gt"]:
                continue
            rows_l.append({
                "scheme": sname, "bucket": b, "n_gt": p["gt"], "tp": p["tp"],
                "recall": round(p["tp"] / p["gt"], 4),
                "semantic_dice": round(float(np.mean(p["sdice"])), 4),
                "semantic_iou": round(float(np.mean(p["sio"])), 4),
                "matched_dice": round(float(np.mean(p["mdice"])), 4),
                "matched_iou": round(float(np.mean(p["miou"])), 4),
            })
        rows_l.append({"scheme": sname, "bucket": "ALL", "n_gt": gt_g, "tp": tp_g,
                       "recall": round(rec_g, 4), "precision": round(prec, 4),
                       "f1": round(2 * prec * rec_g / (prec + rec_g) if (prec + rec_g) else 0.0, 4),
                       "unmatched_pred_img": round(unm_img[sname] / n_img, 3),
                       "pred_img": round(pred_img[sname] / n_img, 3),
                       "ms_img": round(1000 * time_img[sname] / n_img, 1)})
    df = pd.DataFrame(rows_l)
    df.to_csv(RUN / "exp10_multicrop_ablation.csv", index=False)
    pd.set_option("display.width", 250)
    print("EXP10 per-bucket (base / 5crop / 9crop):")
    print(df[df.bucket != "ALL"].pivot_table(
        index="scheme", columns="bucket",
        values=["recall", "semantic_dice", "matched_dice"]).round(4).to_string())
    print("\nEXP10 global:")
    print(df[df.bucket == "ALL"][["scheme", "recall", "precision", "f1",
                                  "unmatched_pred_img", "pred_img", "ms_img"]]
          .to_string(index=False))
    print(f"(images={n_img}, val excludes stage1-only time in ms_img)")


if __name__ == "__main__":
    main()
