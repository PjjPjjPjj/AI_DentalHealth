"""End-to-end VAL evaluation of Stage2 V2 checkpoints (real pipeline).

Real chain, frozen: Stage1 (c1=0.05, calculus_yolo11m_full best), crop
bbox-center 160x160, NMS IoU=0.5, score=s2_conf; Stage2 model = given
checkpoint (imgsz=640). Inference runs once at base c2=0.05; grids replayed.

Per stage2 ckpt + c2 grid -> Precision/Recall/F1, per-bucket Recall@0.5,
TP-only Dice (assert >=0.667), GT-avg Dice, semantic Dice, FP/img.
Usage: PYTHONPATH=. python scripts/exp16_v2_e2e_eval.py CKPT1 [CKPT2 ...]
"""
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
from src.inference.twostage import TARGET, TwoStageSegmenter
from src.utils.paths import PROCESSED_DIR

DATA = PROCESSED_DIR / "calculus_seg"
RUN = Path("outputs/calculus_yolo11m_twostage")
C1 = 0.05
SIDE = 160
NMS_IOU = 0.5
IOU_MATCH = 0.5
BUCKET_EDGES = (0.00552, 0.01565)
BUCKETS = ("small", "medium", "large")
C2_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)


def bucket_of(r: float) -> str:
    if r < BUCKET_EDGES[0]:
        return "small"
    if r < BUCKET_EDGES[1]:
        return "medium"
    return "large"


def evaluate_checkpoint(s2_w: str):
    seg = TwoStageSegmenter("outputs/calculus_yolo11m_full/weights/best.pt",
                            s2_w, c1=C1, c2=0.05, device=0, fusion="mask_nms",
                            nms_iou=NMS_IOU, score_mode="s2")
    man = pd.read_csv(DATA / "split_manifest.csv")
    rows = man[man.split == "val"]
    n_img = 0
    per = {}
    for _, r in rows.iterrows():
        img = cv2.imread(r["image_path"])
        H, W = img.shape[:2]
        boxes, confs = seg._stage1_full(r["image_path"])
        gt = cv2.imread(r["mask_original_path"], cv2.IMREAD_UNCHANGED)
        has = bool(r["has_calculus"])
        gts = []
        if has and gt is not None and (gt > 0).any():
            gtb = gt > 0
            n, comp, _, _ = cv2.connectedComponentsWithStats(gtb.astype(np.uint8), 8)
            for i in range(1, n):
                if (comp == i).sum() < 5:
                    continue
                m = (comp == i).astype(np.uint8)
                px = int(m.sum())
                gts.append({"mask": m, "bucket": bucket_of(px / float(H * W))})
        crops_recs = []
        for b, sc1 in zip(boxes, confs):
            cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
            x0 = int(round(cx - SIDE / 2)); y0 = int(round(cy - SIDE / 2))
            x0 = max(0, min(x0, W - SIDE)); y0 = max(0, min(y0, H - SIDE))
            x1, y1 = min(x0 + SIDE, W), min(y0 + SIDE, H)
            ch, cw = y1 - y0, x1 - x0
            crop = cv2.resize(img[y0:y1, x0:x1], (TARGET, TARGET),
                              interpolation=cv2.INTER_AREA)
            for inst in seg._stage2_instances(crop):
                mi = cv2.resize(inst["bin640"], (cw, ch),
                                interpolation=cv2.INTER_NEAREST)
                crops_recs.append({"x0": x0, "y0": y0, "cw": cw, "ch": ch,
                                   "bin": mi, "s2_conf": inst["s2_conf"]})
        for c in C2_GRID:
            recs = [cr for cr in crops_recs if cr["s2_conf"] >= c]
            keep = greedy_nms(recs, "mask_nms", NMS_IOU, H, W)
            preds = [rec_bin(k, H, W) for k in keep]
            s = per.setdefault(c, {"preds": 0, "unm": 0,
                                   "tp": {b: 0 for b in BUCKETS},
                                   "gt": {b: 0 for b in BUCKETS},
                                   "tp_dice": [], "gt_dice": [],
                                   "sem": []})
            s["preds"] += len(preds)
            pairs = []
            for pi, P in enumerate(preds):
                for gi, G in enumerate(gts):
                    inter = int((P & G["mask"]).sum())
                    uni = int((P | G["mask"]).sum())
                    io = inter / uni if uni else 0.0
                    if io >= IOU_MATCH:
                        pairs.append((io, pi, gi))
            pairs.sort(reverse=True)
            used_p, used_g, matched = set(), set(), []
            for io, pi, gi in pairs:
                if pi in used_p or gi in used_g:
                    continue
                used_p.add(pi); used_g.add(gi)
                matched.append((pi, gi, io))
            s["unm"] += len(preds) - len(matched)
            matched_gi = set()
            for pi, gi, io in matched:
                matched_gi.add(gi)
                P = preds[pi]; G = gts[gi]
                inter = int((P & G["mask"]).sum())
                dc = 2 * inter / (int(P.sum()) + int(G["mask"].sum())) \
                    if (int(P.sum()) + int(G["mask"].sum())) else 1.0
                assert dc >= 0.6667 - 1e-3
                s["tp"][G["bucket"]] += 1
                s["tp_dice"].append(dc)
            union = np.zeros((H, W), np.uint8)
            for P in preds:
                union |= P
            for gi, G in enumerate(gts):
                s["gt"][G["bucket"]] += 1
                gm = G["mask"]
                inter = int((gm & union).sum())
                uni = int((gm | union).sum())
                io = inter / uni if uni else 0.0
                dc = 2 * inter / (int(gm.sum()) + int(union.sum())) \
                    if (int(gm.sum()) + int(union.sum())) else 1.0
                s["sem"].append(dc)
                s["gt_dice"].append(dc if gi in matched_gi else 0.0)
        n_img += 1
    out = []
    for c in C2_GRID:
        s = per[c]
        tp = sum(s["tp"].values())
        gt_ = sum(s["gt"].values())
        rec = tp / gt_ if gt_ else 0.0
        prec = tp / s["preds"] if s["preds"] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out.append({"c2": c, "precision": round(prec, 4), "recall": round(rec, 4),
                    "f1": round(f1, 4), "tp_dice": round(float(np.mean(s["tp_dice"])), 4),
                    "gt_dice": round(float(np.mean(s["gt_dice"])), 4),
                    "sem_dice": round(float(np.mean(s["sem"])), 4),
                    "fp_img": round(s["unm"] / n_img, 3),
                    "pred_img": round(s["preds"] / n_img, 3),
                    "r_small": round(s["tp"]["small"] / s["gt"]["small"], 4),
                    "r_medium": round(s["tp"]["medium"] / s["gt"]["medium"], 4),
                    "r_large": round(s["tp"]["large"] / s["gt"]["large"], 4)})
    return pd.DataFrame(out)


if __name__ == "__main__":
    for w in sys.argv[1:]:
        name = Path(w).stem
        df = evaluate_checkpoint(w)
        df.to_csv(RUN / f"v2_e2e_{name}.csv", index=False)
        print(f"\n=== Stage2 ckpt: {w} ===")
        print(df[["c2", "precision", "recall", "f1", "tp_dice", "gt_dice",
                  "sem_dice", "fp_img", "pred_img", "r_small", "r_medium",
                  "r_large"]].round(4).to_string(index=False))
