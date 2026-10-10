"""V1 vs V2 full c2 scan + PR curve & AP@IoU0.5 (real pipeline, VAL).

Stage1 frozen (c1=0.05). Per stage2 weights file:
  - c2 scan grid 0.05..0.70 evaluated through the real pipeline
    (bbox-center 160 crop -> stage2 -> independent instances -> restore ->
     mask-NMS @0.5, 1-to-1 greedy matching, TP-dice assert);
  - PR/AP: predictions ranked by s2_conf desc; process in order; each pred
    matches its best remaining GT (IoU>=0.5, strict 1-to-1); interpolated
    101-point AP.

Usage: python scripts/exp17_v1v2_pr_ap.py W1 [W2 ...]
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
C2_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
           0.45, 0.50, 0.60, 0.70)


def bucket_of(r: float) -> str:
    if r < BUCKET_EDGES[0]:
        return "small"
    if r < BUCKET_EDGES[1]:
        return "medium"
    return "large"


def collect(w):
    seg = TwoStageSegmenter("outputs/calculus_yolo11m_full/weights/best.pt",
                            w, c1=C1, c2=0.05, device=0, fusion="mask_nms",
                            nms_iou=NMS_IOU, score_mode="s2")
    man = pd.read_csv(DATA / "split_manifest.csv")
    rows = man[man.split == "val"]
    imgs = []  # {H,W,gts,recs}
    for _, r in rows.iterrows():
        img = cv2.imread(r["image_path"])
        H, W = img.shape[:2]
        boxes, confs = seg._stage1_full(r["image_path"])
        gt = cv2.imread(r["mask_original_path"], cv2.IMREAD_UNCHANGED)
        gts = []
        if gt is not None and (gt > 0).any():
            gtb = gt > 0
            n, comp, _, _ = cv2.connectedComponentsWithStats(gtb.astype(np.uint8), 8)
            for i in range(1, n):
                if (comp == i).sum() < 5:
                    continue
                m = (comp == i).astype(np.uint8)
                px = int(m.sum())
                gts.append({"mask": m, "bucket": bucket_of(px / float(H * W))})
        recs = []
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
                recs.append({"conf": inst["s2_conf"], "bin": mi, "x0": x0,
                             "y0": y0, "cw": cw, "ch": ch})
        imgs.append({"H": H, "W": W, "gts": gts, "recs": recs})
    return imgs


def per_c2(imgs, c):
    tp = {b: 0 for b in BUCKETS}
    gt_n = {b: 0 for b in BUCKETS}
    tp_dice, gt_dice, sem = [], [], []
    n_pred = n_unm = 0
    for im in imgs:
        H, W = im["H"], im["W"]
        sel = [{**x, "score": x["conf"]} for x in im["recs"] if x["conf"] >= c]
        keep = greedy_nms(sel, "mask_nms", NMS_IOU, H, W)
        preds = [rec_bin(k, H, W) for k in keep]
        n_pred += len(preds)
        gts = im["gts"]
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
        n_unm += len(preds) - len(matched)
        for pi, gi, io in matched:
            P = preds[pi]; G = gts[gi]
            inter = int((P & G["mask"]).sum())
            dc = 2 * inter / (int(P.sum()) + int(G["mask"].sum())) \
                if (int(P.sum()) + int(G["mask"].sum())) else 1.0
            assert dc >= 0.6667 - 1e-3
            tp[G["bucket"]] += 1
            tp_dice.append(dc)
        union = np.zeros((H, W), np.uint8)
        for P in preds:
            union |= P
        for gi, G in enumerate(gts):
            gt_n[G["bucket"]] += 1
            inter = int((G["mask"] & union).sum())
            uni = int((G["mask"] | union).sum())
            io = inter / uni if uni else 0.0
            dc = 2 * inter / (int(G["mask"].sum()) + int(union.sum())) \
                if (int(G["mask"].sum()) + int(union.sum())) else 1.0
            sem.append(dc)
            gt_dice.append(dc if gi in [g for _, g, _ in matched] else 0.0)
    tp_tot = sum(tp.values())
    gt_tot = sum(gt_n.values())
    rec = tp_tot / gt_tot if gt_tot else 0.0
    prec = tp_tot / n_pred if n_pred else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "c2": c, "precision": round(prec, 4), "recall": round(rec, 4),
        "f1": round(f1, 4),
        "tp_dice": round(float(np.mean(tp_dice)), 4),
        "gt_dice": round(float(np.mean(gt_dice)), 4),
        "sem_dice": round(float(np.mean(sem)), 4),
        "fp_img": round(n_unm / len(imgs), 3),
        "pred_img": round(n_pred / len(imgs), 3),
        "r_small": round(tp["small"] / gt_n["small"], 4),
        "r_medium": round(tp["medium"] / gt_n["medium"], 4),
        "r_large": round(tp["large"] / gt_n["large"], 4),
    }


def pr_ap(imgs):
    ranked = [(x["conf"], ii, pi)
              for ii, im in enumerate(imgs)
              for pi, x in enumerate(im["recs"])]
    ranked.sort(key=lambda t: -t[0])
    used_g = {ii: set() for ii in range(len(imgs))}
    tp_all = 0
    gt_tot = sum(len(im["gts"]) for im in imgs)
    pts = []
    for conf, ii, pi in ranked:
        im = imgs[ii]
        x = im["recs"][pi]
        buf = np.zeros((im["H"], im["W"]), np.uint8)
        buf[x["y0"]:x["y0"] + x["ch"], x["x0"]:x["x0"] + x["cw"]] = x["bin"]
        best = None
        for gi, G in enumerate(im["gts"]):
            if gi in used_g[ii]:
                continue
            inter = int((buf & G["mask"]).sum())
            uni = int((buf | G["mask"]).sum())
            io = inter / uni if uni else 0.0
            if io >= IOU_MATCH and (best is None or io > best[1]):
                best = (gi, io)
        matched = False
        if best is not None:
            used_g[ii].add(best[0])
            tp_all += 1
            matched = True
        pts.append((conf, tp_all, len(ranked[:len(pts) + 1])))
    curve = []
    for rq in np.linspace(0.0, 1.0, 101):
        best_p = 0.0
        for conf, t, npred in pts:
            r = t / gt_tot if gt_tot else 0.0
            if r >= rq:
                best_p = max(best_p, t / npred)
        curve.append({"recall": round(float(rq), 3), "precision": round(best_p, 4)})
    ap = float(np.mean([c["precision"] for c in curve]))
    return curve, ap


if __name__ == "__main__":
    for w in sys.argv[1:]:
        imgs = collect(w)
        rows = [per_c2(imgs, c) for c in C2_GRID]
        curve, ap = pr_ap(imgs)
        name = Path(w).stem
        df = pd.DataFrame(rows)
        df.to_csv(RUN / f"scan_{name}.csv", index=False)
        pd.DataFrame(curve).to_csv(RUN / f"pr_curve_{name}.csv", index=False)
        print(f"\n=== {name} | AP@IoU0.5 = {ap:.4f} ===")
        print(df.round(4).to_string(index=False))
