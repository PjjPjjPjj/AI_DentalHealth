"""EXP15: c2 scan on the REAL two-stage pipeline (VAL only, no training).

Real chain:
  image -> Stage1 (c1=0.05) -> predicted bbox -> bbox-centered 160x160 crop
  -> Stage2 (imgsz=640, independent instances) -> restore to full image
  -> mask-NMS (IoU=0.5, score=s2_conf) -> final instance list

Inference runs ONCE at c2=0.05 (instances keep confs); higher c2 replayed by
filtering instance conf >= c2 (equivalent to predict(conf=c2)).

Per c2: global Precision/Recall/F1@IOU0.5 (strict 1-to-1 greedy), per-bucket
Recall@0.5, TP-only matched Dice (assert >= 0.667 since IoU>=0.5),
GT-averaged Dice (unmatched GT = 0), semantic Dice (final union mask vs GT),
unmatched prediction instances/image, predictions/image (all 672 val images).
Output: outputs/calculus_yolo11m_twostage/exp15_pipeline_c2_scan.csv
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

from src.inference.twostage import TwoStageSegmenter, TARGET
from src.inference.fusion import greedy_nms, rec_bin
from src.utils.paths import PROCESSED_DIR

DATA = PROCESSED_DIR / "calculus_seg"
RUN = Path("outputs/calculus_yolo11m_twostage")
BUCKET_EDGES = (0.00552, 0.01565)
BUCKETS = ("small", "medium", "large")
C2_GRID = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
C1 = 0.05
SIDE = 160
NMS_IOU = 0.5
IOU_MATCH = 0.5


def bucket_of(ratio: float) -> str:
    if ratio < BUCKET_EDGES[0]:
        return "small"
    if ratio < BUCKET_EDGES[1]:
        return "medium"
    return "large"


def main() -> None:
    seg = TwoStageSegmenter("outputs/calculus_yolo11m_full/weights/best.pt",
                            "outputs/calculus_yolo11m_crop/weights/best.pt",
                            c1=C1, c2=0.05, device=0, fusion="mask_nms",
                            nms_iou=NMS_IOU, score_mode="s2")
    man = pd.read_csv(DATA / "split_manifest.csv")
    rows = man[man.split == "val"]

    n_img = 0
    per_c2 = {}
    for _, r in rows.iterrows():
        img = cv2.imread(r["image_path"])
        H, W = img.shape[:2]
        boxes, confs = seg._stage1_full(r["image_path"])
        gt = cv2.imread(r["mask_original_path"], cv2.IMREAD_UNCHANGED)
        has = bool(r["has_calculus"])
        gts = []
        if has and gt is not None and (gt > 0).any():
            gtb = (gt > 0)
            n, comp, _, _ = cv2.connectedComponentsWithStats(gtb.astype(np.uint8), 8)
            for i in range(1, n):
                if (comp == i).sum() < 5:
                    continue
                m = (comp == i).astype(np.uint8)
                px = int(m.sum())
                gts.append({"mask": m, "bucket": bucket_of(px / float(H * W))})
        crops_recs = []  # per stage1 box: list of (x0,y0,cw,ch, bin, s2_conf)
        for b, sc1 in zip(boxes, confs):
            cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
            x0 = int(round(cx - SIDE / 2)); y0 = int(round(cy - SIDE / 2))
            x0 = max(0, min(x0, W - SIDE)); y0 = max(0, min(y0, H - SIDE))
            x1, y1 = min(x0 + SIDE, W), min(y0 + SIDE, H)
            crop = cv2.resize(img[y0:y1, x0:x1], (TARGET, TARGET),
                              interpolation=cv2.INTER_AREA)
            insts = seg._stage2_instances(crop)
            ch, cw = y1 - y0, x1 - x0
            for inst in insts:
                mi = cv2.resize(inst["bin640"], (cw, ch),
                                interpolation=cv2.INTER_NEAREST)
                crops_recs.append({"x0": x0, "y0": y0, "cw": cw, "ch": ch,
                                   "bin": mi, "s2_conf": inst["s2_conf"]})
        for c in C2_GRID:
            recs = [cr for cr in crops_recs if cr["s2_conf"] >= c]
            keep = greedy_nms(recs, "mask_nms", NMS_IOU, H, W)
            preds = [rec_bin(k, H, W) for k in keep]
            slot = per_c2.setdefault(c, {"preds": 0, "unm": 0, "gts": [],
                                         "tp_bucket": {b: 0 for b in BUCKETS},
                                         "gt_bucket": {b: 0 for b in BUCKETS},
                                         "tp_dice": [], "gt_dice": [],
                                         "s_dice": [], "tp_io": []})
            slot["preds"] += len(preds)
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
            slot["unm"] += len(preds) - len(matched)
            matched_gi = set()
            for pi, gi, io in matched:
                matched_gi.add(gi)
                G = gts[gi]
                P = preds[pi]
                inter = int((P & G["mask"]).sum())
                dc = 2 * inter / (int(P.sum()) + int(G["mask"].sum())) \
                    if (int(P.sum()) + int(G["mask"].sum())) else 1.0
                assert dc >= 0.6667 - 1e-3, f"TP dice below 0.667: {dc}"
                slot["tp_bucket"][G["bucket"]] += 1
                slot["tp_dice"].append(dc)
                slot["tp_io"].append(io)
            union = np.zeros((H, W), np.uint8)
            for P in preds:
                union |= P
            for gi, G in enumerate(gts):
                slot["gt_bucket"][G["bucket"]] += 1
                gm = G["mask"]
                inter = int((gm & union).sum())
                uni = int((gm | union).sum())
                io = inter / uni if uni else 0.0
                dc = 2 * inter / (int(gm.sum()) + int(union.sum())) \
                    if (int(gm.sum()) + int(union.sum())) else 1.0
                slot["gt_dice"].append(dc)
                slot["s_dice"].append(dc)
                if gi not in matched_gi:
                    slot["gt_dice"][-1] = 0.0  # unmatched GT => 0 (matches GT-averaged def)
        n_img += 1

    rows = []
    for c in C2_GRID:
        s = per_c2[c]
        tp = sum(s["tp_bucket"].values())
        gt_ = sum(s["gt_bucket"].values())
        rec = tp / gt_ if gt_ else 0.0
        prec = tp / s["preds"] if s["preds"] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        dct = float(np.mean(s["tp_dice"])) if s["tp_dice"] else 0.0
        rows.append({"c2": c, "precision": round(prec, 4), "recall": round(rec, 4),
                     "f1": round(f1, 4),
                     "tp_only_dice": round(dct, 4),
                     "gt_avg_dice": round(float(np.mean(s["gt_dice"])), 4),
                     "semantic_dice": round(float(np.mean(s["s_dice"])), 4),
                     "unmatched_img": round(s["unm"] / n_img, 3),
                     "pred_img": round(s["preds"] / n_img, 3)})
        for b in BUCKETS:
            rows.append({"c2": c, "bucket": b, "n_gt": s["gt_bucket"][b],
                         "tp": s["tp_bucket"][b],
                         "recall_b": round(s["tp_bucket"][b] / s["gt_bucket"][b]
                                           if s["gt_bucket"][b] else 0.0, 4)})
    df = pd.DataFrame(rows)
    df.to_csv(RUN / "exp15_pipeline_c2_scan.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"EXP15: real-pipeline c2 scan (images={n_img}, strict 1-to-1, TP dice assert OK)")
    print(df[df.bucket.isna()][["c2", "precision", "recall", "f1",
                               "tp_only_dice", "gt_avg_dice", "semantic_dice",
                               "unmatched_img", "pred_img"]].to_string(index=False))
    print("\nper-bucket Recall@IoU0.5:")
    print(df[~df.bucket.isna()].pivot_table(index="c2", columns="bucket",
                                            values="recall_b").round(4).to_string())


if __name__ == "__main__":
    main()
