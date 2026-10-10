"""EXP7: instance-level mask-NMS ablation with ONE-TO-ONE greedy matching (VAL).

Fixed: c1=0.05, c2=0.05, bbox-center crop, MARGIN=1.8, MIN_WINDOW=160,
stage2 imgsz=640, score = s2_conf.
Variable: mask-NMS IoU {0.3, 0.5, 0.7}.

Matching: predictions (raw stage2 instances kept after NMS, restored to
full-image coords) vs GT instances; all pairs sorted by IoU desc; greedy
1-to-1, matched if IoU >= 0.5. One prediction may NOT cover 2 GTs (and v.v.).

Per size bucket: n_gt, TP, FN, Recall@0.5, mean_matched_IoU, mean_matched_Dice.
Global: Precision@0.5, F1@0.5, unmatched predictions/image (all val images vs
predictions), predictions/image.
NOTE (no semantic-union metric used for selection; FP/image NOT used).
Outputs: outputs/calculus_yolo11m_twostage/exp7_inst_nms_ablation.csv
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

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.inference.fusion import greedy_nms, rec_bin
from src.inference.twostage import TwoStageSegmenter
from src.utils.paths import PROCESSED_DIR

DATA = PROCESSED_DIR / "calculus_seg"
RUN = Path("outputs/calculus_yolo11m_twostage")
BUCKET_EDGES = (0.00552, 0.01565)
BUCKETS = ("small", "medium", "large")
IOU_THR = 0.5
NMS_GRID = (0.3, 0.5, 0.7)


def bucket_of(ratio: float) -> str:
    if ratio < BUCKET_EDGES[0]:
        return "small"
    if ratio < BUCKET_EDGES[1]:
        return "medium"
    return "large"


def match_one_to_one(preds: list[np.ndarray], gts: list[dict], iou_thr: float):
    """preds: list of full-image uint8 masks; gts: list dict{mask,bucket}.
    Returns (matched list of (pi, gi, iou), unmatched_preds, unmatched_gts)."""
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
    unmatched_p = [pi for pi in range(len(preds)) if pi not in used_p]
    unmatched_g = [gi for gi in range(len(gts)) if gi not in used_g]
    return matched, unmatched_p, unmatched_g


def main() -> None:
    seg = TwoStageSegmenter("outputs/calculus_yolo11m_full/weights/best.pt",
                            "outputs/calculus_yolo11m_crop/weights/best.pt",
                            c1=0.05, c2=0.05, device=0, fusion="mask_nms",
                            nms_iou=0.5, score_mode="s2")
    man = pd.read_csv(DATA / "split_manifest.csv")
    rows = man[man.split == "val"]
    n_img = 0
    n_pred_img = {t: 0 for t in NMS_GRID}
    n_unm_pred_img = {t: 0 for t in NMS_GRID}
    per: dict = {t: {b: {"gt": 0, "tp": 0, "miou": [], "mdice": []} for b in BUCKETS}
                 for t in NMS_GRID}
    for _, r in rows.iterrows():
        img = cv2.imread(r["image_path"])
        H, W = img.shape[:2]
        recs = seg.candidates(r["image_path"], img)
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
        n_img += 1
        for t in NMS_GRID:
            kept = greedy_nms(recs, "mask_nms", t, H, W)
            n_pred_img[t] += len(kept)
            if not gts:
                n_unm_pred_img[t] += len(kept)
                continue
            preds = [rec_bin(c, H, W) for c in kept]
            matched, unm_p, _ = match_one_to_one(preds, gts, IOU_THR)
            n_unm_pred_img[t] += len(unm_p)
            for pi, gi, io in matched:
                P = preds[pi]; G = gts[gi]
                inter = int((P & G["mask"]).sum())
                dc = 2 * inter / (int(P.sum()) + int(G["mask"].sum())) \
                    if (int(P.sum()) + int(G["mask"].sum())) else 1.0
                b = gts[gi]["bucket"]
                per[t][b]["gt"] += 1
                per[t][b]["tp"] += 1
                per[t][b]["miou"].append(io)
                per[t][b]["mdice"].append(dc)
            for gi in range(len(gts)):
                if gi in [m[1] for m in matched]:
                    continue
                per[t][gts[gi]["bucket"]]["gt"] += 1

    rows_l = []
    for t in NMS_GRID:
        tp_tot, gt_tot = 0, 0
        for b in BUCKETS:
            p = per[t][b]
            tp_tot += p["tp"]; gt_tot += p["gt"]
            rec = p["tp"] / p["gt"] if p["gt"] else 0.0
            rows_l.append({"nms_iou": t, "bucket": b, "n_gt": p["gt"],
                           "tp": p["tp"], "fn": p["gt"] - p["tp"],
                           "recall": round(rec, 4),
                           "matched_iou": round(float(np.mean(p["miou"])) if p["miou"] else 0.0, 4),
                           "matched_dice": round(float(np.mean(p["mdice"])) if p["mdice"] else 0.0, 4)})
        # global precision / f1
        n_pred = n_pred_img[t]
        prec = tp_tot / n_pred if n_pred else 0.0
        f1 = 2 * prec * (tp_tot / gt_tot) / (prec + tp_tot / gt_tot) if (prec + tp_tot / gt_tot) else 0.0
        rows_l.append({"nms_iou": t, "bucket": "ALL", "n_gt": gt_tot, "tp": tp_tot,
                       "fn": gt_tot - tp_tot, "recall": round(tp_tot / gt_tot, 4),
                       "precision": round(prec, 4), "f1": round(f1, 4),
                       "unmatched_pred_img": round(n_unm_pred_img[t] / n_img, 4),
                       "pred_img": round(n_pred_img[t] / n_img, 4)})
    df = pd.DataFrame(rows_l)
    df.to_csv(RUN / "exp7_inst_nms_ablation.csv", index=False)
    for t in NMS_GRID:
        sub = df[(df.nms_iou == t) & (df.bucket != "ALL")]
        allr = df[(df.nms_iou == t) & (df.bucket == "ALL")].iloc[0]
        print(f"\n== NMS IoU={t} (score=s2_conf) ==")
        print(sub[["bucket", "n_gt", "tp", "recall", "matched_iou",
                   "matched_dice"]].to_string(index=False))
        print(f"   global Prec={allr['precision']:.4f} F1={allr['f1']:.4f} "
              f"recall={allr['recall']:.4f} unmatched_pred/img={allr['unmatched_pred_img']:.3f} "
              f"pred/img={allr['pred_img']:.3f} (images={n_img})")


if __name__ == "__main__":
    main()
