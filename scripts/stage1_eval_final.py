"""Stage1 (candidate generator) evaluation on the development (val) split.

Input weights (YOLO model file). Frozen c1=0.05, iou=0.6, imgsz=640,
retina masks, max_det=100 (identical to baseline pipeline).
Per size bucket: GT n, proposal Recall (bbox overlap>0), proposal Recall
(mask-IoU>=0.1), Operational Crop Recall R_crop95 (any candidate's
bbox-center 160 crop with GT inclusion>=0.95), no-proposal rate.
Global: strict 1-to-1 proposal Precision/Recall/F1 (mask IoU>=0.5),
predictions/image, latency mean/median/P95, GPU peak MB.
Small stratified by GT bbox short-side <10/10-15/15-20/>=20px.
Outputs: outputs/calculus_cls/stage1_eval_<name>.csv
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
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from src.utils.paths import PROCESSED_DIR

DATA = PROCESSED_DIR / "calculus_seg"
OUT = Path("outputs/calculus_cls")
C1 = 0.05
SIDE = 160
IOU_MATCH = 0.5
BUCKET_EDGES = (0.00552, 0.01565)
BUCKETS = ("small", "medium", "large")


def bucket_of(r: float) -> str:
    if r < BUCKET_EDGES[0]:
        return "small"
    if r < BUCKET_EDGES[1]:
        return "medium"
    return "large"


def main() -> None:
    weights = sys.argv[1]
    label = Path(weights).stem.replace(".", "_")
    from ultralytics import YOLO
    model = YOLO(weights)
    man = pd.read_csv(DATA / "split_manifest.csv")
    rows = man[man.split == "val"]
    records = []
    lat = []
    n_prop = 0
    torch.cuda.reset_peak_memory_stats()
    for _, r in rows.iterrows():
        img = cv2.imread(r["image_path"])
        H, W = img.shape[:2]
        gtb = cv2.imread(r["mask_original_path"], cv2.IMREAD_UNCHANGED)
        gtb = (gtb > 0) if gtb is not None else np.zeros((H, W), np.uint8)
        n, comp, _, _ = cv2.connectedComponentsWithStats(gtb.astype(np.uint8), 8)
        gts = []
        for i in range(1, n):
            if (comp == i).sum() < 5:
                continue
            m = (comp == i).astype(np.uint8)
            gx, gy, gw, gh = cv2.boundingRect(m)
            px = int(m.sum())
            gts.append({"mask": m, "px": px, "box": (gx, gy, gw, gh),
                        "bucket": bucket_of(px / float(H * W)),
                        "short": min(gw, gh)})
        t0 = time.time()
        rp = model.predict(r["image_path"], conf=C1, iou=0.6, imgsz=640,
                           device=0, retina_masks=True, max_det=100,
                           verbose=False)[0]
        lat.append((time.time() - t0) * 1000)
        boxes = (rp.boxes.xyxy.cpu().numpy()
                 if rp.boxes is not None and len(rp.boxes)
                 else np.zeros((0, 4)))
        mk = (rp.masks.data.cpu().numpy()
              if rp.masks is not None else None)
        n_prop += len(boxes)
        masks = [((mk[j] > 0.5).astype(np.uint8)) if mk is not None else None
                 for j in range(len(boxes))]
        # crop windows (bbox-center 160, clamped)
        wins = []
        for b in boxes:
            cx, cy = (int(b[0]) + int(b[2])) / 2, (int(b[1]) + int(b[3])) / 2
            x0 = int(round(cx - SIDE / 2)); y0 = int(round(cy - SIDE / 2))
            x0 = max(0, min(x0, W - SIDE)); y0 = max(0, min(y0, H - SIDE))
            x1 = min(x0 + SIDE, W); y1 = min(y0 + SIDE, H)
            wins.append((x0, y0, x1, y1))
        for g in gts:
            gx, gy, gw, gh = g["box"]
            hit_bb = False
            mask_iou = 0.0
            crop95 = False
            for bi, bs in enumerate(boxes):
                inter = (max(0, min(gx + gw, int(bs[2])) - max(gx, int(bs[0]))) *
                         max(0, min(gy + gh, int(bs[3])) - max(gy, int(bs[1]))))
                if inter > 0:
                    hit_bb = True
                if masks[bi] is not None:
                    i2 = int((masks[bi] & g["mask"]).sum())
                    u = int((masks[bi] | g["mask"]).sum())
                    mask_iou = max(mask_iou, i2 / u if u else 0.0)
            for (wx0, wy0, wx1, wy1) in wins:
                inside = int(g["mask"][wy0:wy1, wx0:wx1].sum())
                if inside / float(g["px"]) >= 0.95:
                    crop95 = True
                    break
            records.append({"bucket": g["bucket"], "short": g["short"],
                            "hit_bb": int(hit_bb),
                            "mid_hit": int(mask_iou >= 0.1),
                            "crop95": int(crop95), "px": g["px"]})
    df = pd.DataFrame(records)
    mem = torch.cuda.max_memory_allocated() / 1e6
    lat = np.array(lat)
    n_img = len(rows)

    # strict 1-to-1 proposal matching (mask IoU>=0.5) for global P/R/F1:
    # recompute lazily per image (cheap enough: n_prop small)
    tp_m = 0
    for _, r in rows.iterrows():
        img = cv2.imread(r["image_path"])
        H, W = img.shape[:2]
        gtb = cv2.imread(r["mask_original_path"], cv2.IMREAD_UNCHANGED)
        gtb = (gtb > 0) if gtb is not None else np.zeros((H, W), np.uint8)
        n, comp, _, _ = cv2.connectedComponentsWithStats(gtb.astype(np.uint8), 8)
        gms = [(comp == i).astype(np.uint8) for i in range(1, n)
               if (comp == i).sum() >= 5]
        rp = model.predict(r["image_path"], conf=C1, iou=0.6, imgsz=640,
                           device=0, retina_masks=True, max_det=100,
                           verbose=False)[0]
        if rp.boxes is not None and len(rp.boxes) and rp.masks is not None:
            mk = rp.masks.data.cpu().numpy()
            used_g, used_p = set(), set()
            pairs = []
            for pi in range(len(mk)):
                P = (mk[pi] > 0.5).astype(np.uint8)
                for gi, G in enumerate(gms):
                    inter = int((P & G).sum())
                    uni = int((P | G).sum())
                    io = inter / uni if uni else 0.0
                    if io >= IOU_MATCH:
                        pairs.append((io, pi, gi))
            pairs.sort(reverse=True)
            for io, pi, gi in pairs:
                if pi in used_p or gi in used_g:
                    continue
                used_p.add(pi); used_g.add(gi)
                tp_m += 1

    rows_l = []
    for b in BUCKETS:
        s = df[df.bucket == b]
        rows_l.append({"bucket": b, "n_gt": len(s),
                       "recall_bbox": round(float(s.hit_bb.mean()), 4),
                       "recall_mIoU01": round(float(s.mid_hit.mean()), 4),
                       "R_crop95": round(float(s.crop95.mean()), 4),
                       "no_proposal": round(float(1 - s.hit_bb.mean()), 4)})
    for lo, hi, tag in ((0, 10, "small:<10"), (10, 15, "small:10-15"),
                        (15, 20, "small:15-20"), (20, 10**6, "small:>=20")):
        s = df[(df.bucket == "small") & (df.short >= lo) & (df.short < hi)]
        if len(s):
            rows_l.append({"bucket": tag, "n_gt": len(s),
                           "recall_bbox": round(float(s.hit_bb.mean()), 4),
                           "recall_mIoU01": round(float(s.mid_hit.mean()), 4),
                           "R_crop95": round(float(s.crop95.mean()), 4),
                           "no_proposal": round(float(1 - s.hit_bb.mean()), 4)})
    gt_tot = len(df)
    rows_l.append({"bucket": "GLOBAL", "n_gt": gt_tot,
                   "recall_bbox": round(float(df.hit_bb.mean()), 4),
                   "recall_mIoU01": round(float(df.mid_hit.mean()), 4),
                   "R_crop95": round(float(df.crop95.mean()), 4),
                   "no_proposal": round(float(1 - df.hit_bb.mean()), 4),
                   "proposal_precision": round(tp_m / max(n_prop, 1), 4),
                   "proposal_recall": round(tp_m / max(gt_tot, 1), 4),
                   "proposal_f1": round(2 * (tp_m / max(n_prop, 1)) *
                                        (tp_m / max(gt_tot, 1)) /
                                        max((tp_m / max(n_prop, 1)) +
                                            (tp_m / max(gt_tot, 1)), 1e-9), 4),
                   "predictions_image": round(n_prop / n_img, 3),
                   "lat_mean": round(float(np.mean(lat)), 1),
                   "lat_med": round(float(np.median(lat)), 1),
                   "lat_p95": round(float(np.percentile(lat, 95)), 1),
                   "gpu_peak_mb": round(mem, 0)})
    odm = pd.DataFrame(rows_l)
    odm.to_csv(OUT / f"stage1_eval_{label}.csv", index=False)
    print(odm.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
