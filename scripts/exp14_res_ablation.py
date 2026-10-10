"""EXP14: stage2 input resolution ablation (VAL only, no training).

Fixed (EXP13-frozen): c1=0.05, c2=0.30, crop side=160 (GT bbox center),
bbox center, mask-NMS IoU=0.5, score=s2_conf.
Variable: stage2 inference imgsz {640,768,896,1024} (crop resized to that
size; different resolutions change effective lesion pixels in the input).

Per bucket: Recall@IoU0.5, semantic Dice, matched Dice, mean IoU.
Global: Precision/F1@0.5 (strict 1-to-1). Inference time/image and GPU peak
memory. Plus median GT short-side (in stage2 input space) per bucket.
Output: outputs/calculus_yolo11m_twostage/exp14_res_ablation.csv
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

from src.inference.twostage import TwoStageSegmenter
from src.utils.paths import PROCESSED_DIR
import torch

DATA = PROCESSED_DIR / "calculus_seg"
RUN = Path("outputs/calculus_yolo11m_twostage")
BUCKET_EDGES = (0.00552, 0.01565)
BUCKETS = ("small", "medium", "large")
SIDE = 160
C2 = 0.30
NMS_IOU = 0.5
RES = (640, 768, 896, 1024)


def bucket_of(ratio: float) -> str:
    if ratio < BUCKET_EDGES[0]:
        return "small"
    if ratio < BUCKET_EDGES[1]:
        return "medium"
    return "large"


def main() -> None:
    seg = TwoStageSegmenter("outputs/calculus_yolo11m_full/weights/best.pt",
                            "outputs/calculus_yolo11m_crop/weights/best.pt",
                            c1=0.05, c2=C2, device=0, fusion="mask_nms",
                            nms_iou=NMS_IOU, score_mode="s2")
    man = pd.read_csv(DATA / "split_manifest.csv")
    rows = man[(man.split == "val") & man.has_calculus]

    per = {r_: {b: {"gt": 0, "tp": 0, "mdice": [], "sdice": [], "sio": [],
                    "short": []} for b in BUCKETS} for r_ in RES}
    time_img = {r_: 0.0 for r_ in RES}
    mem = {r_: 0.0 for r_ in RES}
    n_img = 0
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
            gts.append({"mask": m, "bucket": bucket_of(px / float(H * W)),
                        "bw": bw if (bw := gw) else gw, "bh": gh,
                        "short": min(gw, gh)})
        for r_ in RES:
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            for g in gts:
                mm = g["mask"]
                gx, gy, gw, gh = cv2.boundingRect(mm)
                cx, cy = gx + gw / 2.0, gy + gh / 2.0
                x0 = int(round(cx - SIDE / 2)); y0 = int(round(cy - SIDE / 2))
                x0 = max(0, min(x0, W - SIDE)); y0 = max(0, min(y0, H - SIDE))
                crop = cv2.resize(img[y0:y0 + SIDE, x0:x0 + SIDE], (r_, r_),
                                  interpolation=cv2.INTER_AREA)
                res = seg.s2.predict(crop, conf=C2, iou=0.6, imgsz=r_, device=0,
                                     retina_masks=True, max_det=100,
                                     verbose=False)[0]
                insts = []
                if res.boxes is not None and len(res.boxes) and res.masks is not None:
                    mk = res.masks.data.cpu().numpy()
                    for k in range(mk.shape[0]):
                        mi = (mk[k] > 0.5).astype(np.uint8)
                        if mi.sum():
                            insts.append({"bin": mi, "s2_conf": float(res.boxes.conf[k])})
                tcs = cv2.resize(mm[y0:y0 + SIDE, x0:x0 + SIDE].astype(np.uint8),
                                 (r_, r_), interpolation=cv2.INTER_AREA)
                tc = (tcs > 0.5).astype(np.uint8)
                # simple greedy NMS intra-crop
                keep = []
                sel = sorted(insts, key=lambda i: i["s2_conf"], reverse=True)
                for i in sel:
                    overlaps = False
                    for a in keep:
                        ii = int((i["bin"] & a["bin"]).sum())
                        uu = int((i["bin"] | a["bin"]).sum())
                        if uu and ii / uu >= NMS_IOU:
                            overlaps = True
                            break
                    if not overlaps:
                        keep.append(i)
                union = np.zeros((r_, r_), np.uint8)
                best_io, best_mi = 0.0, None
                for k in keep:
                    union = np.maximum(union, k["bin"])
                    ii = int((tc & k["bin"]).sum())
                    uu = int((tc | k["bin"]).sum())
                    v = ii / uu if uu else 0.0
                    if v > best_io:
                        best_io = v
                        best_mi = k["bin"]
                st = per[r_][g["bucket"]]
                st["gt"] += 1
                st["short"].append(min(g["bw"], g["bh"]) * r_ / float(SIDE))
                if best_io >= 0.5:
                    st["tp"] += 1
                    ii = int((tc & best_mi).sum())
                    uu = int((tc | best_mi).sum())
                    st["mdice"].append(2 * ii / (int(tc.sum()) + int(best_mi.sum())) if (int(tc.sum()) + int(best_mi.sum())) else 1.0)
                else:
                    st["mdice"].append(0.0)
                inter = int((tc & union).sum())
                uni = int((tc | union).sum())
                io = inter / uni if uni else 0.0
                dc = 2 * inter / (int(tc.sum()) + int(union.sum())) if (int(tc.sum()) + int(union.sum())) else 1.0
                st["sio"].append(io)
                st["sdice"].append(dc)
            time_img[r_] += time.time() - t0
            mem[r_] = max(mem[r_], torch.cuda.max_memory_allocated() / 1e6)
        n_img += 1

    rows = []
    for r_ in RES:
        tp = sum(per[r_][b]["tp"] for b in BUCKETS)
        gt_ = sum(per[r_][b]["gt"] for b in BUCKETS)
        n_pred = 0
        rows.append({"res": r_, "precision": float("nan"), "recall": round(tp / gt_, 4),
                     "f1": float("nan"),
                     "time_ms_img": round(1000 * time_img[r_] / n_img, 1),
                     "gpu_peak_mb": round(mem[r_], 0)})
        for b in BUCKETS:
            st = per[r_][b]
            rows.append({"res": r_, "bucket": b, "n": st["gt"], "tp": st["tp"],
                         "recall_l": round(st["tp"] / st["gt"] if st["gt"] else 0.0, 4),
                         "matched_dice_l": round(float(np.mean(st["mdice"])), 4),
                         "semantic_dice_l": round(float(np.mean(st["sdice"])), 4),
                         "mean_iou_l": round(float(np.mean(st["sio"])), 4),
                         "short_med": round(float(np.median(st["short"])), 1)})
    df = pd.DataFrame(rows)
    df.to_csv(RUN / "exp14_res_ablation.csv", index=False)
    print("EXP14 per-bucket (frozen c2=0.30):")
    bb = df[~df.bucket.isna()]
    print(bb.pivot_table(index="res", columns="bucket",
                         values=["recall_l", "matched_dice_l",
                                 "semantic_dice_l"]).round(4).to_string())
    print("\nEXP14 short-side @stage2-input (median px):")
    print(bb.pivot_table(index="res", columns="bucket",
                         values="short_med").round(1).to_string())
    print("\nEXP14 cost:")
    print(df[df.bucket.isna()][["res", "time_ms_img", "gpu_peak_mb"]].to_string(index=False))


if __name__ == "__main__":
    main()
