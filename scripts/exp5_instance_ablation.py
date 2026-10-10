"""EXP5: raw-stage2-instance candidates + mask-level NMS ablation (VAL only).

Baseline: current pipeline (window-level union of stage2 instances).
Grid:     score = s2_conf            x mask-NMS IoU {0.3, 0.5, 0.7}
          score = 0.7*s2+0.3*s1_conf x mask-NMS IoU {0.3, 0.5, 0.7}
Fixed:    c1=0.05, c2=0.05, margin=1.8, MIN_WINDOW=160, stage2 imgsz=640.
Metrics per size bucket: Dice, mean IoU, Recall@IoU0.3, Recall@IoU0.5,
global: FP image rate (negative val images judged positive @score>=0.05).
Outputs: outputs/calculus_yolo11m_twostage/exp5_instance_ablation.csv
"""
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

from src.inference.fusion import fuse_mask, greedy_nms, rec_bin
from src.inference.twostage import TwoStageSegmenter
from src.utils.paths import PROCESSED_DIR

DATA = PROCESSED_DIR / "calculus_seg"
RUN = Path("outputs/calculus_yolo11m_twostage")
BUCKET_EDGES = (0.00552, 0.01565)
BUCKETS = ("small", "medium", "large")
C1, C2 = 0.05, 0.05
GRID = []
for sc in ("s2", "mix07"):
    for iou in (0.3, 0.5, 0.7):
        GRID.append((f"mask_nms[{sc}@{iou:.1f}]", sc, iou))
GRID.insert(0, ("window_union_legacy", None, None))


def bucket_of(ratio: float) -> str:
    if ratio < BUCKET_EDGES[0]:
        return "small"
    if ratio < BUCKET_EDGES[1]:
        return "medium"
    return "large"


def main() -> None:
    seg = TwoStageSegmenter("outputs/calculus_yolo11m_full/weights/best.pt",
                            "outputs/calculus_yolo11m_crop/weights/best.pt",
                            c1=C1, c2=C2, device=0)
    man = pd.read_csv(DATA / "split_manifest.csv")
    rows = man[man.split == "val"]

    per_cfg = {cfg[0]: {b: [] for b in BUCKETS} for cfg in GRID}
    per_cfg_inst = {cfg[0]: {b: [] for b in BUCKETS} for cfg in GRID}
    n_fp = {cfg[0]: 0 for cfg in GRID}
    n_neg = 0
    for _, r in rows.iterrows():
        img = cv2.imread(r["image_path"])
        H, W = img.shape[:2]
        recs = seg.candidates(r["image_path"], img)
        max_conf = max([cr["s2_conf"] for cr in recs]) if recs else 0.0
        if not bool(r["has_calculus"]):
            n_neg += 1
            if max_conf >= 0.05:
                for cfg in GRID:
                    n_fp[cfg[0]] += 1
            continue
        gt = cv2.imread(r["mask_original_path"], cv2.IMREAD_UNCHANGED) > 0
        n, comp, _, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8), 8)
        gts = []
        for i in range(1, n):
            if (comp == i).sum() < 5:
                continue
            m = (comp == i).astype(np.uint8)
            px = int(m.sum())
            gts.append({"mask": m, "bucket": bucket_of(px / float(H * W))})
        for name, sc_mode, iou in GRID:
            recs2 = list(recs)
            if sc_mode:
                for cr in recs2:
                    cr["score"] = (cr["s2_conf"] if sc_mode == "s2"
                                   else 0.7 * cr["s2_conf"] + 0.3 * cr["s1_conf"])
                mask = fuse_mask(recs2, H, W, "mask_nms", iou) > 0
                out_insts = [rec_bin(cr, H, W) for cr in greedy_nms(recs2,
                                                                    "mask_nms", iou, H, W)]
            else:
                mask = np.zeros((H, W), np.uint8)
                wmap: dict = {}
                for cr in recs2:
                    key = (cr["x0"], cr["y0"], cr["cw"], cr["ch"])
                    w = wmap.get(key)
                    if w is None:
                        w = np.zeros((H, W), np.uint8)
                        wmap[key] = w
                    w[cr["y0"]:cr["y0"] + cr["ch"], cr["x0"]:cr["x0"] + cr["cw"]] |= cr["bin"]
                for w in wmap.values():
                    mask |= w
                mask = mask > 0
                out_insts = list(wmap.values())
            for g in gts:
                gm = g["mask"]
                inter = int((gm & mask).sum())
                uni = int((gm | mask).sum())
                io = inter / uni if uni else 0.0
                dc = 2 * inter / (int(gm.sum()) + int(mask.sum())) if (int(gm.sum()) + int(mask.sum())) else 1.0
                per_cfg[name][g["bucket"]].append((io, dc))
                best_io = 0.0
                for om in out_insts:
                    ii = int((gm & om).sum())
                    uu = int((gm | om).sum())
                    v = ii / uu if uu else 0.0
                    if v > best_io:
                        best_io = v
                per_cfg_inst[name][g["bucket"]].append(best_io)
    rows_l = []
    for name, _, _ in GRID:
        for b in BUCKETS:
            vals = per_cfg[name][b]
            iv = per_cfg_inst[name][b]
            if not vals:
                continue
            ious = np.array([v[0] for v in vals])
            dcs = np.array([v[1] for v in vals])
            inst = np.array(iv)
            rows_l.append({"config": name, "bucket": b, "n": len(vals),
                           "dice_union": float(dcs.mean()),
                           "iou_union": float(ious.mean()),
                           "r03_union": float((ious >= 0.3).mean()),
                           "r05_union": float((ious >= 0.5).mean()),
                           "inst_best_iou": float(inst.mean()),
                           "inst_r03": float((inst >= 0.3).mean()),
                           "inst_r05": float((inst >= 0.5).mean())})
    df = pd.DataFrame(rows_l)
    fp_row = {cfg[0]: n_fp[cfg[0]] / n_neg for cfg in GRID}
    df.to_csv(RUN / "exp5_instance_ablation.csv", index=False)
    pv = df.pivot_table(index="config", columns="bucket",
                        values=["dice_union", "r05_union", "inst_r05",
                                "inst_best_iou"]).round(4)
    print("EXP5 (semantic union mask): per-bucket Dice | R@0.5 | INSTANCE-match R@0.5 | best IoU")
    print(pv.to_string())
    print("\nFP image rate (neg val): ", {k: round(v, 3) for k, v in fp_row.items()})
    print(f"(val negatives={n_neg})")


if __name__ == "__main__":
    main()
