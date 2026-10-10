"""PHASE B: frozen Stage1 small-object proposal audit (train split only).

For c1 in {0.05 (primary), 0.02 (reference)}: per GT instance (>=5px),
measure proposal recall under (a) bbox-overlap>0 (loose) and
(b) stage1-mask/GT mask IoU>=0.1 (mid), no-proposal GT fraction, best
proposal confidence median/P25/P75; size stats: GT pixel area, bbox short
side, lesion/image area ratio. Per size bucket.
Trigger: small proposal recall < 0.85 at c1=0.05 (loose) => consider re-train.
Old test untouched. Output: outputs/calculus_cls/phaseB_audit.csv
"""
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

DATA = PROCESSED_DIR / "calculus_seg"
OUT = Path("outputs/calculus_cls")
BUCKET_EDGES = (0.00552, 0.01565)
BUCKETS = ("small", "medium", "large")


def bucket_of(r: float) -> str:
    if r < BUCKET_EDGES[0]:
        return "small"
    if r < BUCKET_EDGES[1]:
        return "medium"
    return "large"


def main() -> None:
    seg = TwoStageSegmenter("outputs/calculus_yolo11m_full/weights/best.pt",
                            "outputs/calculus_yolo11m_crop/weights/best.pt",
                            c1=0.05, c2=0.05, device=0, nms_iou=0.5,
                            score_mode="s2")
    man = pd.read_csv(DATA / "split_manifest.csv")
    # IMPORTANT: Stage1 was trained on train split -> in-sample bias; use the
    # development (main val) split, where Stage1 has never seen the images,
    # for the proposal-recall audit.
    rows = man[man.split == "val"]
    recs = []
    for c1 in (0.05, 0.02):
        seg.c1 = c1
        for ri, (_, r) in enumerate(rows.iterrows()):
            img = cv2.imread(r["image_path"])
            H, W = img.shape[:2]
            gtb = cv2.imread(r["mask_original_path"], cv2.IMREAD_UNCHANGED)
            gtb = (gtb > 0) if gtb is not None else np.zeros((H, W), np.uint8)
            if not gtb.any():
                continue
            n, comp, _, _ = cv2.connectedComponentsWithStats(
                gtb.astype(np.uint8), 8)
            gts = []
            for i in range(1, n):
                if (comp == i).sum() < 5:
                    continue
                m = (comp == i).astype(np.uint8)
                gx, gy, gw, gh = cv2.boundingRect(m)
                px = int(m.sum())
                gts.append({"mask": m, "px": px, "bbox": (gx, gy, gw, gh),
                            "bucket": bucket_of(px / float(H * W))})
            boxes, confs, masks = seg._stage1_masks(r["image_path"])
            for g in gts:
                gx, gy, gw, gh = g["bbox"]
                best_iou_m = 0.0
                best_conf = 0.0
                hit = False
                for bi, (b, msk) in enumerate(zip(boxes, masks)):
                    bc = int(b[0]); b_x2 = int(b[2]); by = int(b[1]); by2 = int(b[3])
                    inter = (max(0, min(gx + gw, b_x2) - max(gx, bc)) *
                             max(0, min(gy + gh, by2) - max(gy, by)))
                    if inter > 0:
                        hit = True
                        best_conf = max(best_conf, float(confs[bi]))
                    msk = msk.astype(np.uint8)
                    inter2 = int((msk & g["mask"]).sum())
                    uni = int((msk | g["mask"]).sum())
                    mio = inter2 / uni if uni else 0.0
                    if mio > best_iou_m:
                        best_iou_m = mio
                recs.append({"c1": c1, "bucket": g["bucket"], "px": g["px"],
                             "bbox_w": gw, "bbox_h": gh,
                             "short": min(gw, gh),
                             "ratio": g["px"] / float(H * W),
                             "hit_bbox": int(hit),
                             "conf": best_conf,
                             "mask_iou": best_iou_m})
            if (ri + 1) % 500 == 0:
                print(f"  c1={c1} {ri + 1}/{len(rows)}", flush=True)
    df = pd.DataFrame(recs)
    df["mid_hit"] = (df.mask_iou >= 0.1).astype(int)
    df.to_csv(OUT / "phaseB_audit.csv", index=False)
    out = []
    for c1 in (0.05, 0.02):
        sub = df[df.c1 == c1]
        for b in BUCKETS:
            s = sub[sub.bucket == b]
            hitc = s[s.hit_bbox == 1].conf
            out.append({
                "c1": c1, "bucket": b, "n_gt": len(s),
                "proposal_recall(bbox>0)": round(float(s.hit_bbox.mean()), 4),
                "proposal_recall(mIoU>=0.1)": round(float(s.mid_hit.mean()), 4),
                "no_proposal_frac": round(float(1 - s.hit_bbox.mean()), 4),
                "conf_med": round(float(hitc.median()), 4) if len(hitc) else None,
                "conf_p25": round(float(hitc.quantile(0.25)), 4) if len(hitc) else None,
                "conf_p75": round(float(hitc.quantile(0.75)), 4) if len(hitc) else None,
                "gt_px_med": round(float(s.px.median()), 1),
                "bbox_short_med": round(float(s.short.median()), 1),
                "ratio_med": round(float(s.ratio.median()), 5)})
    odf = pd.DataFrame(out)
    odf.to_csv(OUT / "phaseB_audit_summary.csv", index=False)
    print(odf.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
