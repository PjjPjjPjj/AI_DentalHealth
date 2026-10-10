"""EXP17: candidate-level label audit of the classifier positive set.

For every Stage1 candidate (bbox + its own instance mask) on train split
(c1=0.05), we recompute:
  1 mask IoU vs each GT instance   (full-image space)
  2 bbox IoU vs GT bbox
  3 candidate-mask coverage by GT  (|cand & GT| / |cand|)
  4 GT coverage by candidate-mask  (|cand & GT| / |GT|)
  5 center distance to nearest GT center
Only candidates whose CROP contains GT (max inclusion>=0.8; same rule as the
classifier dataset positive definition) are audited; per candidate the GT
with the best mask-IoU is treated as the intended match.

Classes:
  A "clear good candidate"     : candidate really matches a GT
  B "crop-has-GT but candidate mismatches": candidate barely touches GT
Counts & ratios per size bucket. No dataset modification, no training.
Outputs: outputs/calculus_cls/exp17_label_audit.csv
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

from src.inference.twostage import TARGET, TwoStageSegmenter
from src.utils.paths import PROCESSED_DIR

DATA = PROCESSED_DIR / "calculus_seg"
OUT = Path("outputs/calculus_cls")
C1 = 0.05
SIDE = 160
BUCKET_EDGES = (0.00552, 0.01565)
BUCKETS = ("small", "medium", "large")


def bucket_of(r: float) -> str:
    if r < BUCKET_EDGES[0]:
        return "small"
    if r < BUCKET_EDGES[1]:
        return "medium"
    return "large"


def instances(mask: np.ndarray, min_area=5):
    n, comp, _, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        if (comp == i).sum() < min_area:
            continue
        m = (comp == i).astype(np.uint8)
        gx, gy, gw, gh = cv2.boundingRect(m)
        out.append({"mask": m, "box": (gx, gy, gx + gw, gy + gh),
                    "px": int(m.sum()), "bucket": None})
    return out


def main() -> None:
    seg = TwoStageSegmenter("outputs/calculus_yolo11m_full/weights/best.pt",
                            "outputs/calculus_yolo11m_crop/weights/best.pt",
                            c1=C1, c2=0.05, device=0, nms_iou=0.5, score_mode="s2")
    man = pd.read_csv(DATA / "split_manifest.csv")
    rows = man[man.split == "train"]
    recs = {b: [] for b in BUCKETS}
    n_candidates = 0
    for ri, (_, r) in enumerate(rows.iterrows()):
        img = cv2.imread(r["image_path"])
        H, W = img.shape[:2]
        gt = cv2.imread(r["mask_original_path"], cv2.IMREAD_UNCHANGED)
        gtb = (gt > 0) if gt is not None else np.zeros((H, W), np.uint8)
        full_gts = instances(gtb)
        for g in full_gts:
            g["bucket"] = bucket_of(g["px"] / float(H * W))
        boxes, confs, masks = seg._stage1_masks(r["image_path"])
        for (b, _sc1, msk) in zip(boxes, confs, masks):
            if msk is None or len(msk.shape) > 2 or msk.sum() == 0:
                continue
            n_candidates += 1
            bx = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
            # crop presence rule (same as classifier dataset)
            x0 = int(round((bx[0] + bx[2]) / 2 - SIDE / 2))
            y0 = int(round((bx[1] + bx[3]) / 2 - SIDE / 2))
            x0 = max(0, min(x0, W - SIDE)); y0 = max(0, min(y0, H - SIDE))
            x1, y1 = x0 + SIDE, y0 + SIDE
            crop_gts, incs = [], []
            for g in full_gts:
                inter_c = int(g["mask"][y0:y1, x0:x1].sum())
                if inter_c > 0:
                    crop_gts.append(g)
                    incs.append(inter_c / float(g["px"]) if g["px"] else 0.0)
            if not crop_gts or max(incs) < 0.8:
                continue
            # best-matched GT for this candidate
            msk = msk.astype(np.uint8)
            best = None
            for g in full_gts:
                gm = g["mask"].astype(np.uint8)
                inter = int((msk & gm).sum())
                uni = int((msk | gm).sum())
                mio = inter / uni if uni else 0.0
                if best is None or mio > best[0]:
                    best = (mio, g, inter)
            mio, g, inter = best
            bx_inter = (max(0, min(bx[2], g["box"][2]) - max(bx[0], g["box"][0])) *
                        max(0, min(bx[3], g["box"][3]) - max(bx[1], g["box"][1])))
            bbox_iou = bx_inter / float(
                (bx[2] - bx[0]) * (bx[3] - bx[1]) +
                (g["box"][2] - g["box"][0]) * (g["box"][3] - g["box"][1]) - bx_inter) \
                if bx_inter else 0.0
            cand_cov = inter / float(msk.sum()) if msk.sum() else 0.0
            gt_cov = inter / float(g["px"])
            cx1, cy1 = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
            cx2, cy2 = (g["box"][0] + g["box"][2]) / 2, (g["box"][1] + g["box"][3]) / 2
            dist = float(np.hypot(cx1 - cx2, cy1 - cy2))
            recs[g["bucket"]].append({"mask_iou": mio, "bbox_iou": bbox_iou,
                                      "cand_cov_by_gt": cand_cov,
                                      "gt_cov_by_cand": gt_cov,
                                      "center_dist": dist})
    rows = []
    for b in BUCKETS:
        v = recs[b]
        if not v:
            continue
        df = pd.DataFrame(v)
        for bt, br in [("clear(mIoU>=0.3 & gtCov>=0.5)",
                        (df.mask_iou >= 0.3) & (df.gt_cov_by_cand >= 0.5)),
                       ("partial(0.1<=mIoU<0.3)",
                        (df.mask_iou >= 0.1) & (df.mask_iou < 0.3)),
                       ("bad(mIoU<0.1)",
                        df.mask_iou < 0.1)]:
            rows.append({"bucket": b, "audit_class": bt,
                         "n": int(br.sum()),
                         "ratio": round(float(br.mean()), 4)})
        quant = df[["mask_iou", "bbox_iou", "cand_cov_by_gt",
                    "gt_cov_by_cand", "center_dist"]].quantile(
            [0.25, 0.5, 0.75]).round(4)
        rows.append({"bucket": b, "audit_class": "q25/50/75",
                     "n": int(len(df)), "ratio": -1.0})
        print(f"== bucket {b} (n={len(df)}) quantiles ==")
        print(quant.to_string())
    adf = pd.DataFrame(rows)
    adf.to_csv(OUT / "exp17_label_audit.csv", index=False)
    print("\n== summary (per bucket) ==")
    print(adf[adf.audit_class != "q25/50/75"].pivot_table(
        index="audit_class", columns="bucket", values=["n", "ratio"]).round(4)
        .to_string())
    print(f"\ncandidates total scanned={n_candidates}")


if __name__ == "__main__":
    main()
