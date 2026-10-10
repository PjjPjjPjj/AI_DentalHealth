"""EXP18B: Verifier B false-negative audit (validation positive instances).

Groups positive instances by verifier score: <0.05 / 0.05-0.20 / 0.20-0.50 /
>=0.50. Stats per group: n, bucket counts, s1_conf/s2_conf (mean/median),
best GT IoU (mean/median), stage1/stage2 mask area ratio (to crop 160^2),
s1-s2 mask IoU, bbox w/h, centroid (u,v).
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from pathlib import Path

V2 = Path("datasets/processed/calculus_v2/verifier_dataset")
OUT = Path("outputs/calculus_cls")


def main() -> None:
    meta = pd.read_csv(V2 / "metadata.csv")
    meta["label"] = meta.label.map({"positive": 1, "negative": 0})
    sc = pd.read_csv(OUT / "verifier_B" / "val_scores.csv")
    m = meta[meta.split == "val"].merge(sc, left_on="sample_id", right_on="sample_id",
                                        how="left")
    m = m[m.label == 1].copy()  # positives only
    groups = [("<0.05", lambda s: s < 0.05),
              ("0.05-0.20", lambda s: (s >= 0.05) & (s < 0.20)),
              ("0.20-0.50", lambda s: (s >= 0.20) & (s < 0.50)),
              (">=0.50", lambda s: s >= 0.50)]
    rows = []
    for name, mask in groups:
        sub = m[m.score.map(mask)].copy()
        if len(sub) == 0:
            rows.append({"group": name, "n": 0})
            print(f"== {name} (n=0) ==")
            continue
        # extras: bbox/centroid of s2 mask
        bw, bh, cx, cy = [], [], [], []
        for _, r in sub.iterrows():
            mk = cv2.imread(str(V2 / "val" / ("positive" if r.label == 1 else "negative")
                                / f"{r.sample_id}_s2mask.png"),
                            cv2.IMREAD_GRAYSCALE)
            mk = (mk > 127).astype(np.uint8)
            gx, gy, gw, gh = cv2.boundingRect(mk)
            bw.append(gw); bh.append(gh)
            ys, xs = np.nonzero(mk)
            cx.append(float(xs.mean()) / 160.0 if len(xs) else 0.5)
            cy.append(float(ys.mean()) / 160.0 if len(ys) else 0.5)
        sub["bw"], sub["bh"] = bw, bh
        sub["cx"], sub["cy"] = cx, cy
        rows.append({
            "group": name, "n": len(sub),
            "small": int((sub.bucket == "small").sum()),
            "medium": int((sub.bucket == "medium").sum()),
            "large": int((sub.bucket == "large").sum()),
            "s1_conf_mean": round(float(sub.s1_conf.mean()), 3),
            "s1_conf_med": round(float(sub.s1_conf.median()), 3),
            "s2_conf_mean": round(float(sub.s2_conf.mean()), 3),
            "s2_conf_med": round(float(sub.s2_conf.median()), 3),
            "gt_iou_mean": round(float(sub.best_gt_iou.mean()), 3),
            "gt_iou_med": round(float(sub.best_gt_iou.median()), 3),
            "s2area_ratio_mean": round(float((sub.s2_mask_area / (160 * 160)).mean()), 4),
            "s2area_ratio_med": round(float((sub.s2_mask_area / (160 * 160)).median()), 4),
            "s1area_ratio_med": round(float((sub.s1_mask_area / (160 * 160)).median()), 4),
            "s1s2_iou_mean": round(float(sub.s1_s2_mask_iou.mean()), 3),
            "bbox_w_med": round(float(sub.bw.median()), 1),
            "bbox_h_med": round(float(sub.bh.median()), 1),
            "centroid_u": round(float(sub.cx.mean()), 3),
            "centroid_v": round(float(sub.cy.mean()), 3),
        })
        print(f"== {name} n={len(sub)} ==")
        print(rows[-1])
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exp18b_fn_audit.csv", index=False)
    print("\nsaved exp18b_fn_audit.csv")


if __name__ == "__main__":
    main()
