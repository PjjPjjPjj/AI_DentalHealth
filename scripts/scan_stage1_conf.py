"""Stage-1 coverage rate vs c1 conf threshold.

For each test-positive image, run stage1 (whole-image YOLO11m-seg) at several
conf values and measure how many GT instances (per size bucket) get a
candidate box with bbox overlap > 0.
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
from ultralytics import YOLO

from src.inference.twostage import TARGET
from src.utils.paths import PROCESSED_DIR

DATA = PROCESSED_DIR / "calculus_seg"
RUN = Path("outputs/calculus_yolo11m_twostage")


def main() -> None:
    man = pd.read_csv(DATA / "split_manifest.csv")
    rows = man[(man.split == "test") & man.has_calculus]
    model = YOLO("outputs/calculus_yolo11m_full/weights/best.pt")
    confs = [0.05, 0.1, 0.15, 0.2, 0.3]

    insts_by_img = {}  # image_id -> [ (box, bucket) ]
    for _, r in rows.iterrows():
        img = cv2.imread(r["image_path"])
        H, W = img.shape[:2]
        gt = cv2.imread(r["mask_original_path"], cv2.IMREAD_UNCHANGED) > 0
        n, comp, _, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8), 8)
        ii = []
        for i in range(1, n):
            if (comp == i).sum() < 5:
                continue
            m = (comp == i).astype(np.uint8)
            import numpy as _np
            gx, gy, gw, gh = cv2.boundingRect(m)
            gpx = int(m.sum())
            ratio = gpx / float(H * W)
            bucket = "small" if ratio < 0.00552 else ("medium" if ratio < 0.01565 else "large")
            ii.append(((gx, gy, gx + gw, gy + gh), bucket))
        insts_by_img[r["image_id"]] = ii

    seen = {"small": [0, 0], "medium": [0, 0], "large": [0, 0]}
    counts = {c: {b: [0, 0] for b in ("small", "medium", "large")} for c in confs}
    for _, r in rows.iterrows():
        ii = insts_by_img[r["image_id"]]
        for b in ("small", "medium", "large"):
            seen[b][1] += sum(1 for _, bb in ii if bb == b)
        for c in confs:
            res = model.predict(r["image_path"], conf=c, iou=0.6, imgsz=TARGET,
                                device=0, retina_masks=True, max_det=100,
                                verbose=False)[0]
            cand = (res.boxes.xyxy.cpu().numpy()
                    if (res.boxes is not None and len(res.boxes)) else np.zeros((0, 4)))
            for (gx, gy, gx2, gy2), bucket in ii:
                hit = False
                for bx0, by0, bx1, by1 in cand:
                    ix0, iy0 = float(max(gx, bx0)), float(max(gy, by0))
                    ix1, iy1 = float(min(gx2, bx1)), float(min(gy2, by1))
                    if max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0) > 0:
                        hit = True
                        break
                counts[c][bucket][1] += 1
                if hit:
                    counts[c][bucket][0] += 1

    lines = ["c1      small            medium           large"]
    for c in confs:
        parts = []
        for b in ("small", "medium", "large"):
            k, n = counts[c][b]
            parts.append(f"{k}/{n} ({k / n * 100:.1f}%)" if n else "0/0")
        lines.append(f"{c:5.2f}  {parts[0]:>16s}  {parts[1]:>16s}  {parts[2]:>16s}")
    print("\n".join(lines))
    (RUN / "stage1_conf_scan.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
