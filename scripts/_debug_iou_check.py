"""Quick sanity check: compare per-instance max-IoU (single det comp, as in
compute_twostage_map) vs union-IoU on the same images."""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.inference.twostage import TwoStageSegmenter
from src.utils.paths import PROCESSED_DIR

DATA = PROCESSED_DIR / "calculus_seg"
RUN = Path("outputs/calculus_yolo11m_twostage")


def main() -> None:
    thr = json.loads((RUN / "thresholds.json").read_text())
    c1, c2 = float(thr["stage1_conf"]), float(thr["selected_conf"])
    seg = TwoStageSegmenter("outputs/calculus_yolo11m_full/weights/best.pt",
                            "outputs/calculus_yolo11m_crop/weights/best.pt",
                            c1=c1, c2=c2, device=0)
    man = pd.read_csv(DATA / "split_manifest.csv")
    rows = man[(man.split == "test") & man.has_calculus].head(6)

    for _, r in rows.iterrows():
        img = cv2.imread(r["image_path"])
        H, W = img.shape[:2]
        gt = cv2.imread(r["mask_original_path"], cv2.IMREAD_UNCHANGED) > 0
        boxes = seg._stage1(r["image_path"])
        nboxes = 0 if boxes is None else len(boxes)

        dets = []
        union = np.zeros((H, W), np.uint8)
        if boxes is not None:
            for b in boxes:
                x0, y0, x1, y1 = seg._window(b, H, W)
                ch, cw = y1 - y0, x1 - x0
                crop = cv2.resize(img[y0:y1, x0:x1], (640, 640), interpolation=cv2.INTER_AREA)
                st_mask, _ = seg._stage2(crop)
                if st_mask is None:
                    continue
                m = cv2.resize(st_mask, (cw, ch), interpolation=cv2.INTER_NEAREST)
                buf = np.zeros((H, W), np.uint8)
                buf[y0:y1, x0:x1] = m
                union = np.maximum(union, buf)
                n, comp, _, _ = cv2.connectedComponentsWithStats(buf, 8)
                for i in range(1, n):
                    if (comp == i).sum() >= 5:
                        dets.append((comp == i).astype(np.uint8))
        unionb = union > 0
        n, comp, _, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8), 8)
        print(f"{r['image_id']}: HxW={H}x{W} nboxes={nboxes} n_det_comps={len(dets)}")
        for i in range(1, n):
            if (comp == i).sum() < 5:
                continue
            gm = (comp == i).astype(np.uint8)
            ious = []
            for dm in dets:
                inter = int((gm & dm).sum())
                uni = int((gm | dm).sum())
                ious.append(inter / uni if uni else 0.0)
            inter_u = int((gm & unionb).sum())
            uni_u = int((gm | unionb).sum())
            iou_u = inter_u / uni_u if uni_u else 0.0
            mx = max(ious) if ious else 0.0
            px = int(gm.sum())
            print(f"   gt px={px:6d}  max_det_iou={mx:.4f}  union_iou={iou_u:.4f}")


if __name__ == "__main__":
    main()
