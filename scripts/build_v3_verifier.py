"""V3 verifier dataset: two-stage candidates labeled vs GT calculus.

Per image (train+val splits): S1(c1=0.05) -> bbox-center 160 window -> S2
instances (conf>=0.05, intra-window mask-NMS 0.5).  Instance label:
  positive: best IoU (640-space) with GT calculus >= 0.5
  negative: < 0.1  (incl. instances sitting on other-lesion pixels)
  ambiguous (0.1..0.5): excluded
Saved: 160px RGB crop PNG + S2 mask PNG + S1 mask PNG + metadata.csv
Output: datasets/processed/calculus_v3_verifier/
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from scripts.v3_pipeline_lib import s1_pass, s2_instances
from src.inference.fusion import greedy_nms
from src.utils.paths import PROCESSED_DIR

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

DATA = PROCESSED_DIR / "calculus_seg_v3"
OUT = PROCESSED_DIR / "calculus_v3_verifier"
S1_W = "outputs/calculus_v3c_stage1/weights/best.pt"
S2_W = "outputs/calculus_v3b_stage2/weights/best.pt"
C1 = 0.02
SIDE = 160
TARGET = 640
NMS_IOU = 0.5
RAW = Path(f"{_ROOT}/datasets/raw/oral_mamba/lables")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    for sp in ("train", "val"):
        for cls in ("positive", "negative"):
            (OUT / sp / cls).mkdir(parents=True)
    s1, s2 = YOLO(S1_W), YOLO(S2_W)
    man = pd.read_csv(DATA / "split_manifest_v3.csv")
    meta = []
    for sp in ("train", "val"):
        rows = man[man.split == sp].reset_index(drop=True)
        n_pos = n_neg = n_amb = 0
        for ri, r in rows.iterrows():
            img = cv2.imread(r.image_path)
            H, W = img.shape[:2]
            full = cv2.imread(str(RAW / f"{r.image_id}.png"), cv2.IMREAD_UNCHANGED)
            if full is None:
                full = np.zeros((H, W), np.uint8)
            if full.ndim == 3:
                full = full[:, :, 0]
            calc_full = (full == 3).astype(np.uint8)
            other_full = np.isin(full, [1, 2, 4]).astype(np.uint8)

            boxes, confs1, _ = s1_pass(s1, r.image_path)
            for bi, (b, sc1) in enumerate(zip(boxes, confs1)):
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                side = min(SIDE, max(W, H))
                x0 = int(round(cx - side / 2)); y0 = int(round(cy - side / 2))
                x0 = max(0, min(x0, W - side)); y0 = max(0, min(y0, H - side))
                x1, y1 = min(x0 + side, W), min(y0 + side, H)
                crop_img = img[y0:y1, x0:x1]
                if crop_img.size == 0:
                    continue
                crop = cv2.resize(crop_img, (TARGET, TARGET),
                                  interpolation=cv2.INTER_AREA)
                gt_win = cv2.resize(calc_full[y0:y1, x0:x1], (TARGET, TARGET),
                                    interpolation=cv2.INTER_NEAREST)
                ot_win = cv2.resize(other_full[y0:y1, x0:x1], (TARGET, TARGET),
                                    interpolation=cv2.INTER_NEAREST)
                raw_insts = s2_instances(s2, crop)
                if not raw_insts:
                    continue
                sel = [{"x0": 0, "y0": 0, "cw": TARGET, "ch": TARGET,
                        "bin": i["bin640"], "s2_conf": i["s2_conf"],
                        "score": i["s2_conf"]} for i in raw_insts]
                keep = greedy_nms(sel, "mask_nms", NMS_IOU, TARGET, TARGET)
                for k, inst in enumerate(keep):
                    P = inst["bin"]
                    best_iou = 0.0
                    n_gt, comp = cv2.connectedComponents(gt_win, 8)
                    for gi in range(1, n_gt):
                        G = (comp == gi)
                        if G.sum() < 5:
                            continue
                        inter = int((P & G).sum())
                        uni = int((P | G).sum())
                        if uni:
                            best_iou = max(best_iou, inter / uni)
                    if 0.1 <= best_iou < 0.5:
                        n_amb += 1
                        continue
                    label = 1 if best_iou >= 0.5 else 0
                    other_ratio = float((P & (ot_win > 0)).sum()) / max(int(P.sum()), 1)
                    sample_id = f"{r.image_id}__b{bi}i{k}"
                    sub = OUT / sp / ("positive" if label else "negative")
                    cv2.imwrite(str(sub / f"{sample_id}.png"), crop_img)
                    cv2.imwrite(str(sub / f"{sample_id}_s2mask.png"),
                                (P * 255).astype(np.uint8))
                    area = int(P.sum())
                    gx, gy, gw, gh = cv2.boundingRect(P)
                    meta.append({
                        "sample_id": sample_id, "split": sp, "label": label,
                        "other_ratio": round(other_ratio, 4),
                        "s1_conf": float(sc1), "s2_conf": inst["s2_conf"],
                        "s2_area_ratio": area / float(TARGET * TARGET),
                        "bbox_area_ratio": (gw * gh) / float(TARGET * TARGET),
                        "bbox_aspect": min(gw, gh) / max(gw, gh, 1),
                        "best_gt_iou": round(best_iou, 4),
                        "bucket": ("small" if area / float(TARGET * TARGET) < 0.00552
                                   else "medium" if area / float(TARGET * TARGET) < 0.01565
                                   else "large"),
                        "domain": r.domain, "image_id": r.image_id})
                    n_pos += label == 1
                    n_neg += label == 0
            if (ri + 1) % 300 == 0:
                print(f"{sp} {ri + 1}/{len(rows)} pos={n_pos} neg={n_neg} "
                      f"amb={n_amb}", flush=True)
        print(f"[{sp}] pos={n_pos} neg={n_neg} ambiguous_skipped={n_amb}")
    pd.DataFrame(meta).to_csv(OUT / "metadata.csv", index=False)
    print("total samples:", len(meta))


if __name__ == "__main__":
    main()
