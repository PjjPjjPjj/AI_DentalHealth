"""Build stage-2 crop dataset for two-stage calculus segmentation.

Stage 2 model (YOLO11m-seg @640) trains on LOCAL CROP images:
  - each GT calculus connected component -> bounding box
  - crop window = bbox + 1.8x margin (padded, clamped to image), ensure >=160px
  - letterbox to 640x640 (gray fill), GT polygon transformed accordingly
  - background crops: sampled from negative images (whole-image letterbox)

Per-split output:
  datasets/processed/calculus_seg_crops/{images,labels}/{train,val,test}
  + crops_manifest.csv  (image_id, crop_id, window, scale, split)
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

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from src.utils.logger import get_logger
from src.utils.paths import PROCESSED_DIR

logger = get_logger("twostage_data")

SRC = PROCESSED_DIR / "calculus_seg"
OUT = PROCESSED_DIR / "calculus_seg_crops"
TARGET = 640
MARGIN = 1.8
MIN_WINDOW = 160
BG_PER_SPLIT = {"train": 350, "val": 80, "test": 80}


def find_instances(gt: np.ndarray, min_area: int = 30) -> list[tuple[int, int, int, int]]:
    n, comp, stats, _ = cv2.connectedComponentsWithStats(gt, 8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        boxes.append((x, y, w, h))
    return boxes


def crop_window(b: tuple[int, int, int, int], H: int, W: int) -> tuple[int, int, int, int]:
    x, y, w, h = b
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * MARGIN
    side = max(side, MIN_WINDOW)
    side = min(side, float(max(H, W)))
    x0 = int(round(cx - side / 2)); y0 = int(round(cy - side / 2))
    x0 = max(0, min(x0, W - int(side))); y0 = max(0, min(y0, H - int(side)))
    return x0, y0, int(side), int(side)


def letterbox_crop(img: np.ndarray, x0: int, y0: int, s: int, T: int) -> tuple[np.ndarray, float]:
    crop = img[y0:y0 + s, x0:x0 + s]
    h, w = crop.shape[:2]
    r = T / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((T, T, 3), 114, np.uint8)
    canvas[:nh, :nw] = resized
    return canvas, r


def transform_poly_lines(lines: list[str], x0: int, y0: int, r: float, T: int, W: int, H: int) -> list[str]:
    out_lines = []
    for line in lines:
        p = [float(v) for v in line.split()[1:]]
        n = len(p) // 2
        pts = []
        for i in range(n):
            x = p[2 * i] * W - x0
            y = p[2 * i + 1] * H - y0
            pts.append((min(max(x * r, 0), T), min(max(y * r, 0), T)))
        if len(pts) < 3:
            continue
        out_lines.append("0 " + " ".join(f"{x / T:.6f} {y / T:.6f}" for x, y in pts))
    return out_lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-area", type=int, default=30)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "images").mkdir(parents=True)
    (OUT / "labels").mkdir(parents=True)
    (OUT / "images" / "crops").mkdir()
    (OUT / "images" / "bg").mkdir()

    man = pd.read_csv(SRC / "split_manifest.csv")
    rows = []
    for split in ["train", "val", "test"]:
        sub = man[man.split == split]
        img_dir = OUT / "images" / split
        lbl_dir = OUT / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        n_inst = 0
        for _, r in sub.iterrows():
            img = cv2.imread(r["image_path"])
            if img is None:
                continue
            H, W = img.shape[:2]
            gt = cv2.imread(r["mask_original_path"], cv2.IMREAD_UNCHANGED)
            gt = (gt > 0).astype(np.uint8)
            if bool(r["has_calculus"]):
                boxes = find_instances(gt, args.min_area)
                if not boxes:
                    # fallback: whole-image bbox via mask
                    ys, xs = np.where(gt > 0)
                    boxes = [(xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)]
                for bi, b in enumerate(boxes):
                    x0, y0, s, _ = crop_window(b, H, W)
                    canvas, rr = letterbox_crop(img, x0, y0, s, TARGET)
                    txt = SRC / "labels" / split / (r["image_id"] + ".txt")
                    lines = [ln for ln in txt.read_text().strip().split("\n") if ln]
                    new_lines = transform_poly_lines(lines, x0, y0, rr, TARGET, W, H)
                    if not new_lines:
                        continue
                    cid = f"{r['image_id']}__c{bi}__x{x0}_y{y0}_s{s}"
                    cv2.imwrite(str(img_dir / (cid + ".jpg")), canvas)
                    (lbl_dir / (cid + ".txt")).write_text("\n".join(new_lines))
                    rows.append({"crop_id": cid, "split": split, "type": "instance",
                                 "image_id": r["image_id"], "window": [x0, y0, s], "scale": rr,
                                 "has_calculus": True})
                    n_inst += 1
            else:
                pass
        # background crops from negative images of this split
        neg = list(sub[~sub.has_calculus].sample(frac=1.0, random_state=args.seed).head(BG_PER_SPLIT[split]).itertuples())
        for i, r in enumerate(neg):
            img = cv2.imread(r.image_path)
            if img is None:
                continue
            H, W = img.shape[:2]
            canvas, rr = letterbox_crop(img, 0, 0, max(H, W), TARGET)
            cid = f"bg_crop_{split}_{i:05d}"
            cv2.imwrite(str(img_dir / (cid + ".jpg")), canvas)
            (lbl_dir / (cid + ".txt")).write_text("")
            rows.append({"crop_id": cid, "split": split, "type": "background",
                         "image_id": r.image_id, "window": [0, 0, max(H, W)], "scale": rr,
                         "has_calculus": False})
        logger.info(f"{split}: instances={n_inst} bg={BG_PER_SPLIT[split]}")
    pd.DataFrame(rows).to_csv(OUT / "crops_manifest.csv", index=False)
    data_yaml = {
        "path": str(OUT.resolve()),
        "train": "images/train", "val": "images/val", "test": "images/test",
        "names": {0: "calculus"},
    }
    (OUT / "crops.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    logger.info(f"done -> {OUT}")


if __name__ == "__main__":
    main()
