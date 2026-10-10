"""Convert IO150K / TeethSEG -> tooth instance segmentation dataset.

IO150K-ish layouts (detected at runtime):
  A) raw/io150k/images/*.jpg + masks/*.png (multi-instance masks, each pixel =
     tooth id; instance = connected component / distinct value)
  B) raw/io150k/*/masks/<stem>.png with one mask per tooth (name encoded)
  C) JSON with polygons (coco-style)

V1 classes merged to single 'tooth' (plan 15). Domain notes: split items into
  real RGB / rendered-scan / other where filename or sidecar metadata says so;
  if it cannot be determined, record as 'unknown'.

Output:
  datasets/processed/tooth_seg/{images,labels}/{train,val,test}
  configs/tooth_data.yaml
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
from src.utils.paths import METADATA_DIR, PROCESSED_DIR

logger = get_logger("io150k_convert")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def extract_instances_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    """Multi-instance mask: distinct pixel values > 0 -> instance masks."""
    instances = []
    vals = np.unique(mask)
    for v in vals:
        if v == 0:
            continue
        instances.append((mask == v).astype(np.uint8))
        if len(instances) > 200:
            break
    return instances


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="datasets/raw/io150k")
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw = Path(args.raw)
    out = Path(args.out) if args.out else (PROCESSED_DIR / "tooth_seg")
    out.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in raw.rglob("*") if p.suffix.lower() in IMG_EXTS
                    and "mask" not in p.name.lower() and "masks" not in str(p).lower())
    if not images:
        logger.info(f"No images under {raw} - skipped (BLOCKED_EXTERNAL).")
        return

    mask_dirs = [raw / "masks", raw / "mask", raw / "annotations"]
    mask_dirs = [d for d in mask_dirs if d.exists()]
    pairs: dict[Path, list[np.ndarray]] = {}
    for im in images:
        found = []
        for md in mask_dirs:
            m = md / (im.stem + ".png")
            m2 = md / (im.stem + ".jpg")
            for cand in (m, m2):
                if not cand.exists():
                    continue
                mask = cv2.imread(str(cand), cv2.IMREAD_UNCHANGED)
                if mask is None:
                    continue
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
                found.extend(extract_instances_from_mask(mask))
        if found:
            pairs[im] = found
    logger.info(f"Images {len(images)} with instance masks: {len(pairs)}")
    if not pairs:
        logger.info("No pairs found; check raw layout.")
        return

    rng = random.Random(args.seed)
    keys = sorted(pairs.keys())
    rng.shuffle(keys)
    n = len(keys)
    n_val = max(1, int(n * 0.15)); n_test = max(1, int(n * 0.15))
    parts = {
        "train": keys[: n - n_val - n_test],
        "val": keys[n - n_val - n_test: n - n_test],
        "test": keys[n - n_test:],
    }
    manifest = []
    for split, imgs in parts.items():
        img_dir = out / "images" / split
        lbl_dir = out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for im in imgs:
            img = cv2.imread(str(im))
            if img is None:
                continue
            h, w = img.shape[:2]
            lines = []
            for inst in pairs[im]:
                contours, _ = cv2.findContours(inst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    if cv2.contourArea(c) < 15:
                        continue
                    pts = c.reshape(-1, 2)
                    lines.append("0 " + " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in pts))
            shutil.copy(im, img_dir / im.name)
            (lbl_dir / (im.stem + ".txt")).write_text("\n".join(lines))
            manifest.append({
                "dataset_name": "io150k", "image_id": im.stem,
                "image_path": str(img_dir / im.name),
                "annotation_path": str(lbl_dir / (im.stem + ".txt")),
                "subject_id": split, "source": "IO150K/TeethSEG", "task": "tooth_seg",
                "split": split, "width": w, "height": h,
                "license": "verify - dataset release terms", "original_split": split,
                "processed": True, "notes": "single class 'tooth'; domain=unknown (verify per source)",
            })

    pd.DataFrame(manifest).to_csv(METADATA_DIR / "split_manifest.csv", index=False)
    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train", "val": "images/val", "test": "images/test",
        "names": {0: "tooth"},
    }
    (Path("configs/tooth_data.yaml")).write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    logger.info(f"Converted -> {out}")


if __name__ == "__main__":
    main()
