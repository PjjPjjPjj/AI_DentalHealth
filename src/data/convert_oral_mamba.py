"""Convert Oral-Mamba -> calculus segmentation YOLO-seg dataset.

Verified raw layout (from the uploaded `oral_mamba` copy, audited 2026-08-26):
  raw/oral_mamba/
      images/*.jpg     (3375)
      lables/*.png     (3365, single-channel multi-class masks)

Class map (verified by filename-prefix + visual probe + area stats):
  0=background, 1=gingivitis, 2=plaque, 3=calculus, 4=caries
V1 uses calculus only: mask value == CLASS_ID (default 3).

The converter accepts variants too: mask/calculus, masks, mask/ dirs and
binary masks (value>0). Class id decisions are documented in the metadata
manifest notes.

Output:
  datasets/processed/calculus_seg/{images,labels}/{train,val,test}
  configs/calculus_data.yaml
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

logger = get_logger("oralmamba_convert")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_CLASS_ID = 3  # calculus, per verified class map
CLASS_MAP = {0: "background", 1: "gingivitis", 2: "plaque", 3: "calculus", 4: "caries"}


def find_pair(images: list[Path], mask_dirs: list[Path]) -> dict[Path, Path]:
    pairs: dict[Path, Path] = {}
    for im in images:
        for md in mask_dirs:
            cand = md / (im.stem + ".png")
            if cand.exists():
                pairs[im] = cand
                break
    return pairs


def load_calculus_mask(path: Path, class_id: int, img_shape: tuple[int, int],
                       mode: str = "auto") -> np.ndarray | None:
    """Binary mask of calculus class from a PNG mask file.

    mode:
      auto  - max pixel >= 128 -> binary (keep >127); else multi-class (keep == class_id)
      binary - keep (m > 127)
      class  - keep (m == class_id)
    Validated on the uploaded Oral-Mamba copy: multi-class values 0..4, so
    calculus = class_id 3. Legacy binary layouts keep >127.
    """
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        m = m[:, :, 0]
    uniq = sorted(set(int(v) for v in np.unique(m)))
    if mode == "auto":
        mode = "binary" if (uniq and uniq[-1] >= 128) else "class"
    if mode == "binary":
        mask = (m > 127).astype(np.uint8)
    else:
        mask = (m == class_id).astype(np.uint8)
    h, w = img_shape
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="datasets/raw/oral_mamba")
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-id", type=int, default=DEFAULT_CLASS_ID,
                        help=f"class id in multi-class masks (default {DEFAULT_CLASS_ID}=calculus)")
    args = parser.parse_args()

    raw = Path(args.raw)
    out = Path(args.out) if args.out else (PROCESSED_DIR / "calculus_seg")
    out.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in raw.rglob("*") if p.suffix.lower() in IMG_EXTS
                    and "annotat" not in str(p).lower()
                    # exclude any images that might be inside mask-ish dirs
                    and not str(p.parent).replace(str(raw), "").startswith(
                        ("/lables", "/mask", "/masks", "/annotations")))

    mask_dirs = [raw / "lables", raw / "labels", raw / "mask" / "calculus",
                 raw / "mask" / "masks", raw / "masks", raw / "mask"]
    mask_dirs = [d for d in mask_dirs if d.exists()]
    if not mask_dirs:
        logger.info(f"No mask dirs under {raw} - conversion skipped (BLOCKED_EXTERNAL).")
        return

    pairs = find_pair(images, mask_dirs)
    logger.info(f"Images {len(images)}, mask pairs {len(pairs)}")
    if not pairs:
        logger.info("No image-mask pairs found.")
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
            mask = pairs[im]
            img = cv2.imread(str(im))
            if img is None:
                continue
            m_bin = load_calculus_mask(mask, args.class_id, img.shape[:2])
            if m_bin is None:
                logger.warning(f"Unreadable mask {mask} - skipped")
                continue
            h, w = img.shape[:2]
            if int(m_bin.sum()) == 0:
                logger.warning(f"No class {args.class_id} ({CLASS_MAP.get(args.class_id)}) pixels in {mask} - skipped")
                continue
            # YOLO polygon label: outer contours of the binary mask
            contours, _ = cv2.findContours(m_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            lines = []
            for c in contours:
                if cv2.contourArea(c) < 10:
                    continue
                pts = c.reshape(-1, 2)
                poly = f"0 " + " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in pts)
                lines.append(poly)
            dst_img = img_dir / im.name
            shutil.copy(im, dst_img)
            (lbl_dir / (im.stem + ".txt")).write_text("\n".join(lines))
            manifest.append({
                "dataset_name": "oral_mamba", "image_id": im.stem,
                "image_path": str(dst_img), "annotation_path": str(lbl_dir / (im.stem + ".txt")),
                "subject_id": split, "source": "Oral-Mamba", "task": "calculus_seg",
                "split": split, "width": w, "height": h,
                "license": "creative commons - verify original release terms", "original_split": split,
                "processed": True,
                "notes": (f"multi-class mask class map {CLASS_MAP}; V1 keeps calculus only "
                          f"(class_id={args.class_id}); polygons from binary mask"),
            })

    pd.DataFrame(manifest).to_csv(METADATA_DIR / "split_manifest.csv", index=False)
    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train", "val": "images/val", "test": "images/test",
        "names": {0: "calculus"},
    }
    (Path("configs/calculus_data.yaml")).write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    logger.info(f"Converted -> {out}")


if __name__ == "__main__":
    main()
