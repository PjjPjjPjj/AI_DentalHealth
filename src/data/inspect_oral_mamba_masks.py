"""Inspect Oral-Mamba masks: count stats, structure, unique values, semantics.

Semantics are NOT assumed. This script only produces facts:
  - image/label pair statistics
  - mask shape/dtype/unique values (sample + full dataset)
  - per-value binary visualization overlays for manual confirmation

Outputs:
  outputs/reports/oral_mamba_mask_statistics.json
  outputs/visualizations/mask_semantics/value_<X>/*.png
  outputs/reports/mask_semantics.md   (UNCONFIRMED until human/visual proof)
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
import json
import os
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.utils.logger import get_logger
from src.utils.paths import OUTPUTS_DIR, REPORTS_DIR, VISUALIZATIONS_DIR

logger = get_logger("oralmamba_inspect")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
MASK_EXTS = {".png"}


def find_pairs(images: list[Path], masks: list[Path]) -> tuple[dict[str, Path], dict[str, Path]]:
    img_by_stem = {p.stem: p for p in images}
    msk_by_stem = {p.stem: p for p in masks}
    common = sorted(set(img_by_stem) & set(msk_by_stem))
    matched = {s: (img_by_stem[s], msk_by_stem[s]) for s in common}
    missing_labels = {s: img_by_stem[s] for s in set(img_by_stem) - set(msk_by_stem)}
    missing_images = {s: msk_by_stem[s] for s in set(msk_by_stem) - set(img_by_stem)}
    return matched, missing_labels, missing_images


def mask_stats(path: Path) -> dict:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        return {"error": "unreadable"}
    if m.ndim == 3:
        ch = m[:, :, 0]
    else:
        ch = m
    uniq = [int(v) for v in sorted(set(np.unique(ch).tolist()))]
    return {
        "shape": list(m.shape),
        "dtype": str(m.dtype),
        "channels": 1 if m.ndim == 2 else m.shape[2],
        "unique_values": uniq,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="datasets/raw/oral_mamba")
    ap.add_argument("--sample-size", type=int, default=50,
                    help="min number of masks to sample for structure stats")
    ap.add_argument("--viz-sample", type=int, default=30,
                    help="min number of overlay groups per candidate value")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-values", type=int, default=8)
    args = ap.parse_args()

    raw = Path(args.raw)
    img_dir = raw / "images"
    msk_dir = raw / "lables"
    out_json = REPORTS_DIR / "oral_mamba_mask_statistics.json"
    out_viz = VISUALIZATIONS_DIR / "mask_semantics"
    out_md = REPORTS_DIR / "mask_semantics.md"

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    masks = sorted(p for p in msk_dir.iterdir() if p.suffix.lower() in MASK_EXTS)

    logger.info(f"images: {len(images)}  masks: {len(masks)}")
    matched, missing_labels, missing_images = find_pairs(images, masks)

    unique_per_mask: Counter = Counter()
    full_mask_value_counter: Counter = Counter()
    value_to_mask_count: Counter = Counter()
    sample_struct: dict[str, list[dict]] = {}
    image_shape_issues: list = []

    paired_files = list(matched.items())
    rng = random.Random(args.seed)
    sample_keys = rng.sample([s for s, _ in paired_files], min(args.sample_size, len(paired_files)))

    for stem, (im, ms) in paired_files:
        m = cv2.imread(str(ms), cv2.IMREAD_UNCHANGED)
        if m is None:
            logger.warning(f"unreadable mask {ms}")
            continue
        ch = m[:, :, 0] if m.ndim == 3 else m
        vals = [int(v) for v in np.unique(ch)]
        uniq_tuple = tuple(vals)
        unique_per_mask[uniq_tuple] += 1
        for v in vals:
            if v != 0:
                value_to_mask_count[v] += 1
            full_mask_value_counter[v] += int((ch == v).sum())
        if stem in sample_keys:
            sample_struct.setdefault(ms.parent.name, []).append({
                "image_id": stem, "shape": list(m.shape), "dtype": str(m.dtype),
                "unique_values": vals,
            })
        img = cv2.imread(str(im))
        if img is not None and img.shape[:2] != m.shape[:2]:
            image_shape_issues.append({
                "image_id": stem, "image_shape": list(img.shape[:2]),
                "mask_shape": list(m.shape[:2]),
            })

    stats = {
        "image_count": len(images),
        "label_count": len(masks),
        "matched_count": len(matched),
        "missing_images": {k: str(v) for k, v in missing_images.items()},
        "missing_labels": {k: str(v) for k, v in missing_labels.items()},
        "mask_shape_mismatch_with_image": image_shape_issues[:50],
        "unique_value_patterns": {str(k): v for k, v in sorted(unique_per_mask.items(), key=lambda x: -x[1])},
        "pixel_frequency": dict(sorted(full_mask_value_counter.items(), key=lambda x: -x[1])),
        "mask_count_containing_value": dict(sorted(value_to_mask_count.items(), key=lambda x: x[0])),
        "sample_structure": sample_struct,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    logger.info(f"wrote {out_json}")

    candidates = [v for v in sorted(value_to_mask_count) if v != 0 and v < args.max_values]
    logger.info(f"candidate non-background values: {candidates}")

    for value in candidates:
        vdir = out_viz / f"value_{value}"
        vdir.mkdir(parents=True, exist_ok=True)
        pool = [s for s, (im, ms) in paired_files]
        rng.shuffle(pool)
        shown = 0
        for stem in pool:
            if shown >= args.viz_sample:
                break
            im, ms = matched[stem]
            img = cv2.imread(str(im))
            if img is None:
                continue
            m = cv2.imread(str(ms), cv2.IMREAD_UNCHANGED)
            if m is None:
                continue
            ch = m[:, :, 0] if m.ndim == 3 else m
            binm = (ch == value).astype(np.uint8) * 255
            if binm.sum() == 0:
                continue
            h, w = img.shape[:2]
            if binm.shape != img.shape[:2]:
                binm = cv2.resize(binm, (w, h), interpolation=cv2.INTER_NEAREST)
            overlay = img.copy()
            overlay[binm > 0] = (overlay[binm > 0] * 0.45 + np.array([0, 0, 255]) * 0.55).astype(np.uint8)
            combo = np.hstack([img, cv2.cvtColor(binm, cv2.COLOR_GRAY2BGR), overlay])
            cv2.imwrite(str(vdir / f"{stem}.png"), combo)
            shown += 1
        logger.info(f"value_{value}: wrote {shown} overlay groups -> {vdir}")

    official_classes = ["caries", "gingivitis", "calculus"]
    undef_lines = [
        "Official disease classes:",
        f"- {'- '.join(official_classes)}".replace("- - ", "- ") if False else "\n".join(f"- {c}" for c in official_classes),
    ]
    md_lines = [
        "# Oral-Mamba mask semantics",
        "",
        f"- Date: {__import__('datetime').date.today().isoformat()}",
        f"- image_count: {len(images)}",
        f"- label_count: {len(masks)}",
        f"- matched_count: {len(matched)}",
        f"- unique pixel values across dataset: {sorted(full_mask_value_counter)}",
        "- Visualization overlays: outputs/visualizations/mask_semantics/",
        "",
        "```text",
        "Official disease classes:",
        "  - caries",
        "  - gingivitis",
        "  - calculus",
        "",
        "Pixel-value mapping:",
        "UNCONFIRMED",
        "```",
        "",
        "Confirmation requires matching overlay visuals of teeth/gum surfaces.",
        "Absent a reliable source (paper / official repo / annotation doc) or",
        "visual confirmation, label conversion must stop at BLOCKED.",
    ]
    out_md.write_text("\n".join(md_lines))
    logger.info(f"wrote {out_md}")


if __name__ == "__main__":
    main()
