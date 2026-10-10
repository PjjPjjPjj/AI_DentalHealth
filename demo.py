"""OralAI-V1 single-image demo (plan 28).

Usage:
    python demo.py --image demo_images/test.jpg \
        --tooth-weights outputs/checkpoints/tooth/best.pt \
        [--plaque-weights ...] [--calculus-weights ...] [--caries-weights ...]

Outputs:
    outputs/demo/<stem>_result.json
    outputs/demo/<stem>_visualization.jpg

Run without weights: --dry-run uses no model weights and demonstrates the
pipeline structure with an empty result (useful before training data exists).
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
from pathlib import Path

import cv2

from src.inference.pipeline import OralAIPipeline
from src.utils.logger import get_logger
from src.utils.paths import DEMO_DIR, DEMO_IMAGES_DIR
from src.utils.visualization import (
    draw_calculus_masks, draw_caries_boxes, draw_tooth_labels, draw_tooth_outlines,
)

logger = get_logger("demo")


def render_visualization(image_path: Path, result: dict, tooth_masks: list, calculus_masks: list,
                         caries_boxes: list, caries_confs: list, out_path: Path) -> Path:
    img = cv2.imread(str(image_path))
    img = draw_tooth_outlines(img, tooth_masks)
    if calculus_masks:
        img = draw_calculus_masks(img, calculus_masks)
    img = draw_caries_boxes(img, caries_boxes, caries_confs)
    boxes = [t["bbox"] for t in result.get("teeth", [])]
    labels = [t["tooth_id"] for t in result.get("teeth", [])]

    img = draw_tooth_labels(img, boxes, labels)

    # plaque result overlay (blue text at bbox top-right)
    for t in result.get("teeth", []):
        if t.get("plaque", {}).get("confidence") is not None:
            x2, y2 = int(t["bbox"][2]), int(t["bbox"][1])
            cv2.putText(img, f"plaque={t['plaque']['confidence']:.2f}", (max(0, x2 - 130), max(14, y2 + 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
    for les in result.get("unassigned_lesions", []):
        if les.get("type") == "caries" and "bbox" in les:
            x1, y1, x2, y2 = [int(v) for v in les["bbox"]]
            cv2.putText(img, "unassigned", (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 0, 0), 1)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=False)
    parser.add_argument("--tooth-weights", default="outputs/checkpoints/tooth/best.pt")
    parser.add_argument("--plaque-weights", default="outputs/checkpoints/plaque/best.pt")
    parser.add_argument("--calculus-weights", default="outputs/checkpoints/calculus/best.pt")
    parser.add_argument("--caries-weights", default="outputs/checkpoints/caries/best.pt")
    parser.add_argument("--dry-run", action="store_true", help="structure demo, no weights, requires --image")
    args = parser.parse_args()

    if args.image is None:
        candidates = list(DEMO_IMAGES_DIR.glob("*")) + list(DEMO_IMAGES_DIR.glob("**/*"))
        images = [p for p in candidates if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if not images:
            logger.info("No demo image found. Provide --image <path>.")
            return
        image_path = images[0]
    else:
        image_path = Path(args.image)

    if args.dry_run:
        import numpy as np
        from src.inference.image_quality import check_quality
        if not Path(image_path).exists():
            logger.error(
                f"Image not found: {image_path}\n"
                "Point --image at your own intraoral photo, or use the bundled "
                "synthetic schematic (this repo ships no patient imagery):\n"
                "  python demo.py --dry-run --image demo_images/synthetic_schematic.jpg"
            )
            return
        img = cv2.imread(str(image_path))
        if img is None:
            logger.error(f"Could not decode image (unsupported or corrupt): {image_path}")
            return
        q = check_quality(img, {})
        result = {
            "image_id": image_path.name,
            "image_size": {"width": img.shape[1], "height": img.shape[0]},
            "quality": {k: q.get(k) for k in ["usable", "blur_score", "brightness", "highlight_ratio"]},
            "teeth": [], "unassigned_lesions": [],
            "warnings": ["dry_run: no models loaded"],
        }
    else:
        weights = {
            "tooth": args.tooth_weights, "plaque": args.plaque_weights,
            "calculus": args.calculus_weights, "caries": args.caries_weights,
        }
        pipe = OralAIPipeline(weights=weights)
        result = pipe.run(image_path)
        logger.info(f"Pipeline result: {len(result.get('teeth', []))} teeth, "
                    f"{len(result.get('unassigned_lesions', []))} unassigned lesions")

    stem = Path(image_path).stem
    json_out = DEMO_DIR / f"{stem}_result.json"
    (DEMO_DIR).mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    vis_out = DEMO_DIR / f"{stem}_visualization.jpg"
    img = cv2.imread(str(image_path))
    if img is not None:
        img = draw_tooth_labels(img, [t["bbox"] for t in result.get("teeth", [])],
                                [t["tooth_id"] for t in result.get("teeth", [])])
        for t in result.get("teeth", []):
            conf = t.get("plaque", {}).get("confidence")
            plaque_txt = f"plaque={conf:.2f}" if conf is not None else "plaque=NA"
            x2, y2 = int(t["bbox"][2]), max(16, int(t["bbox"][1]) + 16)
            cv2.putText(img, plaque_txt, (max(0, x2 - 130), y2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
        cv2.imwrite(str(vis_out), img)

    logger.info(f"JSON -> {json_out}")
    logger.info(f"Visualization -> {vis_out}")


if __name__ == "__main__":
    main()
