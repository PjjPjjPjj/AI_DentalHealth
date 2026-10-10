"""Demo run with available weights only (calculus seg), skipping missing models.

Usage: python scripts/run_demo_calculus.py [--image demo_images/test.jpg]
Outputs:
  outputs/demo/<stem>_result.json
  outputs/demo/<stem>_visualization.jpg
"""
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
import numpy as np

from src.inference.pipeline import OralAIPipeline
from src.utils.logger import get_logger
from src.utils.paths import DEMO_DIR

logger = get_logger("demo_calculus")

GREEN = (0, 200, 0)
DARKGREEN = (0, 120, 0)


def draw_lesions(img: np.ndarray, lesions: list[dict]) -> np.ndarray:
    h, w = img.shape[:2]
    overlay = img.copy()
    for les in lesions:
        polys = les.get("polygon") or []
        for poly in polys:
            pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in poly], dtype=np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(overlay, [pts], DARKGREEN)
        bbox = les.get("bbox")
        if bbox:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, 2)
            cv2.putText(img, f"calculus {les.get('confidence', 0):.2f}",
                        (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="demo_images/test.jpg")
    ap.add_argument("--calculus-weights",
                    default="outputs/calculus_yolo11m_crop/weights/best.pt")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else DEMO_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = OralAIPipeline(
        weights={"calculus": args.calculus_weights},
        skip_missing_models=True,
    )
    result = pipe.run(args.image)
    lesions = result.get("unassigned_lesions", []) + result.get("calculus_lesions", [])
    lesions = [l for l in lesions if l.get("type") == "calculus" or "calculus" in l]
    logger.info(f"quality.usable={result['quality']['usable']} "
                f"teeth={len(result.get('teeth', []))} calculus_lesions={len(lesions)}")

    stem = Path(args.image).stem
    (out_dir / f"{stem}_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    img = cv2.imread(args.image)
    if img is not None:
        img = draw_lesions(img, lesions)
        vis = out_dir / f"{stem}_visualization.jpg"
        cv2.imwrite(str(vis), img)
        logger.info(f"saved {out_dir / (stem + '_result.json')} and {vis}")


if __name__ == "__main__":
    main()
