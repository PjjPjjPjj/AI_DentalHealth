# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
"""Generate the synthetic demo schematic shipped with the repository.

The repository deliberately ships **no patient imagery** (see
THIRD_PARTY_NOTICES.md section 3.1). To keep `demo.py --dry-run` runnable out of
the box, this script draws a purely synthetic schematic of a dental arch.

It is NOT a real intraoral photograph, contains no patient data, and is not
intended to look clinically realistic. Do not use it to evaluate model quality.

Usage:
    python scripts/make_synthetic_demo.py
    # -> demo_images/synthetic_schematic.jpg
"""
from __future__ import annotations

import os

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "demo_images", "synthetic_schematic.jpg")

W, H = 900, 700


def _tooth(mask: np.ndarray, cx: int, cy: int, w: int, h: int) -> None:
    """Draw one schematic tooth: rounded body + slightly darker neck."""
    cv2.ellipse(mask, (cx, cy), (w // 2, h // 2), 0, 0, 360, 255, -1)
    cv2.ellipse(mask, (cx, cy + h // 3), (w // 3, h // 5), 0, 0, 360, 205, -1)


def build() -> np.ndarray:
    img = np.full((H, W, 3), (240, 238, 234), np.uint8)

    # palate / gingiva background (BGR) so the image is not a flat colour
    cv2.ellipse(img, (W // 2, H // 2), (400, 290), 0, 0, 360, (168, 182, 214), -1)

    upper = np.zeros((H, W), np.uint8)
    lower = np.zeros((H, W), np.uint8)

    # upper arch: 10 teeth on a smile curve, kept narrow enough to stay separate
    for i in range(10):
        t = (i - 4.5) / 4.5                      # -1 .. 1
        cx = int(W / 2 + t * 320)
        cy = int(185 + (t ** 2) * 100)
        w = 62 - int(abs(t) * 14)
        h = 100 - int(abs(t) * 16)
        _tooth(upper, cx, cy, w, h)

    # lower arch: 10 teeth mirrored
    for i in range(10):
        t = (i - 4.5) / 4.5
        cx = int(W / 2 + t * 310)
        cy = int(505 - (t ** 2) * 85)
        w = 58 - int(abs(t) * 12)
        h = 92 - int(abs(t) * 14)
        _tooth(lower, cx, cy, w, h)

    img[upper > 0] = (242, 244, 247)
    img[lower > 0] = (242, 244, 247)

    # interproximal separators make the teeth individually countable
    for mask in (upper, lower):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (118, 124, 132), 2)

    # schematic markers: yellow band (calculus-like) and red spot (caries-like)
    cv2.ellipse(img, (int(W / 2 - 104), 250), (24, 10), 0, 0, 360, (60, 200, 235), -1)
    cv2.circle(img, (int(W / 2 + 126), 268), 12, (58, 58, 220), -1)

    label = "SYNTHETIC SCHEMATIC - NOT A PATIENT IMAGE"
    scale, thick = 0.52, 1
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.putText(img, label, ((W - tw) // 2, H - 22),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (88, 88, 88), thick, cv2.LINE_AA)
    return img


def main() -> None:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cv2.imwrite(OUT, build(), [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
