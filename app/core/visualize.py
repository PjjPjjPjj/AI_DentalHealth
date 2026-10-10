"""Detection visualization: blue=plaque, yellow=calculus, red=caries.

Draws colored boxes (+ optional translucent masks) with class label,
confidence and index. Chinese label rendering uses the bundled NotoSansSC.
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
from PIL import Image

FONT_PATH = Path(__file__).resolve().parents[1] / "fonts" / "NotoSansSC.ttf"

COLORS = {
    "plaque": (255, 144, 30),    # blue (BGR)
    "calculus": (0, 215, 255),   # yellow
    "caries": (0, 0, 255),       # red
}
LABELS = {"plaque": "Plaque", "calculus": "Calculus", "caries": "Caries"}
_MASK_ALPHA = 0.35
_FONT_CACHE: dict[int, object] = {}


def _font(size: float):
    key = int(size)
    if key not in _FONT_CACHE:
        from PIL import ImageFont

        _FONT_CACHE[key] = ImageFont.truetype(str(FONT_PATH), key)
    return _FONT_CACHE[key]


def _put_text(img: np.ndarray, text: str, org: tuple, size: float = 16,
              color=(255, 255, 255), bg=None) -> None:
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    from PIL import ImageDraw

    draw = ImageDraw.Draw(pil_img, "RGBA")
    f = _font(size)
    if bg is not None:
        x0, y0 = org
        tb = draw.textbbox((x0, y0), text, font=f)
        draw.rectangle([tb[0] - 2, tb[1] - 1, tb[2] + 2, tb[3] + 1],
                       fill=(int(bg[2]), int(bg[1]), int(bg[0]), 200))
    draw.text(org, text, font=f, fill=(color[2], color[1], color[0]))
    out = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    img[:] = out


def annotate(image_bgr: np.ndarray, result: dict, show_masks: bool = True,
             masks: list | None = None) -> np.ndarray:
    """Draw all lesions on a copy of the image.

    masks: optional list of (type, mask) tuples for translucent overlay
    (segmentation > bbox per SRS). plaque has no mask (per-tooth classifier).
    """
    vis = image_bgr.copy()
    if show_masks and masks:
        overlay = vis.copy()
        for _typ, m in masks:
            overlay[m > 0] = COLORS.get(_typ, (255, 255, 255))
        vis = cv2.addWeighted(overlay, _MASK_ALPHA, vis, 1 - _MASK_ALPHA, 0)

    def draw_box(bbox, color, lines):
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        ty = y1 - 4
        for line in lines:
            th = 15
            _put_text(vis, line, (x1, max(0, ty - th)), size=15,
                      color=(255, 255, 255), bg=color)
            ty -= th - 2

    idx = 1
    for les in result.get("calculus_lesions", []):
        bbox = les.get("bbox") or _bbox_of_polygons(les.get("polygon"), vis.shape)
        if bbox is None:
            continue
        draw_box(bbox, COLORS["calculus"],
                 [f"#{idx} Calculus {les['confidence']:.2f}"])
        idx += 1
    idx = 1
    for les in result.get("caries_lesions", []):
        draw_box(les["bbox"], COLORS["caries"],
                 [f"#{idx} Caries {les['confidence']:.2f}"])
        idx += 1
    idx = 1
    for t in result.get("teeth", []):
        if t["plaque"]["present"]:
            draw_box(t["bbox"], COLORS["plaque"],
                     [f"#{idx} Plaque {t['plaque']['confidence']:.2f}"])
            idx += 1
    return vis


def _bbox_of_polygons(polys, shape) -> list | None:
    if not polys:
        return None
    h, w = shape[:2]
    xs, ys = [], []
    for poly in polys:
        for px, py in poly:
            xs.append(px * w)
            ys.append(py * h)
    if not xs:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def collect_lesion_masks(result: dict, image_hw) -> list:
    """[(type, mask)] for seg overlay: calculus polygons -> filled masks."""
    h, w = image_hw
    out = []
    for les in result.get("calculus_lesions", []):
        import cv2 as _cv

        m = np.zeros((h, w), np.uint8)
        for poly in les.get("polygon", []):
            pts = np.array([[int(px * w), int(py * h)] for px, py in poly], np.int32)
            if len(pts) >= 3:
                _cv.fillPoly(m, [pts], 1)
        if m.any():
            out.append(("calculus", m))
    return out
