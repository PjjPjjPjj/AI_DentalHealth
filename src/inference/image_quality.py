"""Image quality control (traditional CV methods only, V1).

Checks: blur (Laplacian variance), underexposure / overexposure (mean
brightness + dark/bright ratios), highlight ratio, minimum size.
All thresholds live in configs/pipeline.yaml quality section - NO hardcode.
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


def check_quality(image: np.ndarray | str | Path, cfg: dict | None = None) -> dict:
    cfg = cfg or {}
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            return {
                "usable": False, "blur_score": 0.0, "brightness": 0.0,
                "underexposed": False, "overexposed": False,
                "highlight_ratio": 0.0, "error": "image_unreadable",
            }
    else:
        img = image
        if img is None:
            return {
                "usable": False, "blur_score": 0.0, "brightness": 0.0,
                "underexposed": False, "overexposed": False,
                "highlight_ratio": 0.0, "error": "image_unreadable",
            }

    h, w = img.shape[:2]
    if min(h, w) < int(cfg.get("min_size", 320)):
        return {
            "usable": False, "blur_score": 0.0, "brightness": 0.0,
            "underexposed": False, "overexposed": False,
            "highlight_ratio": 0.0, "error": "too_small", "size": f"{w}x{h}",
        }

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except cv2.error:
        blur_score = 0.0

    mean_brightness = float(gray.mean())
    dark_ratio = float((gray < 30).mean())
    bright_ratio = float((gray > 225).mean())
    highlight_ratio = float((gray > 245).mean())

    blurry = blur_score < float(cfg.get("blur_laplacian_threshold", 100.0))
    underexposed = (
        mean_brightness < float(cfg.get("brightness_underexposed", 60.0))
        or dark_ratio > float(cfg.get("dark_pixel_ratio_limit", 0.35))
    )
    overexposed = (
        mean_brightness > float(cfg.get("brightness_overexposed", 200.0))
        or bright_ratio > float(cfg.get("bright_pixel_ratio_limit", 0.35))
    )
    highlight = highlight_ratio > float(cfg.get("highlight_ratio_limit", 0.12))

    usable = not (blurry or underexposed or overexposed)
    # highlights are warnings, not fatal by default
    return {
        "usable": usable,
        "blur_score": round(blur_score, 2),
        "brightness": round(mean_brightness, 2),
        "dark_pixel_ratio": round(dark_ratio, 4),
        "bright_pixel_ratio": round(bright_ratio, 4),
        "underexposed": bool(underexposed),
        "overexposed": bool(overexposed),
        "highlight_ratio": round(highlight_ratio, 4),
        "highlight_warning": bool(highlight),
        "blurry": bool(blurry),
    }
