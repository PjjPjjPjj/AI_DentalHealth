"""Visualization utilities: draw model results on images for demo/error analysis."""
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

COLOR_TOOTH_OUTLINE = (0, 200, 255)   # yellow-orange
COLOR_CALCULUS = (0, 165, 255)        # orange
COLOR_CARIES = (0, 0, 255)            # red
COLOR_PLAQUE = (255, 0, 0)            # blue
COLOR_TEXT = (255, 255, 255)


def draw_tooth_outlines(img: np.ndarray, tooth_masks: list[np.ndarray]) -> np.ndarray:
    """Draw instance outlines (polygon contours) of tooth masks in place copy."""
    out = img.copy()
    for mask in tooth_masks:
        m = mask if mask.max() <= 1 else (mask > 0).astype(np.uint8)
        m = m.astype(np.uint8) * 255
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, COLOR_TOOTH_OUTLINE, 2)
    return out


def draw_calculus_masks(img: np.ndarray, masks: list[np.ndarray], alpha: float = 0.45) -> np.ndarray:
    """Overlay calculus masks with translucency."""
    from .paths import LOGS_DIR  # noqa: F401  (avoid unused import warnings downstream)
    out = img.copy()
    overlay = img.copy()
    for mask in masks:
        m = (mask > 0) if mask.max() > 1 else mask.astype(bool)
        overlay[m] = COLOR_CALCULUS
    out = cv2.addWeighted(out, 1 - alpha, overlay, alpha, 0)
    return out


def draw_caries_boxes(img: np.ndarray, boxes: list[list[float]], confs: list[float]) -> np.ndarray:
    out = img.copy()
    for bbox, conf in zip(boxes, confs):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), COLOR_CARIES, 2)
        label = f"caries {conf:.2f}"
        cv2.putText(out, label, (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CARIES, 2)
    return out


def draw_tooth_labels(img: np.ndarray, tooth_boxes: list[list[float]],
                      tooth_ids: list[int], note: str = "") -> np.ndarray:
    out = img.copy()
    for bbox, tid in zip(tooth_boxes, tooth_ids):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), COLOR_TOOTH_OUTLINE, 1)
        label = f"T{tid}"
        if note:
            label += f" {note}"
        cv2.putText(out, label, (x1, y1 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TOOTH_OUTLINE, 2)
    return out


def save_figure(img: np.ndarray, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return out_path
