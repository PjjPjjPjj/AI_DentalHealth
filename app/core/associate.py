"""Lesion-tooth association - same rules as src/inference/associate.py."""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import numpy as np


def associate_calculus(lesion_mask: np.ndarray, tooth_masks: list[np.ndarray],
                       min_overlap: float = 0.20) -> dict | None:
    best_idx, best_overlap = -1, 0.0
    area_lesion = float(np.count_nonzero(lesion_mask > 0))
    if area_lesion <= 0:
        return None
    for i, tm in enumerate(tooth_masks):
        inter = float(np.logical_and(lesion_mask > 0, tm > 0).sum())
        if inter <= 0:
            continue
        overlap = inter / area_lesion
        if overlap > best_overlap:
            best_idx, best_overlap = i, overlap
    if best_idx < 0 or best_overlap < min_overlap:
        return {"assigned": False, "tooth": None,
                "overlap_ratio": round(best_overlap, 4), "area_ratio": 0.0}
    tooth_mask = tooth_masks[best_idx]
    area_tooth = max(1.0, float(np.count_nonzero(tooth_mask > 0)))
    inter = float(np.logical_and(lesion_mask > 0, tooth_mask > 0).sum())
    return {
        "assigned": True,
        "tooth": best_idx,
        "overlap_ratio": round(best_overlap, 4),
        "area_ratio": round(inter / area_tooth, 4),
    }


def _bbox_iou(box, mask: np.ndarray) -> float:
    h, w = mask.shape
    x1, y1, x2, y2 = [float(v) for v in box]
    bx1, by1, bx2, by2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if bx2 <= bx1 or by2 <= by1:
        return 0.0
    mm = mask > 0
    inter = float(mm[int(by1):int(by2), int(bx1):int(bx2)].sum())
    union = float(mm.sum() + (int(by2) - int(by1)) * (int(bx2) - int(bx1)) - inter)
    return inter / max(1.0, union)


def associate_caries(bbox, tooth_masks: list[np.ndarray], center_rule: bool = True,
                     min_iou: float = 0.01) -> dict:
    if not tooth_masks:
        return {"assigned": False, "tooth": None, "iou": 0.0}
    if center_rule:
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        for i, tm in enumerate(tooth_masks):
            m = tm > 0
            h, w = m.shape
            icx, icy = int(round(cx)), int(round(cy))
            if 0 <= icx < w and 0 <= icy < h and bool(m[icy, icx]):
                return {"assigned": True, "tooth": i, "iou": "center_in_mask",
                        "overlap": "center_inside"}
    best_idx, best_iou = -1, 0.0
    for i, tm in enumerate(tooth_masks):
        iou = _bbox_iou(bbox, tm)
        if iou > best_iou:
            best_idx, best_iou = i, iou
    if best_idx < 0 or best_iou < min_iou:
        return {"assigned": False, "tooth": None, "iou": round(best_iou, 4)}
    return {"assigned": True, "tooth": best_idx, "iou": round(best_iou, 4),
            "overlap": "bbox_iou_max"}


def mask_geometry(mask: np.ndarray) -> dict:
    """bbox [x1,y1,x2,y2] px + simplified outer polygons (JSON-safe)."""
    import cv2

    m = (mask > 0).astype(np.uint8)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return {"bbox": None, "polygon": []}
    h, w = m.shape[:2]
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < 30:
            continue
        eps = 0.004 * cv2.arcLength(c, True)
        pts = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        polys.append([[round(float(x) / w, 5), round(float(y) / h, 5)] for x, y in pts])
    return {"bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "polygon": polys}
