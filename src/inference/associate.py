"""Lesion-tooth association (plan 19).

Rules:
  Calculus: overlap_ratio = area(lesion & tooth) / area(lesion); assign to the
  tooth with max overlap; require >= calculus_min_overlap; also report
  area_ratio = area(lesion & tooth) / area(tooth).
  Caries: bbox center in tooth mask first; else max overlap of bbox vs mask;
  below caries_min_iou => 'unassigned', never force-match.
  Plaque: V1 uses per-tooth-crop probability (whole tooth crop), G/I/M/D
  interfaces are reserved (return None) until reliable region geometry exists.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import numpy as np


def _mask_area(mask: np.ndarray) -> float:
    return float(np.count_nonzero((mask > 0) if mask.max() > 1 else (mask > 0)))


def _intersection(a: np.ndarray, b: np.ndarray) -> float:
    aa = (a > 0) if a.max() > 1 else a.astype(bool)
    bb = (b > 0) if b.max() > 1 else b.astype(bool)
    if aa.shape != bb.shape:
        raise ValueError(f"mask shape mismatch: {aa.shape} vs {bb.shape}")
    return float(np.logical_and(aa, bb).sum())


def _bbox_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where((mask > 0) if mask.max() > 1 else mask.astype(bool))
    if len(xs) == 0:
        return np.zeros_like(mask, dtype=bool)
    m = np.zeros_like(mask, dtype=bool)
    m[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1] = True
    return m


def _bbox_iou(box: list[float], mask: np.ndarray) -> float:
    h, w = mask.shape
    x1, y1, x2, y2 = [float(v) for v in box]
    bx1, by1, bx2, by2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if bx2 <= bx1 or by2 <= by1:
        return 0.0
    box_mask = np.zeros((h, w), dtype=bool)
    box_mask[int(by1): int(by2), int(bx1): int(bx2)] = True
    mm = (mask > 0) if mask.max() > 1 else mask.astype(bool)
    inter = np.logical_and(box_mask, mm).sum()
    union = np.logical_or(box_mask, mm).sum()
    return float(inter) / max(1.0, float(union))


def associate_calculus(lesion_mask: np.ndarray, tooth_masks: list[np.ndarray],
                       min_overlap: float = 0.20) -> dict:
    """Return {tooth_index, overlap_ratio, area_ratio} or None (unassigned)."""
    best_idx, best_overlap = -1, 0.0
    area_lesion = float(np.count_nonzero(lesion_mask > 0))
    if area_lesion <= 0:
        return None
    for i, tm in enumerate(tooth_masks):
        inter = _intersection(lesion_mask, tm)
        if inter <= 0:
            continue
        overlap = inter / area_lesion
        if overlap > best_overlap:
            best_idx, best_overlap = i, overlap
    if best_idx < 0 or best_overlap < min_overlap:
        return {"assigned": False, "tooth": None, "overlap_ratio": round(best_overlap, 4),
                "area_ratio": 0.0}
    tooth_mask = tooth_masks[best_idx]
    area_tooth = max(1.0, float(np.count_nonzero(tooth_mask > 0)))
    inter = _intersection(lesion_mask, tooth_mask)
    return {
        "assigned": True,
        "tooth": best_idx,
        "overlap_ratio": round(best_overlap, 4),
        "area_ratio": round(inter / area_tooth, 4),
    }


def associate_caries(bbox: list[float], tooth_masks: list[np.ndarray],
                     center_rule: bool = True, min_iou: float = 0.01) -> dict:
    """Center-in-mask first, then max bbox-mask IoU; unassigned instead of forced."""
    if not tooth_masks:
        return {"assigned": False, "tooth": None, "iou": 0.0}
    if center_rule:
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        for i, tm in enumerate(tooth_masks):
            m = (tm > 0) if tm.max() > 1 else tm.astype(bool)
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


def compute_plaque_regions(tooth_mask: np.ndarray) -> dict:
    """Reserved G/I/M/D geometry. V1 returns None for all four and does not
    fabricate unreliable region divisions (plan 19 note)."""
    return {"gingival": None, "incisal": None, "mesial": None, "distal": None}
