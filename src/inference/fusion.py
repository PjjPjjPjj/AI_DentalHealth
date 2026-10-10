"""Candidate mask fusion for the two-stage pipeline.

Candidate record schema (what stage2 produces per stage1 box):
  x0, y0, cw, ch, bin (uint8 ROI, same shape as window), soft (uint8 = prob*255
  ROI), s1_conf, s2_conf, fg_prob, score = s1_conf * fg_prob.

Fusion modes (no GT ever used here; ablation only via validation):
  union              - OR all candidates (legacy, NOT used by default)
  best_score_single  - keep only the highest-score candidate mask
  box_nms            - greedy NMS on window boxes (IoU>=thr), OR the rest
  mask_nms           - greedy NMS on binary masks (IoU>=thr), OR the rest
  prob_max           - per-pixel max of soft maps, threshold 0.5
  prob_weighted      - per-pixel weighted mean (weight = score), threshold 0.5
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

import numpy as np


def rec_bin(rec: dict, H: int, W: int) -> np.ndarray:
    buf = np.zeros((H, W), np.uint8)
    binr = rec.get("bin")
    if binr is not None:
        buf[rec["y0"]:rec["y0"] + rec["ch"], rec["x0"]:rec["x0"] + rec["cw"]] = binr
    return buf


def rec_soft(rec: dict, H: int, W: int) -> np.ndarray:
    buf = np.zeros((H, W), np.float32)
    s = rec.get("soft")
    if s is not None:
        buf[rec["y0"]:rec["y0"] + rec["ch"],
            rec["x0"]:rec["x0"] + rec["cw"]] = s.astype(np.float32) / 255.0
    return buf


def box_iou(a: tuple, b: tuple) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua


def greedy_nms(cands: list[dict], mode: str, iou_thr: float,
               H: int, W: int) -> list[dict]:
    order = sorted(range(len(cands)), key=lambda i: cands[i].get("score", 0.0),
                   reverse=True)
    accepted: list[dict] = []
    for idx in order:
        c = cands[idx]
        keep = True
        if mode == "box_nms":
            cbox = (c["x0"], c["y0"], c["x0"] + c["cw"], c["y0"] + c["ch"])
            for a in accepted:
                abox = (a["x0"], a["y0"], a["x0"] + a["cw"], a["y0"] + a["ch"])
                if box_iou(cbox, abox) >= iou_thr:
                    keep = False
                    break
        elif mode == "mask_nms":
            cm = rec_bin(c, H, W)
            for a in accepted:
                am = rec_bin(a, H, W)
                inter = int(((cm > 0) & (am > 0)).sum())
                uni = int(((cm > 0) | (am > 0)).sum())
                if uni and inter / uni >= iou_thr:
                    keep = False
                    break
        if keep:
            accepted.append(c)
    return accepted


def fuse_mask(cands: list[dict], H: int, W: int,
              mode: str = "mask_nms", nms_iou: float = 0.5) -> np.ndarray:
    out = np.zeros((H, W), np.uint8)
    if mode == "union":
        for c in cands:
            out |= rec_bin(c, H, W)
        return out
    if mode == "best_score_single":
        if not cands:
            return out
        best = max(cands, key=lambda c: c.get("score", 0.0))
        return rec_bin(best, H, W)
    if mode in ("box_nms", "mask_nms"):
        for c in greedy_nms(cands, mode, nms_iou, H, W):
            out |= rec_bin(c, H, W)
        return out
    if mode == "prob_max":
        soft = np.zeros((H, W), np.float32)
        for c in cands:
            soft = np.maximum(soft, rec_soft(c, H, W))
        return (soft > 0.5).astype(np.uint8)
    if mode == "prob_weighted":
        num = np.zeros((H, W), np.float32)
        den = np.zeros((H, W), np.float32)
        for c in cands:
            w = c.get("score", 0.0)
            if w <= 0:
                continue
            s = rec_soft(c, H, W)
            num += w * s
            den += w * (s > 0)
        soft = num / np.maximum(den, 1e-6)
        return (soft > 0.5).astype(np.uint8)
    raise ValueError(f"unknown fusion mode: {mode}")


MODES = ["union", "best_score_single", "box_nms", "mask_nms",
         "prob_max", "prob_weighted"]
