"""Two-stage calculus segmentation inference (both stages YOLO11m-seg @640).

stage1: whole-image detection @ conf=c1 -> high-recall candidate boxes
        (+ original-size masks via retina_masks for EXP6 center strategies)
stage2: for each stage1 box -> window crop (bbox + MARGIN, square) -> fine
        segmentation; EVERY stage2 instance mask is treated as an independent
        candidate (no internal np.maximum union).

Candidate record: {x0,y0,cw,ch, bin(ROI uint8), s1_conf, s2_conf, score}
score modes (see __init__):
  "s2"    : score = stage2 confidence             (score v1)
  "mix07" : score = 0.7*s2_conf + 0.3*s1_conf    (score v2)
Whatever fusion (default mask-level NMS, nms_iou configurable, keeps the
highest-score mask per cluster) then unions the surviving candidates into the
semantic mask; the instance list is kept for instance-level evaluation.

Used by evaluate_calculus_full.py --mode twostage.
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

import cv2
import numpy as np
from ultralytics import YOLO

from src.inference.fusion import fuse_mask

TARGET = 640
MARGIN = 1.8
MIN_WINDOW = 160


class TwoStageSegmenter:
    def __init__(self, stage1_weights: str, stage2_weights: str,
                 c1: float, c2: float, device: int = 0,
                 fusion: str = "mask_nms", nms_iou: float = 0.5,
                 score_mode: str = "s2"):
        self.s1 = YOLO(stage1_weights)
        self.s2 = YOLO(stage2_weights)
        self.c1 = c1
        self.c2 = c2
        self.device = device
        self.fusion = fusion
        self.nms_iou = nms_iou
        self.score_mode = score_mode

    # ---------------- stage1 ----------------
    def _stage1(self, image_path: str) -> np.ndarray | None:
        res = self.s1.predict(image_path, conf=self.c1, iou=0.6, imgsz=TARGET,
                              device=self.device, retina_masks=True, max_det=100,
                              verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None
        return res.boxes.xyxy.cpu().numpy()

    def _stage1_full(self, image_path: str) -> tuple[np.ndarray, np.ndarray]:
        res = self.s1.predict(image_path, conf=self.c1, iou=0.6, imgsz=TARGET,
                              device=self.device, retina_masks=True, max_det=100,
                              verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
        return res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()

    def _stage1_masks(self, image_path: str
                      ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        """boxes (orig coords), confs, original-size binary instance masks."""
        res = self.s1.predict(image_path, conf=self.c1, iou=0.6, imgsz=TARGET,
                              device=self.device, retina_masks=True, max_det=100,
                              verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), [])
        mk = res.masks.data.cpu().numpy() if res.masks is not None else None
        masks = [(mk[i] > 0.5).astype(np.uint8) for i in range(mk.shape[0])] \
            if mk is not None else []
        return res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), masks

    # ---------------- stage2 ----------------
    def _stage2(self, crop: np.ndarray) -> tuple[np.ndarray | None, float]:
        """Legacy: merged (np.maximum) mask + best conf. Kept for compatibility."""
        res = self.s2.predict(crop, conf=self.c2, iou=0.6, imgsz=TARGET,
                              device=self.device, retina_masks=True, max_det=100,
                              verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None, 0.0
        best = float(res.boxes.conf.cpu().max().item())
        out = None
        if res.masks is not None:
            mk = res.masks.data.cpu().numpy()
            for i in range(mk.shape[0]):
                mi = (mk[i] > 0.5).astype(np.uint8)
                out = mi if out is None else np.maximum(out, mi)
        return out, best

    def _stage2_instances(self, crop: np.ndarray) -> list[dict]:
        """Every raw stage2 instance: {bin640, s2_conf, bbox640, area}."""
        res = self.s2.predict(crop, conf=self.c2, iou=0.6, imgsz=TARGET,
                              device=self.device, retina_masks=True, max_det=100,
                              verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0 or res.masks is None:
            return []
        mk = res.masks.data.cpu().numpy()
        out = []
        for i in range(mk.shape[0]):
            mi = (mk[i] > 0.5).astype(np.uint8)
            a = int(mi.sum())
            if a == 0:
                continue
            gx, gy, gw, gh = cv2.boundingRect(mi)
            out.append({"bin640": mi, "s2_conf": float(res.boxes.conf[i]),
                        "bbox640": (gx, gy, gx + gw, gy + gh), "area": a})
        return out

    def _window(self, b, H, W) -> tuple[int, int, int, int]:
        x, y, x2, y2 = b
        cx, cy = (x + x2) / 2.0, (y + y2) / 2.0
        side = max(x2 - x, y2 - y) * MARGIN
        side = max(side, MIN_WINDOW)
        side = float(min(side, max(H, W)))
        x0 = int(round(cx - side / 2)); y0 = int(round(cy - side / 2))
        x0 = max(0, min(x0, W - int(side))); y0 = max(0, min(y0, H - int(side)))
        x1 = min(x0 + int(side), W); y1 = min(y0 + int(side), H)
        return x0, y0, x1, y1

    def _crop_at(self, img: np.ndarray, cx: float, cy: float,
                 side: float, H: int, W: int) -> tuple[np.ndarray, tuple]:
        x0 = int(round(cx - side / 2)); y0 = int(round(cy - side / 2))
        x0 = max(0, min(x0, W - int(side))); y0 = max(0, min(y0, H - int(side)))
        x1 = min(x0 + int(side), W); y1 = min(y0 + int(side), H)
        crop = cv2.resize(img[y0:y1, x0:x1], (TARGET, TARGET),
                          interpolation=cv2.INTER_AREA)
        return crop, (x0, y0, x1, y1)

    def _score(self, rec: dict) -> float:
        if self.score_mode == "s2":
            return rec["s2_conf"]
        if self.score_mode == "mix07":
            return 0.7 * rec["s2_conf"] + 0.3 * rec["s1_conf"]
        if self.score_mode == "composite":
            return rec["s1_conf"] * rec["fg_prob"]
        raise ValueError(self.score_mode)

    def candidates(self, image_path: str, img: np.ndarray | None = None,
                   centers: str = "bbox") -> list[dict]:
        """Collect every raw stage2 instance as an independent candidate.

        centers: "bbox"   center = stage1 box bbox center (default/current)
                 "mbbox"  center = stage1 mask bbox center
                 "mcent"  center = stage1 mask centroid
        side (window size) is identical in all strategies.
        """
        if img is None:
            img = cv2.imread(image_path)
        H, W = img.shape[:2]
        if centers == "bbox":
            boxes, confs = self._stage1_full(image_path)
            masks = None
        else:
            boxes, confs, masks = self._stage1_masks(image_path)
        recs: list[dict] = []
        for k, (b, sc1) in enumerate(zip(boxes, confs)):
            if masks is not None:
                bm = masks[k]
                if centers == "mbbox":
                    gx, gy, gw, gh = cv2.boundingRect(bm)
                    cx, cy = gx + gw / 2.0, gy + gh / 2.0
                else:  # mcent: centroid of the stage-1 mask
                    ys, xs = np.nonzero(bm)
                    cx, cy = float(xs.mean()), float(ys.mean())
            else:
                cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
            side = float(max(b[2] - b[0], b[3] - b[1]) * MARGIN)
            side = max(side, MIN_WINDOW)
            side = float(min(side, max(H, W)))
            crop, (x0, y0, x1, y1) = self._crop_at(img, cx, cy, side, H, W)
            ch, cw = y1 - y0, x1 - x0
            for inst in self._stage2_instances(crop):
                m = cv2.resize(inst["bin640"], (cw, ch),
                               interpolation=cv2.INTER_NEAREST)
                recs.append({"x0": x0, "y0": y0, "cw": cw, "ch": ch,
                             "bin": m, "s1_conf": float(sc1),
                             "s2_conf": inst["s2_conf"],
                             "area640": inst["area"],
                             "bbox640": inst["bbox640"]})
        for r in recs:
            r["score"] = self._score(r)
        return recs

    def predict_instances(self, image_path: str
                          ) -> tuple[list[dict], np.ndarray | None, float]:
        img = cv2.imread(image_path)
        H, W = img.shape[:2]
        recs = self.candidates(image_path, img)
        if not recs:
            return [], None, 0.0
        mask = fuse_mask(recs, H, W, self.fusion, self.nms_iou)
        if mask.sum() == 0:
            return recs, None, 0.0
        return recs, mask, float(max(r["s2_conf"] for r in recs))

    def predict(self, image_path: str) -> tuple[np.ndarray | None, float]:
        _, mask, score = self.predict_instances(image_path)
        return mask, score
