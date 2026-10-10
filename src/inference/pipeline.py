"""Unified inference pipeline (plan 18).

image -> quality -> tooth seg -> calculus seg -> caries det -> plaque -> associate -> JSON

Calculus & caries run whole-image once (never per-tooth x per-model fan-out).
Model modules are optional: missing weights raise clear errors unless
skip_missing_models=True (used by smoke tests / BLOCKED_EXTERNAL runs).
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.inference.associate import associate_calculus, associate_caries, compute_plaque_regions
from src.inference.image_quality import check_quality
from src.inference.models import CalculusModel, CariesModel, PlaqueModel, ToothModel
from src.inference.schemas import validate_result
from src.utils.config import deep_merge, load_yaml
from src.utils.logger import get_logger
from src.utils.paths import CONFIGS_DIR

logger = get_logger("pipeline")


def _pts_to_mask(points, w: int, h: int) -> np.ndarray:
    import numpy as np
    import cv2
    pts = np.array([[p[0] * w, p[1] * h] for p in points], dtype=np.float32)
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m, [pts.astype(np.int32)], 1)
    return m


def _mask_geometry(mask: np.ndarray) -> dict:
    """bbox [x1,y1,x2,y2] in px + normalized outer-contour polygons for JSON output."""
    import cv2
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return {"bbox": None, "polygon": []}
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < 30:
            continue
        eps = 0.004 * cv2.arcLength(c, True)
        pts = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        polys.append([[round(float(x) / w, 5), round(float(y) / h, 5)] for x, y in pts])
    return {"bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "polygon": polys}


class OralAIPipeline:
    def __init__(self, cfg_path: str | Path = None, weights: dict | None = None,
                 skip_missing_models: bool = False, device: str = "0"):
        cfg = load_yaml(cfg_path or CONFIGS_DIR / "pipeline.yaml")
        icfg = cfg.get("inference", {})
        qual = cfg.get("quality", {})
        self.assoc_cfg = cfg.get("pipeline", {}).get("association", {})
        self._weight_cfg = deep_merge({k: v for k, v in cfg.get("weights", {}).items()}, weights or {})
        self.skip_missing = skip_missing_models
        self.device = device

        self.quality_cfg = qual
        self.plaque_img_cfg = icfg.get("plaque", {}).get("image_level", {})
        self.tooth = self._make(ToothModel, "tooth", self._weight_cfg.get("tooth"), icfg.get("tooth", {}))
        self.calculus = self._make_calculus(icfg.get("calculus", {}), self._weight_cfg.get("calculus"))
        self.caries = self._make(CariesModel, "caries", self._weight_cfg.get("caries"), icfg.get("caries", {}))
        self.plaque = self._make(PlaqueModel, "plaque", self._weight_cfg.get("plaque"),
                                 {k: v for k, v in icfg.get("plaque", {}).items() if k != "image_level"})

    def _make_calculus(self, icfg, w) -> Any | None:
        from src.inference.models import CalculusCascadeModel
        if not isinstance(w, dict) \
                or not all(Path(w.get(k, "")).exists() for k in ("s1", "s2")) \
                or not Path(w.get("artifacts", "")).exists():
            if not self.skip_missing:
                raise FileNotFoundError(f"[calculus] cascade weights missing: {w}")
            logger.warning("[calculus] weights missing - model disabled (skip_missing_models=True)")
            return None
        return CalculusCascadeModel(**icfg, **w)

    def _make(self, cls, name, weights, icfg) -> Any | None:
        if weights is None:
            logger.warning(f"[{name}] no weights configured - model disabled")
            return None
        if not Path(weights).exists():
            if not self.skip_missing:
                raise FileNotFoundError(f"[{name}] weights missing: {weights} (use skip_missing_models=True or train first)")  # noqa: E501
            logger.warning(f"[{name}] weights missing - model disabled (skip_missing_models=True)")
            return None
        return cls(weights=weights, **icfg)

    def run(self, image_path: str | Path, image_id: str | None = None) -> dict:
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(img_path)
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot decode image: {img_path}")
        result = self.run_array(img, image_id or img_path.name)
        return result

    def run_array(self, img: np.ndarray, image_id: str) -> dict:
        h, w = img.shape[:2]
        quality = check_quality(img, self.quality_cfg)
        result: dict[str, Any] = {
            "image_id": str(image_id),
            "image_size": {"width": int(w), "height": int(h)},
            "quality": {
                "usable": bool(quality["usable"]),
                "blur_score": quality["blur_score"],
                "brightness": quality["brightness"],
                "highlight_ratio": quality["highlight_ratio"],
                "underexposed": bool(quality.get("underexposed")),
                "overexposed": bool(quality.get("overexposed")),
                "blurry": bool(quality.get("blurry")),
                "highlight_warning": bool(quality.get("highlight_warning")),
            },
            "teeth": [],
            "unassigned_lesions": [],
            "warnings": [],
        }

        if not quality["usable"]:
            result["warnings"].append("quality_control_warning: image flagged not usable; results may be unreliable")  # noqa: E501

        # 1. tooth instances
        tooth_insts = self.tooth.predict(img) if self.tooth else []
        tooth_masks = [t["mask"] for t in tooth_insts]

        # 2. calculus (whole image once)
        calc_lesions = self.calculus.predict(img) if self.calculus else []
        for les in calc_lesions:
            assoc = associate_calculus(les["mask"], tooth_masks,
                                       min_overlap=float(self.assoc_cfg.get("calculus_min_overlap", 0.20)))
            entry = {
                "present": True,
                "confidence": round(les["conf"], 4),
                "area_px": int(np.count_nonzero(les["mask"] > 0)),
                **_mask_geometry(les["mask"]),
            }
            if assoc and assoc.get("assigned"):
                entry["tooth_index"] = int(assoc["tooth"])
                entry["overlap_ratio"] = assoc["overlap_ratio"]
                entry["area_ratio"] = assoc["area_ratio"]
                result.setdefault("calculus_lesions", []).append(entry)
            else:
                result["unassigned_lesions"].append({"type": "calculus", **entry,
                                                     "reason": "below_min_overlap"})

        # 3. caries (whole image once)
        caries_lesions = self.caries.predict(img) if self.caries else []
        for les in caries_lesions:
            assoc = associate_caries(
                les["bbox"], tooth_masks,
                center_rule=bool(self.assoc_cfg.get("caries_center_rule", True)),
                min_iou=float(self.assoc_cfg.get("caries_min_iou", 0.01)),
            )
            entry = {"present": True, "confidence": round(les["conf"], 4), "bbox": [round(v, 1) for v in les["bbox"]]}
            if assoc.get("assigned"):
                entry["tooth_index"] = int(assoc["tooth"])
                result.setdefault("caries_lesions", []).append(entry)
            else:
                reason = "below_min_iou" if not assoc.get("center") else "center_unmatched"
                result["unassigned_lesions"].append({"type": "caries", **entry, "reason": reason})

        # 4. plaque: whole-tooth crop probability (G/I/M/D reserved per plan)
        multi_window = len(getattr(self.plaque, "windows", [[0, 0, 1, 1]])) > 1
        tooth_plaque_probs: list[float] = []
        for i, inst in enumerate(tooth_insts):
            x1, y1, x2, y2 = [int(v) for v in inst["bbox"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = img[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else None
            plaque_prob = None
            if crop is not None and self.plaque:
                try:
                    if multi_window:
                        plaque_prob = round(self.plaque.predict_proba_windows(crop), 4)
                    else:
                        plaque_prob = round(self.plaque.predict_proba(crop), 4)
                except Exception as exc:
                    logger.warning(f"plaque prediction failed on tooth {i}: {exc}")
            regions = compute_plaque_regions(inst["mask"])
            if plaque_prob is not None:
                tooth_plaque_probs.append(plaque_prob)
            result["teeth"].append({
                "tooth_id": i + 1,
                "bbox": [round(float(v), 1) for v in inst["bbox"]],
                "mask": None,
                "plaque": {
                    **regions,
                    "score": plaque_prob if plaque_prob is not None else 0.50,
                    "present": bool(plaque_prob is not None and
                                    plaque_prob >= float(getattr(self.plaque, "threshold", 0.92))),
                    "confidence": plaque_prob if plaque_prob is not None else None,
                    "region_score": {
                        "gingival": None, "incisal": None, "mesial": None, "distal": None,
                        "note": "G/I/M/D region geometry not implemented in V1 (plan 19); score = tooth-crop plaque probability"  # noqa: E501
                    },
                },
                "calculus": {
                    "present": False, "confidence": 0.0, "area_ratio": 0.0,
                },
                "caries": {"present": False, "confidence": 0.0},
            })

        # 4b. image-level plaque aggregation (screening口径, mean-P recommended)
        if tooth_plaque_probs:
            arr = np.array(tooth_plaque_probs, float)
            score_mean = round(float(arr.mean()), 4)
            score_max = round(float(arr.max()), 4)
            top3 = arr[np.argsort(-arr)[:3]]
            score_top3 = round(float(top3.mean()), 4)
        else:
            score_mean = score_max = score_top3 = None
        agg = str(getattr(self, "plaque_img_cfg", {}).get("aggregation", "mean"))
        pic = getattr(self, "plaque_img_cfg", {})
        thr_map = {"mean": float(pic.get("threshold_mean", 0.5231)),
                   "max": float(pic.get("threshold_max", 0.9517)),
                   "top3": float(pic.get("threshold_top3", 0.895))}
        img_thr = thr_map.get(agg, thr_map["mean"])
        img_score = {"mean": score_mean, "max": score_max, "top3": score_top3}.get(agg)
        result["plaque_image_level"] = {
            "aggregation": agg,
            "score_mean": score_mean,
            "score_max": score_max,
            "score_top3": score_top3,
            "score": img_score,
            "threshold": img_thr,
            "present": bool(img_score is not None and img_score >= img_thr),
            "n_teeth_scored": len(tooth_plaque_probs),
            "note": "image-level screening; thr = val positive P5 (Sens>=0.95 rule)",
        }

        # 5. merge lesion info into teeth objects
        for les in result.get("calculus_lesions", []):
            ti = les.get("tooth_index")
            if ti is not None and 0 <= ti < len(result["teeth"]):
                t = result["teeth"][ti]
                t["calculus"]["present"] = True
                t["calculus"]["confidence"] = les["confidence"]
                t["calculus"]["area_ratio"] = les.get("area_ratio", 0.0)
        for les in result.get("caries_lesions", []):
            ti = les.get("tooth_index")
            if ti is not None and 0 <= ti < len(result["teeth"]):
                t = result["teeth"][ti]
                t["caries"]["present"] = True
                t["caries"]["confidence"] = les["confidence"]

        violations = validate_result(result)
        if violations:
            logger.warning(f"Schema violations: {violations}")
        return result
