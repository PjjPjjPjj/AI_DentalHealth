"""DentalAI unified ONNX pipeline: quality -> tooth -> calculus -> caries ->
plaque -> associate -> result dict (+ GUI-friendly summary)."""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from app.core.associate import associate_calculus, associate_caries, mask_geometry
from app.core.models_onnx import (
    CalculusCascadeOnnx,
    CariesModelOnnx,
    PlaqueModelOnnx,
    ToothModelOnnx,
)
from app.core.quality import check_quality

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

DEFAULT_CONFIG = {
    "quality": {
        "blur_laplacian_threshold": 100.0,
        "brightness_underexposed": 60.0,
        "brightness_overexposed": 200.0,
        "dark_pixel_ratio_limit": 0.35,
        "bright_pixel_ratio_limit": 0.35,
        "highlight_ratio_limit": 0.12,
        "min_size": 320,
    },
    "pipeline": {
        "association": {
            "calculus_min_overlap": 0.20,
            "caries_center_rule": True,
            "caries_min_iou": 0.01,
            "unassigned_tag": "unassigned",
        }
    },
    "inference": {
        "tooth": {"imgsz": 896, "conf": 0.30, "iou": 0.70, "max_det": 60},
        "calculus": {"c1": 0.005, "thr_std": 0.60, "thr_large": 0.45,
                     "large_edge": 0.01565, "tau": 0.40, "thr_rescue": 0.25,
                     "s1_conf_min": 0.02, "fb": True, "fb_min_px": 30, "close_k": 5},
        "caries": {"imgsz": 960, "conf": 0.55, "iou": 0.60, "max_det": 100},
        "plaque": {"imgsz": 224, "conf_threshold": 0.746,  # calibrated scale (== raw 0.70)
                   "calibration": [0.4694964, 0.68013026],  # Platt (a,b) from plaque_bin_v2_sano
                   "image_level": {
                       "aggregation": "mean",
                       "threshold_mean": 0.5231,
                       "threshold_max": 0.9517,
                       "threshold_top3": 0.895,
                   }},
    },
}


def load_pipeline_config(path: str | Path | None = None) -> dict:
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_CONFIG.items()}
    if path and Path(path).exists():
        import yaml

        user = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        for section, values in user.items():
            if isinstance(values, dict) and isinstance(cfg.get(section), dict):
                cfg[section].update(values)
            else:
                cfg[section] = values
    return cfg


class DentalPipeline:
    """Self-contained ONNX inference pipeline for the desktop app."""

    def __init__(self, models_dir: str | Path | None = None,
                 config_path: str | Path | None = None,
                 threads: int | None = None, fast_mode: bool = False):
        self.cfg = load_pipeline_config(config_path)
        self.models_dir = Path(models_dir) if models_dir else MODEL_DIR
        self.icfg = self.cfg["inference"]
        self.quality_cfg = self.cfg["quality"]
        self.assoc_cfg = self.cfg["pipeline"]["association"]
        md = self.models_dir
        self.warnings: list[str] = []

        self.tooth = self._make(
            ToothModelOnnx, md / "tooth_seg.onnx", self.icfg["tooth"], threads)
        self.calculus = self._make_calculus(md, threads, fast_mode)
        self.caries = self._make(
            CariesModelOnnx, md / "caries_det.onnx", self.icfg["caries"], threads)
        plaque_icfg = {k: v for k, v in self.icfg["plaque"].items()
                       if k not in ("conf_threshold", "image_level")}
        self.plaque = self._make(
            PlaqueModelOnnx, md / "plaque_cls.onnx", plaque_icfg, threads)

    def _require(self, *paths: Path) -> bool:
        return all(Path(p).exists() for p in paths)

    def _make(self, cls, weights: Path, icfg: dict, threads):
        if not self._require(weights):
            self.warnings.append(f"model missing: {weights.name} (stage disabled)")
            return None
        return cls(weights, **icfg, threads=threads)

    def _make_calculus(self, md: Path, threads, fast_mode: bool = False):
        icfg = self.icfg["calculus"]
        need = [md / "calculus_s1.onnx", md / "calculus_s2.onnx",
                md / "calculus_verifier.onnx", md / "calculus_fusion.npz"]
        if not self._require(*need):
            self.warnings.append("calculus cascade models missing (stage disabled)")
            return None
        imgclf = md / "calculus_imgclf.onnx"
        return CalculusCascadeOnnx(
            s1=need[0], s2=need[1], verifier=need[2],
            imgclf=imgclf if imgclf.exists() else None,
            fusion_npz=need[3], threads=threads, fast_mode=fast_mode, **icfg)

    # ------------------------------------------------------------------
    def run(self, image_path: str | Path, image_id: str | None = None) -> dict:
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(img_path)
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot decode image: {img_path}")
        return self.run_array(img, image_id or img_path.name)

    def run_array(self, img: np.ndarray, image_id: str) -> dict:
        t_start = time.perf_counter()
        h, w = img.shape[:2]
        quality = check_quality(img, self.quality_cfg)
        result: dict = {
            "image_id": str(image_id),
            "image_size": {"width": int(w), "height": int(h)},
            "quality": quality,
            "teeth": [],
            "unassigned_lesions": [],
            "warnings": list(self.warnings),
            "timing": {},
        }
        if not quality["usable"]:
            result["warnings"].append(
                "quality_control_warning: image flagged not usable; results may be unreliable")

        t = time.perf_counter()
        tooth_insts = self.tooth.predict(img) if self.tooth else []
        result["timing"]["tooth"] = round(time.perf_counter() - t, 3)
        tooth_masks = [it["mask"] for it in tooth_insts]

        t = time.perf_counter()
        calc_lesions = self.calculus.predict(img) if self.calculus else []
        result["timing"]["calculus"] = round(time.perf_counter() - t, 3)
        for les in calc_lesions:
            assoc = associate_calculus(
                les["mask"], tooth_masks,
                min_overlap=float(self.assoc_cfg["calculus_min_overlap"]))
            entry = {
                "type": "calculus",
                "present": True,
                "confidence": round(les["conf"], 4),
                "area_px": int(np.count_nonzero(les["mask"] > 0)),
                **mask_geometry(les["mask"]),
            }
            if assoc and assoc.get("assigned"):
                entry["tooth_index"] = int(assoc["tooth"])
                entry["overlap_ratio"] = assoc["overlap_ratio"]
                entry["area_ratio"] = assoc["area_ratio"]
                result.setdefault("calculus_lesions", []).append(entry)
            else:
                result["unassigned_lesions"].append(
                    {**entry, "reason": "below_min_overlap"})

        t = time.perf_counter()
        caries_lesions = self.caries.predict(img) if self.caries else []
        result["timing"]["caries"] = round(time.perf_counter() - t, 3)
        for les in caries_lesions:
            assoc = associate_caries(
                les["bbox"], tooth_masks,
                center_rule=bool(self.assoc_cfg["caries_center_rule"]),
                min_iou=float(self.assoc_cfg["caries_min_iou"]))
            entry = {"type": "caries", "present": True,
                     "confidence": round(les["conf"], 4),
                     "bbox": [round(v, 1) for v in les["bbox"]]}
            if assoc.get("assigned"):
                entry["tooth_index"] = int(assoc["tooth"])
                result.setdefault("caries_lesions", []).append(entry)
            else:
                reason = "below_min_iou" if not assoc.get("center") else "center_unmatched"
                result["unassigned_lesions"].append({**entry, "reason": reason})

        t = time.perf_counter()
        thr_plaque = float(self.icfg["plaque"].get("conf_threshold", 0.92))
        crops, crop_meta = [], []
        for i, inst in enumerate(tooth_insts):
            x1, y1, x2, y2 = [int(v) for v in inst["bbox"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                crops.append(img[y1:y2, x1:x2])
                crop_meta.append(i)
        probs = self.plaque.predict_proba_batch(crops) if (crops and self.plaque) else []
        result["timing"]["plaque"] = round(time.perf_counter() - t, 3)
        prob_by_tooth = dict(zip(crop_meta, probs))

        for i, inst in enumerate(tooth_insts):
            p = prob_by_tooth.get(i)
            result["teeth"].append({
                "tooth_id": i + 1,
                "bbox": [round(float(v), 1) for v in inst["bbox"]],
                "mask": None,
                "plaque": {
                    "score": round(p, 4) if p is not None else 0.50,
                    "present": bool(p is not None and p >= thr_plaque),
                    "confidence": round(p, 4) if p is not None else None,
                },
                "calculus": {"present": False, "confidence": 0.0, "area_ratio": 0.0},
                "caries": {"present": False, "confidence": 0.0},
            })

        # image-level plaque aggregation (screening口径, mean-P recommended)
        scored = [prob_by_tooth[i] for i in crop_meta if prob_by_tooth.get(i) is not None]
        if scored:
            arr = np.array(scored, float)
            s_mean = round(float(arr.mean()), 4)
            s_max = round(float(arr.max()), 4)
            s_top3 = round(float(arr[np.argsort(-arr)[:3]].mean()), 4)
        else:
            s_mean = s_max = s_top3 = None
        imgcfg = self.icfg["plaque"].get("image_level", {})
        agg = str(imgcfg.get("aggregation", "mean"))
        thr_map = {"mean": float(imgcfg.get("threshold_mean", 0.5231)),
                   "max": float(imgcfg.get("threshold_max", 0.9517)),
                   "top3": float(imgcfg.get("threshold_top3", 0.895))}
        img_thr = thr_map.get(agg, thr_map["mean"])
        img_score = {"mean": s_mean, "max": s_max, "top3": s_top3}.get(agg)
        result["plaque_image_level"] = {
            "aggregation": agg,
            "score_mean": s_mean, "score_max": s_max, "score_top3": s_top3,
            "score": img_score, "threshold": img_thr,
            "present": bool(img_score is not None and img_score >= img_thr),
            "n_teeth_scored": len(scored),
        }

        for les in result.get("calculus_lesions", []):
            ti = les.get("tooth_index")
            if ti is not None and 0 <= ti < len(result["teeth"]):
                tt = result["teeth"][ti]
                tt["calculus"] = {"present": True,
                                  "confidence": les["confidence"],
                                  "area_ratio": les.get("area_ratio", 0.0)}
        for les in result.get("caries_lesions", []):
            ti = les.get("tooth_index")
            if ti is not None and 0 <= ti < len(result["teeth"]):
                tt = result["teeth"][ti]
                tt["caries"] = {"present": True, "confidence": les["confidence"]}

        result["timing"]["total"] = round(time.perf_counter() - t_start, 3)
        result["summary"] = summarize(result)
        return result


def summarize(result: dict) -> dict:
    return {
        "tooth_count": len(result.get("teeth", [])),
        "plaque_teeth": int(sum(1 for t in result.get("teeth", [])
                                if t["plaque"]["present"])),
        "calculus_count": len(result.get("calculus_lesions", [])),
        "caries_count": len(result.get("caries_lesions", [])),
        "unassigned_count": len(result.get("unassigned_lesions", [])),
    }
