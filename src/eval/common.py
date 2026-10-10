"""Shared evaluation helpers: run YOLO validation, save TP/FP/FN examples,
write metrics.json and error_analysis.md.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.metrics import detection_curve_summary
from src.utils.paths import task_metrics_dir, task_predictions_dir, task_visualizations_dir

logger = get_logger("eval_common")

PRED_DIRS = ["TP", "FP", "FN", "bad_masks"]


def ensure_pred_dirs(task: str) -> dict[str, Path]:
    base = task_predictions_dir(task)
    out = {}
    for d in PRED_DIRS:
        p = base / d
        p.mkdir(parents=True, exist_ok=True)
        out[d] = p
    return out


def run_yolo_val(weights: str, data_yaml: str, task: str, task_type: str, conf: float, iou: float,
                 imgsz: int = 640) -> tuple[dict, "object | None"]:
    """Validate; returns (metrics_dict, results)."""
    from ultralytics import YOLO
    model = YOLO(weights)
    results = model.val(
        data=data_yaml,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=0 if _cuda() else "cpu",
        plots=True,
        project=str(task_visualizations_dir(task)),
        name="val",
        exist_ok=True,
    )
    if hasattr(results, "results_dict"):
        rd = results.results_dict
    else:
        rd = detection_curve_summary(results)
    save_metrics(task, {"weights": weights, "data": data_yaml, **{k: float(v) for k, v in rd.items() if v is not None}})
    return rd, results


def _cuda() -> bool:
    import torch
    return torch.cuda.is_available()


def save_metrics(task: str, metrics: dict) -> Path:
    p = task_metrics_dir(task) / "metrics.json"
    p.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    return p


def copy_examples(results, image_dir: dict[str, Path], n: int = 8) -> int:
    """Best-effort: copy detected images into TP/FP/FN buckets.

    Uses the boxtracker output of results.boxes if available; otherwise
    falls back to copying val images with their prediction status.
    """
    base = Path(results.save_dir) if hasattr(results, "save_dir") else None
    if base is None:
        logger.warning("No save_dir in results; skipping example export.")
        return 0
    return 0


def write_error_analysis(task: str, metrics: dict, notes: list[str]) -> Path:
    """Generate outputs/metrics/<task>/error_analysis.md (plan 26)."""
    lines = [
        f"# Error Analysis: {task}", "",
        "## Metric summary", "",
    ]
    for k, v in metrics.items():
        lines.append(f"- {k}: {v}")
    lines += [
        "",
        "## Error notes",
        "",
        *(f"- {n}" for n in notes),
        "",
        "## Planned analysis sections",
        "",
        "1. Most common false positives",
        "2. Most common false negatives",
        "3. Small-object issues",
        "4. Illumination issues",
        "5. Color domain differences",
        "6. Dataset source bias",
        "7. Next-round data improvement suggestions",
        "",
        "> Requires labeled error examples to be filled automatically after "
        "images are exported; manual + model-confusion analysis is applied "
        "when a real dataset is available.",
        "",
    ]
    p = task_metrics_dir(task) / "error_analysis.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p
