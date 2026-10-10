"""Evaluate caries detection baseline + export predictions.

Usage:
    python -m src.eval.evaluate_caries --weights outputs/checkpoints/caries/best.pt
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2

from src.eval.common import ensure_pred_dirs, run_yolo_val, write_error_analysis
from src.utils.config import resolve_config
from src.utils.logger import get_logger
from src.utils.paths import task_predictions_dir

logger = get_logger("eval_caries")


def export_caries_examples(weights: str, data_yaml: str, conf: float, iou: float, top: int = 8) -> None:
    """Run validation with plots and copy TP/FP/FN images from val images dir.

    With plots=True Ultralytics writes a val_batch*.jpg; we additionally run
    per-image inference on the val split to bucket images with/without hits.
    """
    from ultralytics import YOLO
    model = YOLO(weights)
    dirs = ensure_pred_dirs("caries")
    data = resolve_config(data_yaml)
    val_dir = Path(data["path"]) / "train" if data.get("train") else None
    val_dir = Path(data["path"]) / (data.get("val") or "images/val")
    stamp = 0
    if val_dir.exists():
        imgs = sorted(p for p in val_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for im in imgs:
            try:
                img = cv2.imread(str(im))
                res = model.predict(img, conf=conf, iou=iou, verbose=False)
                n_boxes = len(res[0].boxes) if res and res[0].boxes is not None else 0
            except Exception:
                continue
            if n_boxes > 0:
                shutil.copy(im, dirs["TP"] / f"{stamp:04d}_{im.name}")
            else:
                shutil.copy(im, dirs["FN"] / f"{stamp:04d}_{im.name}")
            stamp += 1
            if stamp >= top * 4:
                break
    logger.info(f"Exported TP/FP/FN style examples to {task_predictions_dir('caries')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/caries.yaml")
    parser.add_argument("--weights", default="outputs/checkpoints/caries/best.pt")
    parser.add_argument("--data", default="configs/caries_data.yaml")
    parser.add_argument("--export-examples", action="store_true")
    args = parser.parse_args()

    cfg = resolve_config(args.config)
    ecfg = cfg.get("eval", {})
    metrics, _ = run_yolo_val(
        args.weights, args.data, "caries", "detect",
        conf=ecfg.get("conf_threshold", 0.25), iou=ecfg.get("iou_threshold", 0.7),
    )
    write_error_analysis("caries", metrics, [
        "Caries ~ fissure staining confusion is expected; hard negatives are planned separately.",  # noqa: E501
        "Small cavity boxes may be missed at imgsz=640; imgsz=960/1024 as separate experiments.",  # noqa: E501
    ])
    if args.export_examples:
        export_caries_examples(args.weights, args.data, ecfg.get("conf_threshold", 0.25),
                               ecfg.get("iou_threshold", 0.7))
    logger.info(f"Done. metrics -> {task_predictions_dir('caries').parent}/metrics.json")


if __name__ == "__main__":
    main()
