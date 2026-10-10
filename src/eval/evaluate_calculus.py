"""Evaluate calculus segmentation + export TP/FP/FN and bad masks.

Usage:
    python -m src.eval.evaluate_calculus --weights outputs/checkpoints/calculus/best.pt
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

logger = get_logger("eval_calculus")


def export_seg_examples(weights: str, data_yaml: str, conf: float, iou: float, top: int = 8) -> None:
    from ultralytics import YOLO
    model = YOLO(weights)
    dirs = ensure_pred_dirs("calculus")
    data = resolve_config(data_yaml)
    val_dir = Path(data["path"]) / (data.get("val") or "images/val")
    stamp = 0
    if val_dir.exists():
        for im in sorted(p for p in val_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}):
            try:
                res = model.predict(str(im), conf=conf, iou=iou, verbose=False, save_masks=False)
                n_boxes = len(res[0].boxes) if res and getattr(res[0], "boxes", None) is not None else 0
            except Exception:
                continue
            if n_boxes > 0:
                shutil.copy(im, dirs["TP"] / f"{stamp:04d}_{im.name}")
            else:
                shutil.copy(im, dirs["FN"] / f"{stamp:04d}_{im.name}")
            stamp += 1
            if stamp >= top * 4:
                break
    logger.info("Exported calculus TP/FN examples.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/calculus.yaml")
    parser.add_argument("--weights", default="outputs/checkpoints/calculus/best.pt")
    parser.add_argument("--data", default="configs/calculus_data.yaml")
    parser.add_argument("--export-examples", action="store_true")
    args = parser.parse_args()

    cfg = resolve_config(args.config)
    ecfg = cfg.get("eval", {})
    metrics, _ = run_yolo_val(
        args.weights, args.data, "calculus", "segment",
        conf=ecfg.get("conf_threshold", 0.25), iou=ecfg.get("iou_threshold", 0.5),
    )
    write_error_analysis("calculus", metrics, [
        "Calculus vs extrinsic staining / saliva shadow confusions are expected.",
        "Masks exported from polygon labels; small lesions may be under-segmented at 640.",
    ])
    if args.export_examples:
        export_seg_examples(args.weights, args.data, ecfg.get("conf_threshold", 0.25),
                            ecfg.get("iou_threshold", 0.5))
    logger.info("Done.")


if __name__ == "__main__":
    main()
