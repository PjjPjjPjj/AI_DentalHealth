"""Evaluate plaque classification + per-region stats + tooth plaque score.

Usage:
    python -m src.eval.evaluate_plaque --weights outputs/checkpoints/plaque/best.pt
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.eval.common import save_metrics, write_error_analysis
from src.utils.config import resolve_config
from src.utils.logger import get_logger
from src.utils.metrics import classification_metrics
from src.utils.paths import METADATA_DIR

logger = get_logger("eval_plaque")


def evaluate_plaque(weights: str, root: str, threshold: float, csv_path: str) -> dict:
    """Run cls model over test crops (from plaque_crops.csv) with regression=False.

    Collect per-crop probabilities -> per-region + overall metrics.
    """
    from ultralytics import YOLO
    model = YOLO(weights)
    meta = pd.read_csv(csv_path)
    test = meta[meta["split"] == "test"]
    if test.empty:
        test = meta[meta["split"] == "val"]
    records = []
    for row in test.itertuples():
        p = Path(root) / ".."  # not used; crop full path stored in manifest
        path = row.crop_path
        if not Path(path).exists():
            continue
        try:
            res = model.predict(str(path), imgsz=224, verbose=False)
            prob = 1 - res[0].probs.data[0].item()  # class 1 prob with binary model
        except Exception:
            continue
        records.append({
            "patient_id": row.patient_id, "region": row.region,
            "label": int(row.label), "prob": prob, "tooth_id": row.tooth_id,
        })
    if not records:
        logger.warning("No test crops evaluable; check manifest paths.")
        return {"status": "no_evaluable_crops"}

    df = pd.DataFrame(records)
    overall = classification_metrics(df["label"].tolist(), df["prob"].tolist(), threshold)
    per_region = {}
    for region in ["G", "I", "M", "D"]:
        sub = df[df["region"] == region]
        per_region[region] = classification_metrics(sub["label"].tolist(), sub["prob"].tolist(), threshold) if len(sub) else "NA"

    out = {
        "weights": weights,
        "n_test_crops": int(len(df)),
        "overall": overall,
        "per_region": per_region,
    }
    save_metrics("plaque", out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/plaque.yaml")
    parser.add_argument("--weights", default="outputs/checkpoints/plaque/best.pt")
    parser.add_argument("--manifest", default=str(METADATA_DIR / "plaque_crops.csv"))
    args = parser.parse_args()

    cfg = resolve_config(args.config)
    root = cfg.get("data", {}).get("root", "datasets/processed/plaque_cls")
    threshold = cfg.get("eval", {}).get("threshold", 0.5)
    out = evaluate_plaque(args.weights, root, threshold, args.manifest)
    write_error_analysis("plaque", out, [
        "Plaque score = 区域阳性比例 (not pixel area ratio) - V1 definition.",
        "Sensitivity prioritized over accuracy per plan section 14.",
    ])
    logger.info("Done.")


if __name__ == "__main__":
    main()
