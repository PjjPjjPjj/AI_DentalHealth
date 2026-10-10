"""Metrics utilities: detection/segmentation/classification aggregation.

YOLO reports are usually parsed from results.json produced by Ultralytics
validation; this module computes the extra metrics V1 requires:
  - classification: sensitivity, specificity, AUROC, per-region stats
  - segmentation: per-instance Dice / IoU against masks (used when a mask
    representation is available)
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import numpy as np
from sklearn import metrics as sk_metrics


def classification_metrics(y_true: list[int], y_prob: list[float], threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    recall = sensitivity
    f1 = 2 * precision * recall / max(1e-9, precision + recall)

    n_pos = int(y_true.sum())
    if n_pos == 0 and len(y_true) - n_pos == 0:
        auroc = float("nan")
    elif n_pos == 0 or n_pos == len(y_true):
        auroc = float("nan")
    else:
        auroc = float(sk_metrics.roc_auc_score(y_true, y_prob))
    acc = (tp + tn) / max(1, len(y_true))
    return {
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auroc": round(auroc, 4) if auroc == auroc else "NA",
        "accuracy": round(acc, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "threshold": threshold,
    }


def dice_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """Binary masks (bool). Returns dice and iou."""
    inter = float(np.logical_and(pred_mask, gt_mask).sum())
    pred_sum = float(pred_mask.sum())
    gt_sum = float(gt_mask.sum())
    union = pred_sum + gt_sum - inter
    dice = 2 * inter / max(1e-9, pred_sum + gt_sum)
    iou = inter / max(1e-9, union)
    return {"dice": round(dice, 4), "iou": round(iou, 4)}


def mask_tp_fp_fn(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple[int, int, int]:
    """Per-class pixel-level counts (useful for error analysis)."""
    inter = int(np.logical_and(pred_mask, gt_mask).sum())
    tp = inter
    fp = int(pred_mask.sum()) - inter
    fn = int(gt_mask.sum()) - inter
    return tp, fp, fn


def detection_curve_summary(results: dict) -> dict:
    """Extract standard YOLO detection/seg metrics from a results dict."""
    out = {}
    keys = [
        "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)",
        "metrics/mAP50-95(B)", "metrics/precision(M)", "metrics/recall(M)",
        "metrics/mAP50(M)", "metrics/mAP50-95(M)",
    ]
    for k in keys:
        if k in results and results[k] is not None:
            out[k] = float(results[k])
    return out
