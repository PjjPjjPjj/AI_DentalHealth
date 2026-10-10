"""Complete calculus YOLO11s-seg evaluation per the Oral-Mamba task spec.

Strict rules:
  - split test never used for tuning (thresholds chosen on VAL only)
  - final conf threshold frozen, then single test pass
  - main metrics: Dice / IoU / Precision / Recall(F1) / mask mAP50 / mAP50-95
  - plus image-level metrics, lesion-size analysis, bootstrap 95% CI
  - error analysis with visual exports

Usage:
  python -m src.eval.evaluate_calculus_full --stage val --weights outputs/calculus_yolo11s/train/weights/best.pt
  python -m src.eval.evaluate_calculus_full --stage test --weights outputs/calculus_yolo11s/train/weights/best.pt
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

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.utils.paths import PROCESSED_DIR, REPORTS_DIR

logger = get_logger("eval_calculus_full")

DATA_DIR = PROCESSED_DIR / "calculus_seg"
RUN_DIR = Path("outputs/calculus_yolo11s")
_dynamic_run_dir: list = []


def get_run_dir() -> Path:
    if _dynamic_run_dir:
        return _dynamic_run_dir[0]
    return RUN_DIR


def _set_run_dir(p: Path) -> None:
    if _dynamic_run_dir:
        _dynamic_run_dir[0] = p
    else:
        _dynamic_run_dir.append(p)


def predict_split(model, split: str, conf: float) -> list[dict]:
    """Predict every image in split; returns per-image records."""
    from ultralytics import YOLO
    model = model or YOLO(str(RUN_DIR / "train" / "weights" / "best.pt"))
    man = pd.read_csv(DATA_DIR / "split_manifest.csv")
    rows = man[man.split == split]
    recs = []
    for _, r in rows.iterrows():
        img_id = r["image_id"]
        img = cv2.imread(r["image_path"])
        h, w = img.shape[:2]
        gt = cv2.imread(r["mask_original_path"], cv2.IMREAD_UNCHANGED)
        gt = ((gt > 0).astype(np.uint8)) if gt is not None else np.zeros((h, w), np.uint8)
        res = model.predict(
            r["image_path"], conf=conf, iou=0.5, imgsz=640, device=0,
            retina_masks=True, max_det=100, verbose=False)
        res = res[0]
        pred = np.zeros((h, w), np.uint8)
        confs: list[float] = []
        if res.masks is not None and len(res.masks) > 0:
            mk = res.masks.data.cpu().numpy()
            for i in range(mk.shape[0]):
                if res.boxes.conf[i].item() < conf:
                    continue
                confs.append(float(res.boxes.conf[i].item()))
                mi = cv2.resize((mk[i] > 0.5).astype(np.uint8) * 255, (w, h),
                                interpolation=cv2.INTER_NEAREST)
                pred[mi > 0] = 1
        if len(confs) and res.boxes is not None:
            max_conf = float(res.boxes.conf.cpu().max().item())
        else:
            max_conf = 0.0
        inter = int(((pred > 0) & (gt > 0)).sum())
        union = int(((pred > 0) | (gt > 0)).sum())
        tp = inter
        fp = int((pred > 0).sum()) - tp
        fn = int((gt > 0).sum()) - tp
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else (1.0 if tp == 0 and fp == 0 else 0.0)
        iou = tp / union if union else (1.0 if fp == 0 else 0.0)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        recs.append({
            "image_id": img_id, "split": split, "image_path": r["image_path"],
            "mask_path": r["mask_original_path"], "has_calculus": bool(r["has_calculus"]),
            "gt_px": int((gt > 0).sum()), "pred_px": int((pred > 0).sum()),
            "gt_ratio": (gt > 0).sum() / (h * w),
            "n_pred_instances": len(confs), "max_conf": max_conf,
            "dice": dice, "iou": iou, "precision": prec, "recall": rec,
            "tp": tp, "fp": fp, "fn": fn,
        })
    return recs



from src.inference.twostage import TwoStageSegmenter



def predict_split_twostage(s1, s2, split: str, c1: float, c2: float) -> list[dict]:
    """Two-stage predict: stage1 whole-image recall @c1, stage2 crop fine mask @c2."""
    import cv2 as _cv2
    man = pd.read_csv(DATA_DIR / "split_manifest.csv")
    rows = man[man.split == split]
    seg = TwoStageSegmenter(str(s1), str(s2), c1=c1, c2=c2, device=0)
    recs = []
    for _, r in rows.iterrows():
        img = _cv2.imread(r["image_path"])
        h, w = img.shape[:2]
        gt = _cv2.imread(r["mask_original_path"], _cv2.IMREAD_UNCHANGED)
        gt = (gt > 0).astype(np.uint8) if gt is not None else np.zeros((h, w), np.uint8)
        pred, score = seg.predict(r["image_path"])
        pred = pred if pred is not None else np.zeros((h, w), np.uint8)
        inter = int(((pred > 0) & (gt > 0)).sum())
        union = int(((pred > 0) | (gt > 0)).sum())
        tp = inter; fp = int((pred > 0).sum()) - tp; fn = int((gt > 0).sum()) - tp
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else (1.0 if fp == 0 else 0.0)
        iou = tp / union if union else (1.0 if fp == 0 else 0.0)
        recs.append({
            "image_id": r["image_id"], "split": split, "image_path": r["image_path"],
            "mask_path": r["mask_original_path"], "has_calculus": bool(r["has_calculus"]),
            "gt_px": int((gt > 0).sum()), "pred_px": int((pred > 0).sum()),
            "gt_ratio": (gt > 0).sum() / (h * w),
            "n_pred_instances": 1 if (pred > 0).any() and score >= c2 else 0,
            "max_conf": float(score), "dice": dice, "iou": iou,
            "precision": (tp / (tp + fp)) if (tp + fp) else 0.0,
            "recall": (tp / (tp + fn)) if (tp + fn) else 0.0,
            "tp": tp, "fp": fp, "fn": fn,
        })
    return recs

def image_level_metrics(recs: list[dict]) -> dict:
    gt = np.array([r["has_calculus"] for r in recs])
    sc = np.array([r["max_conf"] for r in recs])
    return {"y_true": gt, "y_score": sc}


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s)  # ascending score; highest score gets rank n
    y_sorted = y[order]
    ranks = np.arange(1, len(y) + 1)
    pos_ranks = ranks[y_sorted == 1]
    return float((pos_ranks.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def thresh_metrics(recs: list[dict], conf: float) -> dict:
    y = np.array([r["has_calculus"] for r in recs])
    p = np.array([r["max_conf"] >= conf for r in recs])
    tp = int(((y == 1) & p).sum()); fn = int(((y == 1) & ~p).sum())
    fp = int(((y == 0) & p).sum()); tn = int(((y == 0) & ~p).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    acc = (tp + tn) / len(y)
    return {"t": conf, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "sensitivity": sens, "specificity": spec,
            "f1": f1, "accuracy": acc}


def pixel_agg(recs: list[dict], positives_only: bool = False) -> dict:
    sel = [r for r in recs if (r["has_calculus"] or not positives_only)]
    n = len(sel)
    return {
        "mean_dice": float(np.mean([r["dice"] for r in sel])) if n else None,
        "median_dice": float(np.median([r["dice"] for r in sel])) if n else None,
        "std_dice": float(np.std([r["dice"] for r in sel])) if n else None,
        "mean_iou": float(np.mean([r["iou"] for r in sel])) if n else None,
        "median_iou": float(np.median([r["iou"] for r in sel])) if n else None,
        "std_iou": float(np.std([r["iou"] for r in sel])) if n else None,
        "n": n,
    }


def bootstrap_ci(recs: list[dict], metric_fn, n_resamples: int = 1000, seed: int = 42) -> dict:
    rng = random.Random(seed)
    vals = [metric_fn(r) for r in recs]
    samples = np.array(vals)
    n = len(samples)
    means = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        means.append(float(np.mean(samples[idx])))
    return {
        "point": float(np.mean(samples)),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["val", "test"], required=True)
    ap.add_argument("--weights", default=str(get_run_dir() / "train" / "weights" / "best.pt"))
    ap.add_argument("--n-resamples", type=int, default=1000)
    ap.add_argument("--num-viz", type=int, default=50)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--mode", choices=["single", "twostage"], default="single")
    ap.add_argument("--stage1-weights", default=None)
    ap.add_argument("--stage2-weights", default=None)
    ap.add_argument("--c1-grid", default="0.05,0.10,0.15,0.20,0.30")
    args = ap.parse_args()
    if args.run_dir:
        _set_run_dir(Path(args.run_dir))

    from ultralytics import YOLO
    if args.mode == "twostage":
        model = None
    else:
        model = YOLO(args.weights)
    run_dir = get_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "val":
        if args.mode == "twostage":
            assert args.stage1_weights and args.stage2_weights, "stage weights required in twostage mode"
            grid_c1 = [float(x) for x in args.c1_grid.split(",")]
            best_c = None
            for c1 in grid_c1:
                recs_c = predict_split_twostage(args.stage1_weights, args.stage2_weights, "val", c1, 0.02)
                _set_run_dir(run_dir)
                grid = [thresh_metrics(recs_c, c) for c in np.arange(0.01, 1.0, 0.02)]
                b1 = max(grid, key=lambda g: g["f1"])
                logger.info(f"twostage c1={c1:.2f}: bestF1={b1['f1']:.4f} @c2={b1['t']:.2f} "
                            f"(validated on VAL)")
                if best_c is None or b1["f1"] > best_c[2]["f1"]:
                    best_c = (c1, b1["t"], b1)
            c1_best, c2_best, best = best_c[0], best_c[1], best_c[2]
            auc_best = roc_auc(image_level_metrics(recs_c)["y_true"],
                               image_level_metrics(recs_c)["y_score"])
            thr = {"selected_conf": round(c2_best, 2),
                   "selection_criteria": "max F1 on validation set (two-stage)",
                   "val_auc": round(float(auc_best), 4),
                   "val_metrics_at_selected": best,
                   "stage1_conf": c1_best, "mode": "twostage",
                   "grid_c1": grid_c1,
                   "note": "test NEVER touched for threshold selection"}
            (run_dir / "thresholds.json").write_text(json.dumps(thr, indent=2))
            logger.info(f"val twostage done. AUC={auc_best:.4f} F1={best['f1']:.4f} "
                        f"c1={c1_best:.2f} c2={c2_best:.2f}")
            return
        recs = predict_split(model, "val", conf=0.001)
        auc = roc_auc(image_level_metrics(recs)["y_true"], image_level_metrics(recs)["y_score"])
        grid = [thresh_metrics(recs, c) for c in np.arange(0.01, 1.0, 0.02)]
        best = max(grid, key=lambda g: g["f1"])
        alt = max(grid, key=lambda g: g["sensitivity"])
        thr = {"selected_conf": round(best["t"], 2),
               "selection_criteria": "max F1 on validation set",
               "val_auc": round(float(auc), 4),
               "val_metrics_at_selected": best,
               "val_metrics_high_sens": alt,
               "grid": grid,
               "note": "test NEVER touched for threshold selection"}
        (run_dir / "thresholds.json").write_text(json.dumps(thr, indent=2))
        pd.DataFrame(recs).to_csv(run_dir / "_val_predictions.csv", index=False)
        logger.info(f"val done. AUC={auc:.4f} bestF1={best['f1']:.4f} @conf={best['t']:.2f}")
        return

    # ---- test stage (frozen) ----
    thr = json.loads((run_dir / "thresholds.json").read_text())
    conf = float(thr["selected_conf"])
    logger.info(f"frozen conf={conf}")
    if args.mode == "twostage":
        c1 = float(thr.get("stage1_conf", 0.15))
        assert args.stage1_weights and args.stage2_weights, "stage weights required in twostage mode"
        recs = predict_split_twostage(args.stage1_weights, args.stage2_weights, "test", c1, conf)
    else:
        recs = predict_split(model, "test", conf=conf)
    pd.DataFrame(recs).to_csv(run_dir / "test_predictions.csv", index=False)

    y = np.array([r["has_calculus"] for r in recs])
    sc = np.array([r["max_conf"] for r in recs])
    imgs = thresh_metrics(recs, conf)
    auc = roc_auc(y, sc)

    pos_dice = pixel_agg([r for r in recs if r["has_calculus"]], positives_only=True)
    all_dice = pixel_agg(recs)
    imrecs = np.array(recs)

    ci_rows = []
    for name, fn in [
        ("Dice", lambda r: r["dice"]),
        ("IoU", lambda r: r["iou"]),
    ]:
        ci = bootstrap_ci(recs, fn, args.n_resamples, 42)
        ci_rows.append((name, ci))
    for name, mask_ in [("Image-level Sensitivity", np.array([r["has_calculus"] for r in recs])),
                       ]:
        pass
    # image-level CIs
    rng = random.Random(42)
    n = len(recs)
    sens_vals, spec_vals, f1_vals = [], [], []
    for _ in range(args.n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        yy = y[idx]; pp = np.array([recs[i]["max_conf"] >= conf for i in idx])
        tp = int(((yy == 1) & pp).sum()); fn = int(((yy == 1) & ~pp).sum())
        fp = int(((yy == 0) & pp).sum()); tn = int(((yy == 0) & ~pp).sum())
        sens_vals.append(tp / (tp + fn) if (tp + fn) else 0)
        spec_vals.append(tn / (tn + fp) if (tn + fp) else 0)
        prec = tp / (tp + fp) if (tp + fp) else 0
        f1_vals.append(2 * prec * sens_vals[-1] / (prec + sens_vals[-1]) if (prec + sens_vals[-1]) else 0)
    def pct(a): return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))
    row_imgs = [
        ("Image-level Sensitivity", float(imgs["sensitivity"]), *pct(sens_vals)),
        ("Image-level Specificity", float(imgs["specificity"]), *pct(spec_vals)),
        ("F1", float(imgs["f1"]), *pct(f1_vals)),
    ]

    # instance mAP via ultralytics val on test split (single-stage only;
    # two-stage instance mAP computed separately in compute_twostage_map.py)
    map50 = map5095 = 0.0
    if args.mode != "twostage":
        from ultralytics import YOLO
        vm = YOLO(args.weights)
        vres = vm.val(data=str(DATA_DIR / "calculus.yaml"), split="test", imgsz=640,
                      conf=0.001, iou=0.5, device=0, plots=True,
                      project=str(get_run_dir() / "val_test"), name="results", exist_ok=True)
        rd = vres.results_dict
        map50 = float(rd.get("metrics/mAP50(M)", 0.0))
        map5095 = float(rd.get("metrics/mAP50-95(M)", 0.0))

    # ---- lesion size analysis (per prompt: GT area ratio to image, tercile buckets) ----
    ratio_gt = np.array([r["gt_ratio"] for r in recs if r["has_calculus"] and r["gt_px"] > 0])
    quants = np.quantile(ratio_gt, [1 / 3, 2 / 3])
    size_rows = []
    for name, lo, hi in [("small", 0, quants[0]), ("medium", quants[0], quants[1]),
                         ("large", quants[1], np.inf)]:
        sel = [r for r in recs if r["has_calculus"] and lo <= r["gt_ratio"] < hi]
        if not sel:
            continue
        dice = np.mean([r["dice"] for r in sel])
        iou = np.mean([r["iou"] for r in sel])
        det_rec = np.mean([r["max_conf"] >= conf for r in sel])
        pix_rec = np.sum([r["tp"] for r in sel]) / max(np.sum([r["tp"] for r in sel]) + np.sum([r["fn"] for r in sel]), 1)
        size_rows.append({"size_bucket": name, "n": len(sel),
                          "gt_ratio_range": [round(float(lo), 5), round(float(hi), 5)],
                          "dice": round(float(dice), 4), "iou": round(float(iou), 4),
                          "detection_recall": round(float(det_rec), 4),
                          "pixel_recall": round(float(pix_rec), 4)})
    pd.DataFrame(size_rows).to_csv(run_dir / "lesion_size_analysis.csv", index=False)

    # ---- bootstrap CI table for key metrics ----
    final_metrics = pd.DataFrame([
        {"Metric": "Precision", "Value": None, "CI95": None},
        {"Metric": "Recall/Sensitivity", "Value": None, "CI95": None},
        {"Metric": "F1", "Value": None, "CI95": None},
        {"Metric": "Dice", "Value": ci_rows[0][1]["point"],
         "CI95": f"{ci_rows[0][1]['ci95_low']:.3f}-{ci_rows[0][1]['ci95_high']:.3f}"},
        {"Metric": "IoU", "Value": ci_rows[1][1]["point"],
         "CI95": f"{ci_rows[1][1]['ci95_low']:.3f}-{ci_rows[1][1]['ci95_high']:.3f}"},
        {"Metric": "Mask mAP50", "Value": map50, "CI95": ""},
        {"Metric": "Mask mAP50-95", "Value": map5095, "CI95": ""},
    ])
    # pixel-level overall precision/recall on positive images (pixel-wise)
    tp_tot = int(sum(r["tp"] for r in recs)); fp_tot = int(sum(r["fp"] for r in recs))
    fn_tot = int(sum(r["fn"] for r in recs))
    pix_prec = tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) else 0
    pix_rec = tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) else 0
    pix_f1 = 2 * pix_prec * pix_rec / (pix_prec + pix_rec) if (pix_prec + pix_rec) else 0

    finals = pd.DataFrame([
        {"Metric": "Precision (pixel)", "Value": round(pix_prec, 4), "CI95": ""},
        {"Metric": "Recall/Sensitivity (pixel)", "Value": round(pix_rec, 4), "CI95": ""},
        {"Metric": "F1 (pixel)", "Value": round(pix_f1, 4), "CI95": ""},
        {"Metric": "Dice", "Value": round(ci_rows[0][1]["point"], 4),
         "CI95": f"{ci_rows[0][1]['ci95_low']:.3f}–{ci_rows[0][1]['ci95_high']:.3f}"},
        {"Metric": "IoU", "Value": round(ci_rows[1][1]["point"], 4),
         "CI95": f"{ci_rows[1][1]['ci95_low']:.3f}–{ci_rows[1][1]['ci95_high']:.3f}"},
        {"Metric": "Dice (positive images)", "Value": round(pos_dice["mean_dice"], 4), "CI95": ""},
        {"Metric": "IoU (positive images)", "Value": round(pos_dice["mean_iou"], 4), "CI95": ""},
        {"Metric": "Mask mAP50", "Value": round(map50, 4), "CI95": ""},
        {"Metric": "Mask mAP50-95", "Value": round(map5095, 4), "CI95": ""},
        {"Metric": "Image-level Sensitivity", "Value": round(imgs["sensitivity"], 4),
         "CI95": f"{row_imgs[0][2]:.3f}–{row_imgs[0][3]:.3f}"},
        {"Metric": "Image-level Specificity", "Value": round(imgs["specificity"], 4),
         "CI95": f"{row_imgs[1][2]:.3f}–{row_imgs[1][3]:.3f}"},
        {"Metric": "Image-level Precision", "Value": round(imgs["precision"], 4), "CI95": ""},
        {"Metric": "Image-level F1", "Value": round(imgs["f1"], 4),
         "CI95": f"{row_imgs[2][2]:.3f}–{row_imgs[2][3]:.3f}"},
        {"Metric": "Image-level Accuracy", "Value": round(imgs["accuracy"], 4), "CI95": ""},
        {"Metric": "Image-level AUROC", "Value": round(auc, 4), "CI95": ""},
    ])
    finals.to_csv(run_dir / "final_metrics.csv", index=False)
    (run_dir / "final_metrics.json").write_text(json.dumps({
        "frozen_conf": conf, "test_n": len(recs),
        "pixel": {"precision": pix_prec, "recall": pix_rec, "f1": pix_f1},
        "dice_all": all_dice, "dice_positive": pos_dice,
        "mask_mAP50": map50, "mask_mAP50_95": map5095,
        "image_level": {k: round(float(imgs[k]), 4) for k in
                        ["precision", "sensitivity", "specificity", "f1", "accuracy"]},
        "auroc": round(float(auc), 4),
        "bootstrap_dice_ci": ci_rows[0][1],
        "bootstrap_iou_ci": ci_rows[1][1],
        "bootstrap_image_cis": {
            "sensitivity": {"ci95_low": row_imgs[0][2], "ci95_high": row_imgs[0][3]},
            "specificity": {"ci95_low": row_imgs[1][2], "ci95_high": row_imgs[1][3]},
            "f1": {"ci95_low": row_imgs[2][2], "ci95_high": row_imgs[2][3]},
        },
        "lesion_size": size_rows,
    }, indent=2, default=float))

    # ---- error analysis exports ----
    err_root = get_run_dir() / "error_analysis"
    buckets = ["false_positive", "false_negative", "low_dice", "small_lesion_missed",
               "high_confidence_error"]
    for b in buckets:
        d = err_root / b
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    err_rows = []
    q_small = quants[0]
    if args.mode == "twostage":
        err_model = TwoStageSegmenter(args.stage1_weights, args.stage2_weights,
                                      float(thr.get("stage1_conf", 0.2)), conf)
    else:
        err_model = model if model is not None else YOLO(args.weights)
    for r in recs:
        gt_pos = r["has_calculus"]
        pred_pos = r["max_conf"] >= conf
        tag = None
        if not gt_pos and pred_pos:
            tag = "false_positive"
        elif gt_pos and not pred_pos:
            tag = "false_negative"
            if r["gt_px"] <= q_small:
                tag = "small_lesion_missed"
        elif gt_pos and pred_pos:
            if r["dice"] < 0.3:
                tag = "low_dice"
                if r["max_conf"] >= 0.8:
                    tag = "high_confidence_error"
        if tag:
            dst = err_root / tag / f"{r['image_id']}.png"
            img = cv2.imread(r["image_path"])
            gt = cv2.imread(r["mask_path"], cv2.IMREAD_UNCHANGED)
            if args.mode == "twostage":
                ann = img.copy()
                ann[(gt > 0)] = (ann[gt > 0] * 0.5 + np.array([0, 0, 255]) * 0.5).astype(np.uint8)
                pred_m, _ = err_model.predict(r["image_path"])
                if pred_m is not None:
                    ann[pred_m > 0] = (ann[pred_m > 0] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
                cv2.imwrite(str(dst), ann)
                err_rows.append({"image_id": r["image_id"], "split": "test", "category": tag,
                                 "superset": "false_positive" if tag == "false_positive" else
                                 ("false_negative" if tag in ("false_negative", "small_lesion_missed") else
                                  "low_dice"), "dice": r["dice"], "iou": r["iou"],
                                 "max_conf": r["max_conf"], "gt_px": r["gt_px"], "pred_px": r["pred_px"]})
                continue
            res = err_model.predict(r["image_path"], conf=conf, iou=0.5,
                                    imgsz=640, device=0, retina_masks=True,
                                    verbose=False)[0]
            ann = img.copy()
            ann[(gt > 0)] = (ann[gt > 0] * 0.5 + np.array([0, 0, 255]) * 0.5).astype(np.uint8)
            if res.masks is not None:
                mk = res.masks.data.cpu().numpy()
                h, w = img.shape[:2]
                for i in range(mk.shape[0]):
                    if res.boxes.conf[i].item() < conf:
                        continue
                    mi = cv2.resize((mk[i] > 0.5).astype(np.uint8) * 255, (w, h))
                    ann[mi > 0] = (ann[mi > 0] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
            cv2.imwrite(str(dst), ann)
            err_rows.append({"image_id": r["image_id"], "split": "test", "category": tag,
                             "superset": "false_positive" if tag == "false_positive" else
                             ("false_negative" if tag in ("false_negative", "small_lesion_missed") else
                              "low_dice"), "dice": r["dice"], "iou": r["iou"],
                             "max_conf": r["max_conf"], "gt_px": r["gt_px"], "pred_px": r["pred_px"]})
    pd.DataFrame(err_rows).to_csv(run_dir / "error_analysis_results.csv", index=False)

    # ---- test visualizations (>=50) ----
    viz_dir = get_run_dir() / "test_visualizations"
    if viz_dir.exists():
        shutil.rmtree(viz_dir)
    viz_dir.mkdir(parents=True)
    rng = random.Random(42)
    sel_obj = [r for r in recs if r["has_calculus"]]
    sel_neg = [r for r in recs if not r["has_calculus"]]
    picked = rng.sample(sel_obj, min(len(sel_obj), args.num_viz)) + rng.sample(
        sel_neg, min(len(sel_neg), max(0, args.num_viz - len(sel_obj))))
    from ultralytics import YOLO as _Y2
    if args.mode == "twostage":
        seg2 = TwoStageSegmenter(args.stage1_weights, args.stage2_weights,
                                 float(thr.get("stage1_conf", 0.2)), conf)
        def viz_predict(path):
            return seg2.predict(path)[0]
    else:
        vm2 = model if model is not None else _Y2(args.weights)
        def viz_predict(path):
            res = vm2.predict(path, conf=conf, iou=0.5, imgsz=640, device=0,
                              retina_masks=True, verbose=False)[0]
            h, w = res.orig_shape
            pred = np.zeros((h, w), np.uint8)
            if res.masks is not None:
                mk = res.masks.data.cpu().numpy()
                for i in range(mk.shape[0]):
                    if res.boxes.conf[i].item() < conf:
                        continue
                    mi = cv2.resize((mk[i] > 0.5).astype(np.uint8) * 255, (w, h))
                    pred[mi > 0] = 1
            return pred
    for r in picked:
        img = cv2.imread(r["image_path"])
        gt = cv2.imread(r["mask_path"], cv2.IMREAD_UNCHANGED)
        h, w = img.shape[:2]
        pred = viz_predict(r["image_path"])
        if pred is None:
            pred = np.zeros((h, w), np.uint8)
        gt = (gt > 0).astype(np.uint8)
        inter = ((pred > 0) & (gt > 0)).sum(); union = ((pred > 0) | (gt > 0)).sum()
        dice = 2 * inter / (2 * inter + (pred > 0).sum() - inter + (gt > 0).sum() - inter) if (inter or (pred > 0).sum() or (gt > 0).sum()) else 0
        iou = inter / union if union else 0
        chs = (pred > 0)
        overlay = img.copy()
        b = np.zeros((h, w), np.uint16)
        overlay[gt > 0] = (overlay[gt > 0] * 0.5 + np.array([0, 0, 255]) * 0.5).astype(np.uint8)
        overlay[chs] = (overlay[chs] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
        info = f"conf={r['max_conf']:.2f} dice={dice:.3f} iou={iou:.3f}"
        cv2.putText(overlay, info, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        combo = np.hstack([img, cv2.cvtColor(gt * 255, cv2.COLOR_GRAY2BGR),
                           np.where(chs[:, :, None], np.array([0, 255, 0]), img).astype(np.uint8), overlay])
        cv2.imwrite(str(viz_dir / f"{r['image_id']}.png"), combo)

    summary = {
        "frozen_conf": conf, "n_test": len(recs), "n_pos_test": int(y.sum()),
        "n_neg_test": int((y == 0).sum()), "image_metrics": imgs, "auroc": round(float(auc), 4),
        "pixel_agg_all": all_dice, "pixel_agg_positive": pos_dice,
        "pix_total": {"precision": round(pix_prec, 4), "recall": round(pix_rec, 4), "f1": round(pix_f1, 4)},
    }
    (run_dir / "test_eval_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    logger.info(f"test done. sens={imgs['sensitivity']:.4f} spec={imgs['specificity']:.4f} "
                f"f1={imgs['f1']:.4f} auroc={auc:.4f} mAP50(M)={map50:.4f} mAP50-95(M)={map5095:.4f}")


if __name__ == "__main__":
    main()
