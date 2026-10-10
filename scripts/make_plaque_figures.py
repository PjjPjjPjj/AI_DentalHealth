"""Generate plaque principle/example figures from REAL experiment artifacts.

Outputs -> outputs/demo/pipeline_walkthrough/plaque_figs/
  fig_plaque_crops.jpg      model-view: highest vs lowest prob tooth crops (test)
  fig_plaque_threshold.jpg  operating-point curve from frozen val sweep (eval_plaque_bin)
  fig_plaque_examples_*.jpg full-image plaque overlays on diverse photos
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

sys.path.insert(0, f"{_ROOT}")
ROOT = Path(f"{_ROOT}")
DATA = ROOT / "datasets/processed/plaque_bin_v1"
OUT = ROOT / "outputs/demo/pipeline_walkthrough/plaque_figs"
OUT.mkdir(parents=True, exist_ok=True)


def label_img(img, text, color=(0, 255, 0)):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (30, 30, 30), -1)
    cv2.putText(img, text, (6, 19), 0, 0.55, color, 1)
    return img


def tile(rows, cols, imgs, pad=6):
    h = max(i.shape[0] for i in imgs)
    w = max(i.shape[1] for i in imgs)
    grid = np.full(((h + pad) * rows + pad, (w + pad) * cols + pad, 3), 245, np.uint8)
    for k, im in enumerate(imgs):
        r, c = divmod(k, cols)
        y, x = pad + r * (h + pad), pad + c * (w + pad)
        grid[y:y + im.shape[0], x:x + im.shape[1]] = im
    return grid


def fig_crops():
    """Top-8 highest P(plaque) crops vs bottom-8 lowest (val+test, unique)."""
    from src.inference.models import PlaqueModel
    pm = PlaqueModel(weights=str(ROOT / "outputs/plaque_bin_v1/weights/best.pt"), imgsz=224)
    rows = []
    with open(DATA / "plaque_crops.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["is_dup"] == "0" and r["split"] in ("val", "test"):
                rows.append(r)
    probs, keep = [], []
    batch = []
    for r in rows:
        p = DATA / r["crop"]
        if not p.exists():
            continue
        batch.append((r, str(p)))
    B = 128
    for i in range(0, len(batch), B):
        chunk = batch[i:i + B]
        res = pm._proba_array([p for _, p in chunk])
        probs += list(res)
        keep += [r for r, _ in chunk]
    probs = np.array(probs)
    order = np.argsort(-probs)
    hi = order[:10]
    lo = order[-10:]
    # of the top: keep true positives for honesty; mark labels
    hi_imgs, hi_txt = [], []
    for j in hi:
        im = cv2.imread(str(DATA / keep[j]["crop"]))
        if im is None:
            continue
        im = cv2.resize(im, (200, 150))
        gt = int(keep[j]["label"])
        ok = "o" if (probs[j] >= 0.5) == (gt == 1) else "x"
        hi_imgs.append(label_img(im, f"p={probs[j]:.2f} GT:{'plaque' if gt else 'normal'} {ok}",
                                 (0, 255, 255) if gt else (0, 200, 255)))
        hi_txt.append(keep[j])
        if len(hi_imgs) == 8:
            break
    lo_imgs = []
    for j in lo[::-1]:
        im = cv2.imread(str(DATA / keep[j]["crop"]))
        if im is None:
            continue
        im = cv2.resize(im, (200, 150))
        gt = int(keep[j]["label"])
        lo_imgs.append(label_img(im, f"p={probs[j]:.2f} GT:{'plaque' if gt else 'normal'}",
                                 (0, 255, 0) if gt == 0 else (0, 0, 255)))
        if len(lo_imgs) == 8:
            break
    g1 = tile(2, 4, hi_imgs)
    g2 = tile(2, 4, lo_imgs)
    Wc = max(g1.shape[1], g2.shape[1])
    canvas = np.full((g1.shape[0] + g2.shape[0] + 140, Wc, 3), 255, np.uint8)
    cv2.putText(canvas, "Model view: highest P(plaque) crops (val/test)", (20, 40), 0, 1.0, (0, 0, 0), 2)
    canvas[60:60 + g1.shape[0], :g1.shape[1]] = g1
    y2 = 60 + g1.shape[0] + 40
    cv2.putText(canvas, "Model view: lowest P(plaque) crops (clean tooth surfaces)", (20, y2), 0, 1.0, (0, 0, 0), 2)
    canvas[y2 + 20:y2 + 20 + g2.shape[0], :g2.shape[1]] = g2
    cv2.imwrite(str(OUT / "fig_plaque_crops.jpg"), canvas)
    print("saved fig_plaque_crops.jpg", len(hi_imgs), len(lo_imgs))


def fig_threshold():
    d = json.load(open(ROOT / "outputs/plaque_bin_v1/eval_plaque_bin.json"))
    sw = d["sweep_extended_val"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    fm.fontManager.addfont(str(ROOT / "assets/fonts/NotoSansSC.otf"))
    plt.rcParams["font.family"] = "Noto Sans CJK SC"
    thr = [s["thr"] for s in sw]
    sens = [s["sens"] for s in sw]
    spec = [s["spec"] for s in sw]
    f1 = [s["f1"] for s in sw]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ax.plot(thr, sens, "-o", ms=3, label="Sensitivity (不漏检)")
    ax.plot(thr, spec, "-s", ms=3, label="Specificity (不误报)")
    ax.plot(thr, f1, "-^", ms=3, label="F1")
    op = d["thr_screen"]["thr"]
    ax.axvline(op, color="red", ls="--", lw=1.2)
    ax.annotate(f"冻结工作点 thr={op:.2f}\n(Sens≥0.95 下 Spec 最大)",
                xy=(op, d["thr_screen"]["sens"]), xytext=(op + 0.06, 0.45),
                arrowprops=dict(arrowstyle="->", color="red"), fontsize=11, color="red")
    ax.set_xlabel("判定阈值 P(plaque) ≥ thr", fontsize=12)
    ax.set_ylabel("指标", fontsize=12)
    ax.set_title("牙菌斑二分类：阈值-指标曲线（extended val 实测）", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_plaque_threshold.jpg", bbox_inches="tight")
    plt.close(fig)
    print("saved fig_plaque_threshold.jpg")


def fig_examples():
    from src.inference.models import PlaqueModel, ToothModel
    tm = ToothModel(weights=str(ROOT / "outputs/iop_toothseg/weights/best.pt"),
                    conf=0.3, iou=0.7, imgsz=896)
    pm = PlaqueModel(weights=str(ROOT / "outputs/plaque_bin_v1/weights/best.pt"),
                     imgsz=224, conf_threshold=0.92)
    import pandas as pd
    man = pd.read_csv(ROOT / "datasets/processed/calculus_seg_v3/split_manifest_v3.csv")
    picks = [
        man[(man.split == "test") & (man.has_calculus == True)].iloc[8].image_path,
        str(next((ROOT / "datasets/raw/Plaque").rglob("patient0057_20260118_bottom-left.jpg"))),
        str(next((ROOT / "datasets/raw/caries6313").rglob("anonymous-frontalView-1726826132608.jpg"))) \
            if any((ROOT / "datasets/raw/caries6313").rglob("anonymous-frontalView-1726826132608.jpg")) \
            else man[man.split == "test"].iloc[100].image_path,
    ]
    k = 0
    for p in picks:
        img = cv2.imread(str(p))
        if img is None:
            continue
        H, W = img.shape[:2]
        teeth = tm.predict(img)
        n_hi = 0
        for i, t in enumerate(teeth):
            x0, y0, x1, y1 = [int(max(0, v)) for v in t["bbox"]]
            x1, y1 = min(W, x1), min(H, y1)
            crop = img[y0:y1, x0:x1]
            if not crop.size:
                continue
            prob = pm.predict_proba(crop)
            c = (0, 0, 255) if prob >= 0.92 else (0, 200, 0)
            cv2.rectangle(img, (x0, y0), (x1, y1), c, 2)
            cv2.putText(img, f"T{i+1} p={prob:.2f}", (x0, max(14, y0 - 4)), 0, 0.5, c, 2)
            n_hi += prob >= 0.92
        cv2.putText(img, f"teeth={len(teeth)} plaque(p>=0.92)={n_hi}", (10, 24), 0, 0.8, (255, 255, 0), 2)
        cv2.imwrite(str(OUT / f"fig_plaque_example_{k}.jpg"), img)
        print("saved example", k, Path(p).name, f"{len(teeth)} teeth {n_hi} present")
        k += 1


if __name__ == "__main__":
    fig_threshold()
    fig_examples()
    fig_crops()
