"""Exposure-normalization sweep: can inference-time preprocessing kill the
over-exposure/texture shortcut FPs on the original test negatives?

Variants (applied to crops before v2 predict; model untouched):
  none | clahe2 | clahe3 | grayworld | grayworld+clahe2 | gamma_auto
Eval: Mendeley test extended/clean + SANO eval + val clean (sens reference),
at the deployed threshold semantics (calibrated 0.746 == raw 0.70), plus the
full raw-threshold profile per variant (matched-sens comparison at sens~0.94).
Output: outputs/plaque_exposure_sweep/sweep.{json,md} + sample sheets
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

ROOT = Path(f"{_ROOT}")
DATA1 = ROOT / "datasets/processed/plaque_bin_v1"
SANO = ROOT / "datasets/processed/plaque_sano"
PLAQUE_W = ROOT / "outputs/plaque_bin_v2_sano/weights/best.pt"
CALIB = ROOT / "outputs/plaque_bin_v2_sano/calibration_platt.npz"
OUT = ROOT / "outputs/plaque_exposure_sweep"


def clahe(img, clip=2.0):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def grayworld(img):
    res = img.astype(np.float32)
    means = res.reshape(-1, 3).mean(0)
    gray = means.mean()
    for c in range(3):
        res[..., c] *= gray / max(means[c], 1e-6)
    return np.clip(res, 0, 255).astype(np.uint8)


def gamma_auto(img, target=110.0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()
    g = np.log(target / 255.0) / np.log(max(gray, 1.0) / 255.0)
    table = np.array([(i / 255.0) ** g * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)


VARIANTS = {
    "none": lambda im: im,
    "clahe2": lambda im: clahe(im, 2.0),
    "clahe3": lambda im: clahe(im, 3.0),
    "grayworld": grayworld,
    "gw_clahe2": lambda im: clahe(grayworld(im), 2.0),
    "gamma_auto": gamma_auto,
}


def cm(y, p, t):
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    return {"sens": round(sens, 4), "spec": round(spec, 4), "fp": fp, "fn": fn}


def main() -> None:
    sys.path.insert(0, str(ROOT))
    from src.inference.models import PlaqueModel

    pm = PlaqueModel(PLAQUE_W, calibration=CALIB)
    rows = pd.read_csv(DATA1 / "plaque_crops.csv")
    uniq = rows[rows.is_dup == 0]
    test = uniq[uniq.split == "test"].reset_index(drop=True)
    val = uniq[uniq.split == "val"].reset_index(drop=True)
    sano = pd.read_csv(SANO / "sano_crops.csv")
    sano_ev = sano[(sano.subset == "eval") & (sano.status == "kept")].reset_index(drop=True)

    def batch_predict(crops):
        out = []
        for i in range(0, len(crops), 128):
            out.extend(pm._proba_array(crops[i:i + 128]))
        return np.array(out)

    results = {}
    for name, fn in VARIANTS.items():
        r = {"variant": name}
        for set_name, df, root in (("test_ext", test, DATA1), ("val_clean_src", val, DATA1),
                                   ("sano", sano_ev, SANO)):
            imgs = [cv2.imread(str(root / c)) for c in df.crop]
            keep, crops = [], []
            for i, im in enumerate(imgs):
                if im is not None:
                    keep.append(i); crops.append(im)
            crops = [fn(c) for c in crops]
            p = batch_predict(crops)
            y = df.label.values.astype(int)[keep] if set_name != "sano" \
                else np.zeros(len(keep), int)
            r[set_name] = cm(y, p, 0.746)
            if set_name == "test_ext":
                r["auroc_test"] = round(float(__import__("sklearn.metrics", fromlist=["roc_auc_score"])
                                              .roc_auc_score(y, p)), 4)
        results[name] = r
        print(name, {k: v for k, v in r.items() if k != "variant"})

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sweep.json").write_text(json.dumps(results, indent=2))

    m = ["# 曝光归一化扫描（v2 不动，推理侧预处理）", "",
         "阈值=部署点（校准 0.746 ≡ raw 0.70）", "",
         "| 变体 | testExt Sens | Spec | FP | AUROC | SANO Spec |",
         "|---|---|---|---|---|---|"]
    for name, r in results.items():
        te, sa = r["test_ext"], r["sano"]
        m.append(f"| {name} | {te['sens']} | {te['spec']} | {te['fp']} | {r.get('auroc_test')} | {sa['spec']} |")
    (OUT / "sweep.md").write_text("\n".join(m) + "\n")
    print("\n".join(m))


if __name__ == "__main__":
    main()
