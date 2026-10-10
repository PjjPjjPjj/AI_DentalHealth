"""Calibrate plaque v3 + 3-way comparison (v1 / v2_sano / v3_mined).

Calibration set (never used for training): Mendeley extended val (2,015 crops,
GT labels) + 503 SANO val crops (label=0, 20 held-out images).
Fits Platt (logistic on logit) and isotonic; reports ECE(15-bin) before/after;
freezes operating points:
  raw thr       : extended-val protocol, max spec s.t. sens>=0.95
  calibrated thr: same rule on calibrated scores (combined calib set)

Output: outputs/plaque_bin_v3_mined/eval_v3_calib.{json,md}
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

ROOT = Path(f"{_ROOT}")
DATA1 = ROOT / "datasets/processed/plaque_bin_v1"
DATA3 = ROOT / "datasets/processed/plaque_bin_v3"
SANO = ROOT / "datasets/processed/plaque_sano"
MODELS = {
    "v1": ROOT / "outputs/plaque_bin_v1/weights/best.pt",
    "v2_sano": ROOT / "outputs/plaque_bin_v2_sano/weights/best.pt",
    "v3_mined": ROOT / "runs/classify/runs/classify/outputs/plaque_bin_v3_mined/weights/best.pt",
    "v3c_rob": ROOT / "runs/classify/runs/classify/outputs/plaque_bin_v3c_mined/weights/best.pt",
}
OUT = ROOT / "outputs/plaque_bin_v3_mined"


def predict(model, paths, pidx, bs=128):
    out = []
    for i in range(0, len(paths), bs):
        for r in model.predict(paths[i:i + bs], imgsz=224, verbose=False, device=0):
            out.append(float(r.probs.data[pidx]))
    return np.array(out)


def cm(y, p, t):
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    return {"sens": round(sens, 4), "spec": round(spec, 4), "prec": round(prec, 4),
            "f1": round(2 * prec * sens / max(prec + sens, 1e-9), 4)}


def ece(y, p, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1])
        if m.sum() == 0:
            continue
        e += m.mean() * abs(y[m].mean() - p[m].mean())
    return round(float(e), 4)


def freeze_thr(y, s, sens_min=0.95):
    grid = np.round(np.arange(0.02, 0.981, 0.01), 2)
    best, best_spec = None, -1
    for t in grid:
        pred = (s >= t).astype(int)
        sens = ((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)
        spec = ((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)
        if sens >= sens_min and spec > best_spec:
            best, best_spec = float(t), spec
    return best


def main():
    from ultralytics import YOLO
    from sklearn.metrics import roc_auc_score
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    rows = pd.read_csv(DATA1 / "plaque_crops.csv")
    uniq = rows[rows.is_dup == 0]
    val = uniq[uniq.split == "val"].reset_index(drop=True)
    test = uniq[uniq.split == "test"].reset_index(drop=True)
    val_clean = val[val.is_aug == 0]
    test_clean = test[test.is_aug == 0]

    sc = pd.read_csv(DATA3 / "sano_val_images.json")
    import json as _j
    sc = _j.loads((DATA3 / "sano_val_images.json").read_text())
    sano_all = pd.read_csv(SANO / "sano_crops.csv")
    stems = {Path(s).stem.replace(" ", "_") for s in sc}
    mask = sano_all.image.apply(lambda s: Path(s).stem.replace(" ", "_") in stems)
    sano_val = sano_all[mask & (sano_all.status == "kept")].reset_index(drop=True)

    rep = {}
    calib_store = {}
    for name, w in MODELS.items():
        model = YOLO(str(w))
        pidx = [i for i, n in model.names.items() if n == "plaque"][0]
        pv = predict(model, [str(DATA1 / c) for c in val.crop], pidx)
        pt = predict(model, [str(DATA1 / c) for c in test.crop], pidx)
        ps_val = predict(model, [str(SANO / c) for c in sano_val.crop], pidx)
        ps_ev = predict(model, [str(SANO / c) for c in
                                sano_all[(sano_all.subset == "eval") & (sano_all.status == "kept")].crop],
                        pidx)
        sano_ev = sano_all[(sano_all.subset == "eval") & (sano_all.status == "kept")]

        yv = val.label.values.astype(int)
        yt = test.label.values.astype(int)

        # --- calibration (v3 variants; v1/v2 reported raw for reference) ---
        r = {}
        if name in ("v3_mined", "v3c_rob"):
            pc = np.concatenate([pv, ps_val])
            yc = np.concatenate([yv, np.zeros(len(ps_val), int)])
            logit = np.log(np.clip(pc, 1e-6, 1 - 1e-6) / (1 - np.clip(pc, 1e-6, 1 - 1e-6)))
            pl = LogisticRegression(C=1e6).fit(logit.reshape(-1, 1), yc)
            iso = IsotonicRegression(out_of_bounds="clip").fit(pc, yc)

            def apply_platt(p):
                lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
                return pl.predict_proba(lg.reshape(-1, 1))[:, 1]

            r["calibration"] = {
                "n_calib": int(len(pc)), "n_pos": int(yc.sum()),
                "platt_a": round(float(pl.coef_[0][0]), 4), "platt_b": round(float(pl.intercept_[0]), 4),
                "ece_raw": ece(yc, pc),
                "ece_platt": ece(yc, apply_platt(pc)),
                "ece_isotonic": ece(yc, iso.predict(pc)),
            }
            pc_pl = apply_platt(pc)
            thr_cal = freeze_thr(yc, pc_pl)
            calib_store["platt"] = (float(pl.coef_[0][0]), float(pl.intercept_[0]))
            calib_store["thr_calibrated"] = thr_cal
            r["thr_calibrated_platt"] = thr_cal
            r["calib_combined_at_thr_platt"] = cm(yc, pc_pl, thr_cal)
            # calibrated test/sano
            r["test_extended_cal_platt"] = cm(yt, apply_platt(pt), thr_cal)
            r["sano_eval_cal_platt"] = cm(np.zeros(len(ps_ev), int), apply_platt(ps_ev), thr_cal)

        # --- raw protocol ---
        thr_raw = freeze_thr(yv, pv)
        r.update({
            "weights": str(w), "thr_raw_frozen": thr_raw,
            "auroc_test_clean": round(float(roc_auc_score(
                test_clean.label, [pt[i] for i in test_clean.index])), 4),
            "test_extended_raw": cm(yt, pt, thr_raw),
            "sano_eval_raw": cm(np.zeros(len(ps_ev), int), ps_ev, thr_raw),
            "sano_median_p": round(float(np.median(ps_ev)), 3),
        })
        rep[name] = r

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "calibration_platt.npz", **{k: np.array(v) for k, v in calib_store.items()})
    (OUT / "eval_v3_calib.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))

    v1, v2, v3, v3c = rep["v1"], rep["v2_sano"], rep["v3_mined"], rep["v3c_rob"]
    m = ["# plaque v3/v3c 与校准评估", "",
         f"| 指标 | v1 | v2_sano | v3_mined(+挖掘) | v3c_rob(纯稳健变体) |",
         f"|---|---|---|---|---|",
         f"| AUROC test clean | {v1['auroc_test_clean']} | {v2['auroc_test_clean']} | {v3['auroc_test_clean']} | {v3c['auroc_test_clean']} |",
         f"| raw 冻结 thr | {v1['thr_raw_frozen']} | {v2['thr_raw_frozen']} | {v3['thr_raw_frozen']} | {v3c['thr_raw_frozen']} |",
         f"| testExt Sens/Spec @raw | {v1['test_extended_raw']['sens']}/{v1['test_extended_raw']['spec']} | "
         f"{v2['test_extended_raw']['sens']}/{v2['test_extended_raw']['spec']} | "
         f"{v3['test_extended_raw']['sens']}/{v3['test_extended_raw']['spec']} | "
         f"{v3c['test_extended_raw']['sens']}/{v3c['test_extended_raw']['spec']} |",
         f"| SANO Spec @raw | {v1['sano_eval_raw']['spec']} | {v2['sano_eval_raw']['spec']} | "
         f"{v3['sano_eval_raw']['spec']} | {v3c['sano_eval_raw']['spec']} |", "",
         "## 校准（v3c_rob，calib=Mendeley extended val + 503 SANO val 阴性）",
         f"- ECE: raw {v3c['calibration']['ece_raw']} -> Platt {v3c['calibration']['ece_platt']} / isotonic {v3c['calibration']['ece_isotonic']}",
         f"- Platt: logit(P')={v3c['calibration']['platt_a']}*logit(P)+{v3c['calibration']['platt_b']}",
         f"- 校准分数冻结 thr={v3c['thr_calibrated_platt']}（sens>=0.95 下最大 spec，组合 calib）",
         f"- @校准 thr: testExt {v3c['test_extended_cal_platt']['sens']}/{v3c['test_extended_cal_platt']['spec']} | "
         f"SANO {v3c['sano_eval_cal_platt']['spec']}"]
    (OUT / "eval_v3_calib.md").write_text("\n".join(m) + "\n")
    print("\n".join(m))


if __name__ == "__main__":
    main()
