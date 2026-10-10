"""Richer image-level evidence aggregation on val (no retraining).

Per-image evidence from the two-stage+verifier pipeline:
  e1 max fused prob, e2 2nd max, e3 mean top3, e4 min(sum,3) of fused,
  e5 max s2_conf, e6 max verifier_prob, e7 log1p(n_cand),
  e8 max s2_area_ratio (top3 fused), e9 n_windows_with_pred
Combined with imgB (EffNet+TTA) and imgC (YOLO-cls). Variants compared with
5-fold CV accuracy on val; best variant frozen to ensemble_v3b.npz format.

Output: outputs/calculus_v3_cls/imgagg_val_report.json + chosen npz
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from ultralytics import YOLO

from scripts.v3_pipeline_lib import (DATA, FCOLS, OUT, collect_split,
                                     make_verifier)


def evidence(imgs) -> np.ndarray:
    z = np.load(OUT / "fusion_v3.npz")
    from sklearn.linear_model import LogisticRegression as LR
    m3 = LR(max_iter=1000, class_weight="balanced")
    m3.coef_, m3.intercept_, m3.classes_ = z["coef"], z["intercept"], np.array([0, 1])
    mu, sg = z["mu"], z["sg"]
    rows = []
    for im in imgs:
        feats = im["feats"]
        if not feats:
            rows.append([0] * 9)
            continue
        X = np.array([[f[c] for c in FCOLS] for f in feats], float)
        fused = m3.predict_proba((X - mu) / sg)[:, 1]
        srt = np.sort(fused)[::-1]
        top3 = srt[:3]
        idx3 = np.argsort(-fused)[:3]
        rows.append([
            float(srt[0]),
            float(srt[1]) if len(srt) > 1 else 0.0,
            float(top3.mean()),
            float(min(fused.sum(), 3.0)),
            float(max(f["s2_conf"] for f in feats)),
            float(max(f["verifier_prob"] for f in feats)),
            float(np.log1p(len(feats))),
            float(max(feats[i]["s2_area_ratio"] for i in idx3)),
            float(len(im["rects"])),
        ])
    return np.array(rows, float)


def main() -> None:
    s1m = YOLO("outputs/calculus_v3b_stage1/weights/best.pt")
    s2m = YOLO("outputs/calculus_v3b_stage2/weights/best.pt")
    vb = make_verifier()
    man = pd.read_csv(DATA / "split_manifest_v3.csv")
    print("collecting val ...", flush=True)
    imgs = collect_split(s1m, s2m, man, "val", with_verifier=True, vb=vb)
    E = evidence(imgs)
    y = man[man.split == "val"].has_calculus.values.astype(int)
    pd.DataFrame(E, columns=[f"e{i+1}" for i in range(E.shape[1])]) \
        .assign(y=y).to_csv(OUT / "val_evidence.csv", index=False)

    from scripts.eval_ensemble_v3b import effnet_tta_probs, yolocls_probs
    imgB = effnet_tta_probs(man, "val")
    imgC = yolocls_probs("val")

    variants = {
        "e1": E[:, :1],
        "e_all": E,
        "e_top4": E[:, :4],
        "BC": np.stack([imgB, imgC], 1),
        "e1_BC": np.stack([E[:, 0], imgB, imgC], 1),
        "e_all_BC": np.concatenate([E, np.stack([imgB, imgC], 1)], 1),
    }
    rep = {}
    best = None
    for name, X in variants.items():
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        accs, aucs = [], []
        oof = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            lr = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
            p = lr.predict_proba(X[te])[:, 1]
            oof[te] = p
            tscan = [(((p >= t).astype(int) == y[te]).mean(), t)
                     for t in np.arange(0.05, 0.96, 0.02)]
            accs.append(max(tscan)[0])
            aucs.append(roc_auc_score(y[te], p))
        lr_full = LogisticRegression(max_iter=1000).fit(X, y)
        p_full = lr_full.predict_proba(X)[:, 1]
        scan = [(((p_full >= t).astype(int) == y).mean(), float(t))
                for t in np.arange(0.05, 0.96, 0.01)]
        acc_full, thr_full = max(scan)
        rep[name] = {"cv_acc": round(float(np.mean(accs)), 4),
                     "cv_auroc": round(float(np.mean(aucs)), 4),
                     "full_acc": round(acc_full, 4),
                     "full_auroc": round(float(roc_auc_score(y, p_full)), 4),
                     "thr": round(thr_full, 2)}
        if best is None or rep[name]["cv_acc"] > rep[best]["cv_acc"]:
            best = name
        print(name, rep[name])
    rep["_best"] = best
    (OUT / "imgagg_val_report.json").write_text(json.dumps(rep, indent=2))
    np.savez(OUT / "imgagg_val.npz", E=E, imgB=imgB, imgC=imgC, y=y)


if __name__ == "__main__":
    main()
