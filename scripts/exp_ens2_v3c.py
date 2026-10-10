"""Ensemble v2 image-level fit on val: 4 streams + optional domain routing.

Streams: imgA (v3c pipeline max fused), imgB (EffNet+TTA), imgC (YOLO-cls),
imgD (ConvNeXt-Tiny).  Routing: image max(W,H)>=500 -> 'wide' else 'closeup';
separate logistic + threshold per group (fit on val only).
Freezes best-by-CV variant to ensemble_v3b.npz format (+ routing flag).
Output: outputs/calculus_v3_cls/ensemble_v2_report.json
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import json

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from ultralytics import YOLO

from scripts.eval_ensemble_v3b import effnet_tta_probs, yolocls_probs
from scripts.v3_pipeline_lib import DATA, FCOLS, OUT, collect_split, make_verifier


def imgA_of(man, split):
    s1m = YOLO("outputs/calculus_v3c_stage1/weights/best.pt")
    s2m = YOLO("outputs/calculus_v3b_stage2/weights/best.pt")
    vb = make_verifier()
    imgs = collect_split(s1m, s2m, man, split, with_verifier=True, vb=vb,
                         log_every=350)
    z = np.load(OUT / "fusion_v3.npz")
    m3 = LogisticRegression(max_iter=1000, class_weight="balanced")
    m3.coef_, m3.intercept_, m3.classes_ = z["coef"], z["intercept"], np.array([0, 1])
    mu, sg = z["mu"], z["sg"]
    a = []
    for im in imgs:
        f = im["feats"]
        if f:
            X = np.array([[x[c] for c in FCOLS] for x in f], float)
            a.append(float(m3.predict_proba((X - mu) / sg)[:, 1].max()))
        else:
            a.append(0.0)
    return np.array(a), imgs


def cv_acc(X, y, routed=None):
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    accs = []
    for tr, te in skf.split(X, y):
        if routed is None:
            lr = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
            p = lr.predict_proba(X[te])[:, 1]
            scan = [(((p >= t).astype(int) == y[te]).mean(), t)
                    for t in np.arange(0.05, 0.96, 0.02)]
            accs.append(max(scan)[0])
        else:
            g = routed[te]
            p = np.zeros(len(te))
            for grp in ("wide", "closeup"):
                mtr, mte = routed[tr] == grp, g == grp
                if mtr.sum() < 20 or mte.sum() == 0:
                    continue
                lr = LogisticRegression(max_iter=1000).fit(X[tr][mtr], y[tr][mtr])
                p[mte] = lr.predict_proba(X[te][mte])[:, 1]
            acc = 0.0
            for grp in ("wide", "closeup"):
                m = g == grp
                if m.sum() == 0:
                    continue
                scan = [(((p[m] >= t).astype(int) == y[te][m]).mean(), t)
                        for t in np.arange(0.05, 0.96, 0.02)]
                acc += max(scan)[0] * m.sum()
            accs.append(acc / len(te))
    return float(np.mean(accs))


def main() -> None:
    man = pd.read_csv(DATA / "split_manifest_v3.csv")
    rows = man[man.split == "val"].reset_index(drop=True)
    y = rows.has_calculus.values.astype(int)
    sizes = [max(cv2.imread(p).shape[:2]) for p in rows.image_path]
    routed = np.where(np.array(sizes) >= 500, "wide", "closeup")

    print("collecting imgA (val, tiled) ...", flush=True)
    imgA, _ = imgA_of(man, "val")
    imgB = effnet_tta_probs(man, "val")
    imgC = yolocls_probs("val")
    import torch
    import torch.nn as nn
    from torchvision import models as tv
    m = tv.convnext_tiny(weights=None)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, 1)
    m.load_state_dict(torch.load(OUT / "convnext" / "best.pt", weights_only=False))
    m = m.to(0).eval()
    from scripts.train_v3c_convnext import prep
    probs = []
    with torch.no_grad():
        for i in range(0, len(rows), 24):
            batch = rows.iloc[i:i + 24]
            xs = [prep(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))
                  for p in batch.image_path]
            probs += m(torch.stack(xs).to(0)).squeeze(1).sigmoid().cpu().tolist()
    imgD = np.array(probs)
    for nm, sc in (("imgA", imgA), ("imgB", imgB), ("imgC", imgC), ("imgD", imgD)):
        print(nm, "val AUROC", round(float(roc_auc_score(y, sc)), 4))

    combos = {
        "ABC": np.stack([imgA, imgB, imgC], 1),
        "ABD": np.stack([imgA, imgB, imgD], 1),
        "BCD": np.stack([imgB, imgC, imgD], 1),
        "ABCD": np.stack([imgA, imgB, imgC, imgD], 1),
        "BD": np.stack([imgB, imgD], 1),
    }
    rep = {}
    for nm, X in combos.items():
        rep[nm] = {"cv_acc": round(cv_acc(X, y), 4)}
        rep[nm + "+route"] = {"cv_acc": round(cv_acc(X, y, routed), 4)}
        print(nm, rep[nm], rep[nm + "+route"])
    best = max(((k, v["cv_acc"]) for k, v in rep.items()), key=lambda kv: kv[1])
    print("BEST:", best)
    # refit best on full val and freeze
    name = best[0]
    routed_flag = name.endswith("+route")
    base = name.replace("+route", "")
    X = combos[base]
    if routed_flag:
        coef = {}
        thr = {}
        for grp in ("wide", "closeup"):
            m = routed == grp
            lr = LogisticRegression(max_iter=1000).fit(X[m], y[m])
            p = lr.predict_proba(X[m])[:, 1]
            scan = [(((p >= t).astype(int) == y[m]).mean(), float(t))
                    for t in np.arange(0.05, 0.96, 0.01)]
            acc, t = max(scan)
            coef[grp] = {"coef": lr.coef_, "intercept": lr.intercept_,
                         "thr": t, "acc": acc}
            print(grp, "acc", round(acc, 4), "thr", round(t, 2))
        np.savez(OUT / "ensemble_v3b.npz", coef=np.array([1]),  # placeholder
                 intercept=np.array([0.0]),
                 wide_coef=coef["wide"]["coef"], wide_int=coef["wide"]["intercept"],
                 wide_thr=coef["wide"]["thr"],
                 closeup_coef=coef["closeup"]["coef"],
                 closeup_int=coef["closeup"]["intercept"],
                 closeup_thr=coef["closeup"]["thr"],
                 routed=np.array([1]), streams=np.array(list(base)))
    else:
        lr = LogisticRegression(max_iter=1000).fit(X, y)
        p = lr.predict_proba(X)[:, 1]
        scan = [(((p >= t).astype(int) == y).mean(), float(t))
                for t in np.arange(0.05, 0.96, 0.01)]
        acc, t = max(scan)
        np.savez(OUT / "ensemble_v3b.npz", coef=lr.coef_,
                 intercept=lr.intercept_, thr=t, routed=np.array([0]),
                 streams=np.array(list(base)))
        print("frozen acc", round(acc, 4), "thr", round(t, 2))
    (OUT / "ensemble_v2_report.json").write_text(json.dumps(
        {"val_aurocs": {"A": float(roc_auc_score(y, imgA)),
                        "B": float(roc_auc_score(y, imgB)),
                        "C": float(roc_auc_score(y, imgC)),
                        "D": float(roc_auc_score(y, imgD))},
         "variants": rep, "best": name, "routed": routed_flag,
         "streams": list(base)}, indent=2))
    # persist val streams for later analysis
    np.savez(OUT / "ensemble_val_v2.npz", imgA=imgA, imgB=imgB, imgC=imgC,
             imgD=imgD, y=y, routed=routed)


if __name__ == "__main__":
    main()
