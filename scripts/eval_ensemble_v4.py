"""Ensemble image-level decision for v3b: imgA (pipeline max fused prob),
imgB (EffNet imgclf + hflip TTA), imgC (YOLO11m-cls). Fit logistic on VAL,
threshold at max val accuracy, optionally apply frozen to test.

Usage:
  python scripts/eval_ensemble_v3b.py --fit      # val fit + chosen_ensemble.json
  python scripts/eval_ensemble_v3b.py --test     # frozen final test
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision import models
from torchvision.transforms import functional as TF
from ultralytics import YOLO

import scripts.v3_pipeline_lib as _lib
from scripts.v3_pipeline_lib import (DATA, collect_split, make_verifier)
_lib.OUT = Path("outputs/calculus_v4_cls")
OUT = _lib.OUT
from pathlib import Path as _P

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def effnet_tta_probs(man: pd.DataFrame, split: str) -> np.ndarray:
    m = models.efficientnet_v2_s(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    m.load_state_dict(torch.load(OUT / "imgclf" / "best.pt", weights_only=False))
    m = m.to(0).eval()
    rows = man[man.split == split].reset_index(drop=True)
    probs = []
    with torch.no_grad():
        for i in range(0, len(rows), 32):
            batch = rows.iloc[i:i + 32]
            xs = []
            for _, r in batch.iterrows():
                img = cv2.cvtColor(cv2.imread(r.image_path), cv2.COLOR_BGR2RGB)
                t = TF.to_tensor(Image.fromarray(img))
                t = TF.resize(t, (448, 448), TF.InterpolationMode.BILINEAR)
                t = TF.normalize(t, MEAN, STD)
                xs += [t, torch.flip(t, [2])]
            out = m(torch.stack(xs).to(0)).squeeze(1).sigmoid() \
                .view(-1, 2).cpu().numpy()
            probs += out.mean(1).tolist()
    return np.array(probs)


def yolocls_probs(split: str) -> np.ndarray:
    model = YOLO("outputs/calculus_v3b_yolocls/weights/best.pt")
    man = pd.read_csv(DATA / "split_manifest_v3.csv")
    rows = man[man.split == split].reset_index(drop=True)
    probs = []
    for i in range(0, len(rows), 64):
        batch = rows.iloc[i:i + 64]
        res = model.predict(list(batch.image_path), imgsz=448, device=0,
                            verbose=False, retina_masks=False)
        for r in res:
            p = float(r.probs.data[1].item()) if r.probs is not None else 0.0
            probs.append(p)
        print(f"  yolocls {split} {min(i + 64, len(rows))}/{len(rows)}",
              flush=True)
    return np.array(probs)


def pipeline_imgA(man: pd.DataFrame, split: str) -> tuple[np.ndarray, list]:
    s1m = YOLO("outputs/calculus_v4_stage1/weights/best.pt")
    s2m = YOLO("outputs/calculus_v3b_stage2/weights/best.pt")
    vb = make_verifier()
    imgs = collect_split(s1m, s2m, man, split, with_verifier=True, vb=vb)
    z = np.load(OUT / "fusion_v3.npz")
    m3 = LogisticRegression(max_iter=1000, class_weight="balanced")
    m3.coef_, m3.intercept_, m3.classes_ = z["coef"], z["intercept"], np.array([0, 1])
    mu, sg = z["mu"], z["sg"]
    from scripts.v3_pipeline_lib import FCOLS
    imgA = []
    for im in imgs:
        feats = im["feats"]
        if feats:
            X = np.array([[f[c] for c in FCOLS] for f in feats], float)
            imgA.append(float(m3.predict_proba((X - mu) / sg)[:, 1].max()))
        else:
            imgA.append(0.0)
    return np.array(imgA), imgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    man = pd.read_csv(DATA / "split_manifest_v3.csv")

    if args.fit:
        print("collecting val evidence ...", flush=True)
        imgA, _ = pipeline_imgA(man, "val")
        imgB = effnet_tta_probs(man, "val")
        imgC = yolocls_probs("val")
        y = man[man.split == "val"].has_calculus.values.astype(int)
        for name, sc in (("imgA", imgA), ("imgB", imgB), ("imgC", imgC)):
            print(name, "auroc", round(float(roc_auc_score(y, sc)), 4))
        Xi = np.stack([imgA, imgB, imgC], 1)
        lr = LogisticRegression(max_iter=1000).fit(Xi, y)
        prob = lr.predict_proba(Xi)[:, 1]
        auc = roc_auc_score(y, prob)
        scan = [(((prob >= t).astype(int) == y).mean(), float(t))
                for t in np.arange(0.05, 0.96, 0.01)]
        acc, thr = max(scan)
        print("ENSEMBLE val AUROC", round(auc, 4), "best acc", round(acc, 4),
              "thr", round(thr, 2))
        print("coefs:", lr.coef_.round(3).tolist(), "int:", float(lr.intercept_[0]))
        np.savez(OUT / "ensemble_v3b.npz", coef=lr.coef_,
                 intercept=lr.intercept_, thr=thr)
        pd.DataFrame({"y": y, "imgA": imgA, "imgB": imgB, "imgC": imgC,
                      "prob": prob}).to_csv(OUT / "ensemble_val.csv", index=False)

    if args.test:
        z = np.load(OUT / "ensemble_v3b.npz")
        lr = LogisticRegression(max_iter=1000)
        lr.coef_, lr.intercept_, lr.classes_ = z["coef"], z["intercept"], np.array([0, 1])
        thr = float(z["thr"])
        print("collecting test evidence ...", flush=True)
        imgA, imgs = pipeline_imgA(man, "test")
        imgB = effnet_tta_probs(man, "test")
        imgC = yolocls_probs("test")
        y = man[man.split == "test"].has_calculus.values.astype(int)
        prob = lr.predict_proba(np.stack([imgA, imgB, imgC], 1))[:, 1]
        pred = (prob >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        n = len(y)
        print(json.dumps({
            "thr": round(thr, 2), "accuracy": round((tp + tn) / n, 4),
            "sensitivity": round(tp / max(tp + fn, 1), 4),
            "specificity": round(tn / max(tn + fp, 1), 4),
            "precision": round(tp / max(tp + fp, 1), 4),
            "f1": round(2 * tp / max(2 * tp + fp + fn, 1), 4),
            "roc_auc": round(float(roc_auc_score(y, prob)), 4),
            "pr_auc": round(float(average_precision_score(y, prob)), 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}, indent=2))
        df = pd.DataFrame({"image_id": [im["image_id"] for im in imgs],
                           "domain": [im["domain"] for im in imgs],
                           "y": y, "prob": prob.round(4), "pred": pred})
        rows = []
        for dom, sub in df.groupby("domain"):
            rows.append({"domain": dom, "n": len(sub),
                         "acc": round(float((sub.y == sub.pred).mean()), 4),
                         "sens": round(float(((sub.y == 1) & (sub.pred == 1)).sum()
                                             / max(int(sub.y.sum()), 1)), 4),
                         "pos": int(sub.y.sum())})
        print(pd.DataFrame(rows).to_string(index=False))
        df.to_csv(OUT / "ensemble_test_preds.csv", index=False)
        pd.DataFrame(rows).to_csv(OUT / "ensemble_test_domains.csv", index=False)
        (OUT / "final_image_ensemble_metrics.json").write_text(json.dumps(
            {"threshold": round(thr, 2),
             "accuracy": round((tp + tn) / n, 4),
             "sensitivity": round(tp / max(tp + fn, 1), 4),
             "specificity": round(tn / max(tn + fp, 1), 4),
             "roc_auc": round(float(roc_auc_score(y, prob)), 4),
             "pr_auc": round(float(average_precision_score(y, prob)), 4),
             "per_domain": rows}, indent=2))


if __name__ == "__main__":
    main()
