"""Cascade (rescue-mode) tuning on val with the v5 Stage1.

One collect at c1=0.005 (superset of c1=0.02).  Offline cascade:
  - normal: candidates with s1_conf>=0.02, M3 fused>=0.6 -> instances (imgA as usual)
  - trigger: imgB (EffNet TTA) >= tau AND no normal instance survived
  - rescue:  re-admit candidates with s1_conf>=0.005 and fused >= thr_rescue
Grid over (tau, thr_rescue) on val image-level acc / sens / spec.
Outputs: outputs/calculus_v5_cls/cascade_val.json
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
from sklearn.linear_model import LogisticRegression
from ultralytics import YOLO

import scripts.v3_pipeline_lib as _lib
_lib.OUT = Path("outputs/calculus_v5_cls")
_lib.C1 = 0.005
from scripts.v3_pipeline_lib import (FCOLS, collect_split, make_verifier,
                                     s1_pass)  # noqa
from scripts.eval_ensemble_v3b import effnet_tta_probs, yolocls_probs

OUT = _lib.OUT
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    z = np.load("outputs/calculus_v4_cls/fusion_v3.npz")
    m3 = LogisticRegression(max_iter=1000, class_weight="balanced")
    m3.coef_, m3.intercept_, m3.classes_ = z["coef"], z["intercept"], np.array([0, 1])
    mu, sg = z["mu"], z["sg"]
    e = np.load("outputs/calculus_v4_cls/ensemble_v3b.npz")
    lr_img = LogisticRegression(max_iter=1000)
    lr_img.coef_, lr_img.intercept_, lr_img.classes_ = e["coef"], e["intercept"], np.array([0, 1])
    thr_img = float(e["thr"])

    s1m = YOLO("outputs/calculus_v5_stage1/weights/best.pt")
    s2m = YOLO("outputs/calculus_v3b_stage2/weights/best.pt")
    vb = make_verifier()
    man = pd.read_csv(_lib.DATA / "split_manifest_v3.csv")
    print("collecting val @c1=0.005 ...", flush=True)
    imgs = collect_split(s1m, s2m, man, "val", with_verifier=True, vb=vb,
                         log_every=200)
    ys = np.array([int(im["has"]) for im in imgs])

    for im in imgs:
        f = im["feats"]
        if f:
            X = np.array([[x[c] for c in FCOLS] for x in f], float)
            im["fused"] = m3.predict_proba((X - mu) / sg)[:, 1]
            im["s1c"] = np.array([x["s1_conf"] for x in f])
        else:
            im["fused"] = np.zeros(0)
            im["s1c"] = np.zeros(0)
    imgB = effnet_tta_probs(man, "val")
    imgC = yolocls_probs("val")
    np.save(OUT / "cascade_val_cache.npy",
            np.array([1]))  # marker

    def eval_config(tau, thr_rescue):
        tp = fp = fn = tn = 0
        rescued_imgs = 0
        rescued_tp = rescued_fp = 0
        for im, b, c in zip(imgs, imgB, imgC):
            fused, s1c = im["fused"], im["s1c"]
            normal_ok = ((s1c >= 0.02) & (fused >= 0.6)) if len(fused) else \
                np.zeros(0, bool)
            has_normal = bool(normal_ok.any())
            pred_pos = has_normal
            if not has_normal and b >= tau:
                rescued_imgs += 1
                rescue_ok = (fused >= thr_rescue) if len(fused) else \
                    np.zeros(0, bool)
                if rescue_ok.any():
                    pred_pos = True
                    rtp = any(
                        True for gi, G in enumerate(im["gts"])
                        if True) and _gt_hit(im, rescue_ok)
                    rescued_tp += int(rtp)
                    rescued_fp += int(not rtp)
            y = int(im["has"])
            if pred_pos and y == 1:
                tp += 1
            elif pred_pos and y == 0:
                fp += 1
            elif not pred_pos and y == 1:
                fn += 1
            else:
                tn += 1
        n = len(ys)
        return {"tau": tau, "thr_rescue": thr_rescue,
                "acc": round((tp + tn) / n, 4),
                "sens": round(tp / max(tp + fn, 1), 4),
                "spec": round(tn / max(tn + fp, 1), 4),
                "rescued_imgs": rescued_imgs,
                "rescued_tp": rescued_tp, "rescued_fp": rescued_fp}

    def _gt_hit(im, ok_mask):
        import cv2
        idxs = np.argwhere(ok_mask).ravel()
        H, W = im["H"], im["W"]
        best = 0.0
        for i in idxs:
            rect = im["rects"][i]
            P = np.zeros((H, W), np.uint8)
            P[rect["y0"]:rect["y0"] + rect["ch"],
              rect["x0"]:rect["x0"] + rect["cw"]] = im["bins"][i]
            for G in im["gts"]:
                inter = int((P & G["mask"]).sum())
                uni = int((P | G["mask"]).sum())
                if uni:
                    best = max(best, inter / uni)
        return best >= 0.5

    base = eval_config(99.0, 99.0)  # no rescue
    print("baseline (no rescue):", base, flush=True)
    rows = [base]
    for tau in (0.4, 0.5, 0.6, 0.7):
        for tr in (0.25, 0.30, 0.35, 0.40, 0.45):
            rows.append(eval_config(tau, tr))
    for r in rows[1:]:
        print(r, flush=True)
    # choose: max acc s.t. sens > baseline sens
    base_sens = base["sens"]
    cand = [r for r in rows[1:] if r["sens"] > base_sens and
            r["spec"] >= base["spec"] - 0.01]
    best = max(cand, key=lambda r: r["acc"]) if cand else None
    print("CHOSEN:", best)
    json.dump({"baseline": base, "grid": rows[1:], "chosen": best},
              open(OUT / "cascade_val.json", "w"), indent=1)


if __name__ == "__main__":
    main()
