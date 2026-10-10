"""FINAL TEST v5-fix4: v5 cascade + ADAPTIVE stage2 window (S2_ADAPTIVE).

Only change vs final_test_v5: _lib.S2_ADAPTIVE=True (s2 crop side grows with
the s1 box when max(bw,bh) >= 0.55*160; clamp [160, max(W,H)]). Stage2/verifier
weights untouched; M3 + image ensemble refit on v5 val as before.

Output: outputs/calculus_v5_fix4/final_fix4_metrics.json
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
from sklearn.metrics import average_precision_score, roc_auc_score
from ultralytics import YOLO

import scripts.v3_pipeline_lib as _lib
_lib.OUT = Path("outputs/calculus_v5_fix4")
_lib.C1 = 0.005
_lib.S2_ADAPTIVE = True
_lib.S2_TTA = True
from scripts.v3_pipeline_lib import BUCKETS, DATA, FCOLS, collect_split, make_verifier
from scripts.eval_ensemble_v3b import effnet_tta_probs, yolocls_probs
from src.inference.fusion import greedy_nms, rec_bin

OUT = _lib.OUT
IOU_MATCH = 0.5
THR_STD = 0.6
THR_LARGE = 0.45
LARGE_EDGE = 0.01565
TAU = 0.4
THR_RESCUE = 0.25
NMS_IOU = 0.5


def fit_m3_on_val(imgs):
    from sklearn.preprocessing import StandardScaler
    X, y = [], []
    for im in imgs:
        H, W = im["H"], im["W"]
        for i, f in enumerate(im["feats"]):
            rect = im["rects"][i]
            P = np.zeros((H, W), np.uint8)
            P[rect["y0"]:rect["y0"] + rect["ch"],
              rect["x0"]:rect["x0"] + rect["cw"]] = im["bins"][i]
            best = 0.0
            for G in im["gts"]:
                inter = int((P & G["mask"]).sum())
                uni = int((P | G["mask"]).sum())
                if uni:
                    best = max(best, inter / uni)
            X.append([f[c] for c in FCOLS]); y.append(int(best >= 0.5))
    X = np.array(X, float); y = np.array(y)
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
        sc.transform(X), y)
    np.savez(OUT / "fusion_v3.npz", mu=sc.mean_, sg=sc.scale_,
             coef=lr.coef_, intercept=lr.intercept_)
    m3 = LogisticRegression(max_iter=1000, class_weight="balanced")
    m3.coef_, m3.intercept_, m3.classes_ = lr.coef_, lr.intercept_, np.array([0, 1])
    return m3, (sc.mean_, sc.scale_)


def inst_metrics(imgs, mode):
    tp = fp = preds_tot = rescued = 0
    gtb = {b: 0 for b in BUCKETS}
    tpb = {b: 0 for b in BUCKETS}
    td = []
    for im in imgs:
        H, W = im["H"], im["W"]
        fused, s1c = im["fused"], im["s1c"]
        normal_ok = ((s1c >= 0.02) & (fused >= np.where(
            np.array([f["s2_area_ratio"] for f in im["feats"]]) >= LARGE_EDGE,
            THR_LARGE, THR_STD))) if len(fused) else np.zeros(0, bool)
        selected = list(np.argwhere(normal_ok).ravel())
        rescue_used = False
        if mode == "cascade" and not selected and im["imgB"] >= TAU:
            rescue_used = True
            rescue_ok = fused >= THR_RESCUE if len(fused) else np.zeros(0, bool)
            selected = list(np.argwhere(rescue_ok).ravel())
        rescued += int(rescue_used)
        recs = []
        for i in selected:
            rect = im["rects"][i]
            recs.append({"x0": rect["x0"], "y0": rect["y0"], "cw": rect["cw"],
                         "ch": rect["ch"], "bin": im["bins"][i],
                         "score": im["feats"][i]["s2_conf"]})
        preds = [rec_bin(k, H, W) for k in greedy_nms(recs, "mask_nms",
                                                      NMS_IOU, H, W)] if recs else []
        preds_tot += len(preds)
        pairs = []
        for pi, P in enumerate(preds):
            for gi, G in enumerate(im["gts"]):
                inter = int((P & G["mask"]).sum())
                uni = int((P | G["mask"]).sum())
                io = inter / uni if uni else 0.0
                if io >= IOU_MATCH:
                    pairs.append((io, pi, gi))
        pairs.sort(reverse=True)
        used_p, used_g = set(), set()
        for io, pi, gi in pairs:
            if pi in used_p or gi in used_g:
                continue
            used_p.add(pi); used_g.add(gi)
            P, G = preds[pi], im["gts"][gi]
            inter = int((P & G["mask"]).sum())
            td.append(2 * inter / max(int(P.sum()) + int(G["mask"].sum()), 1))
            tpb[G["bucket"]] += 1
        for G in im["gts"]:
            gtb[G["bucket"]] += 1
        tp += len(used_g)
        fp += len(preds) - len(used_p)
    rec = tp / max(sum(gtb.values()), 1)
    prec = tp / max(preds_tot, 1)
    return {"precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
            "r_small": round(tpb["small"] / max(gtb["small"], 1), 4),
            "r_medium": round(tpb["medium"] / max(gtb["small"] + gtb["medium"], 1), 4)
            if False else round(tpb["medium"] / max(gtb["medium"], 1), 4),
            "r_large": round(tpb["large"] / max(gtb["large"], 1), 4),
            "tp_dice": round(float(np.mean(td)), 4) if td else 0.0,
            "fp_img": round(fp / len(imgs), 3),
            "pred_img": round(preds_tot / len(imgs), 3)}, rescued


def main() -> None:
    s1m = YOLO("outputs/calculus_v5_stage1/weights/best.pt")
    s2m = YOLO("outputs/calculus_v3b_stage2/weights/best.pt")
    vb = make_verifier()
    man = pd.read_csv(DATA / "split_manifest_v3.csv")

    print("collecting val (v5-fix4) ...", flush=True)
    val = collect_split(s1m, s2m, man, "val", with_verifier=True, vb=vb,
                        log_every=350)
    val_y = np.array([int(im["has"]) for im in val])
    m3, (mu, sg) = fit_m3_on_val(val)
    for im in val:
        f = im["feats"]
        if f:
            X = np.array([[x[c] for c in FCOLS] for x in f], float)
            im["fused"] = m3.predict_proba((X - mu) / sg)[:, 1]
        else:
            im["fused"] = np.zeros(0)
    valA = np.array([float(im["fused"].max()) if len(im["fused"]) else 0.0
                     for im in val])
    valB = effnet_tta_probs(man, "val")
    valC = yolocls_probs("val")
    lr_refit = LogisticRegression(max_iter=1000).fit(
        np.stack([valA, valB, valC], 1), val_y)
    np.savez(OUT / "ensemble_v3b.npz", coef=lr_refit.coef_,
             intercept=lr_refit.intercept_,
             thr=0.38, routed=np.array([0]), streams=np.array(["A", "B", "C"]))
    lr_img, thr_img = lr_refit, 0.38

    print("collecting test (v5-fix4, c1=0.005) ...", flush=True)
    imgs = collect_split(s1m, s2m, man, "test", with_verifier=True, vb=vb,
                         log_every=350)
    ys = np.array([int(im["has"]) for im in imgs])
    imgB = effnet_tta_probs(man, "test")
    imgC = yolocls_probs("test")
    for im, b in zip(imgs, imgB):
        im["imgB"] = float(b)
        f = im["feats"]
        if f:
            X = np.array([[x[c] for c in FCOLS] for x in f], float)
            im["fused"] = m3.predict_proba((X - mu) / sg)[:, 1]
            im["s1c"] = np.array([x["s1_conf"] for x in f])
        else:
            im["fused"] = np.zeros(0)
            im["s1c"] = np.zeros(0)

    inst_nocascade, _ = inst_metrics(imgs, "normal")
    inst_cascade, n_rescued = inst_metrics(imgs, "cascade")

    imgA = []
    for im in imgs:
        f = im["feats"]
        if not f:
            imgA.append(0.0)
            continue
        ok_norm = ((im["s1c"] >= 0.02) & (im["fused"] >= np.where(
            np.array([x["s2_area_ratio"] for x in f]) >= LARGE_EDGE,
            THR_LARGE, THR_STD)))
        if ok_norm.any():
            imgA.append(float(im["fused"][ok_norm].max()))
        elif im["imgB"] >= TAU and (im["fused"] >= THR_RESCUE).any():
            imgA.append(float(im["fused"][im["fused"] >= THR_RESCUE].max()))
        else:
            imgA.append(0.0)
    imgA = np.array(imgA)
    prob = lr_img.predict_proba(np.stack([imgA, imgB, imgC], 1))[:, 1]
    pred = (prob >= thr_img).astype(int)
    tp = int(((pred == 1) & (ys == 1)).sum())
    fp_i = int(((pred == 1) & (ys == 0)).sum())
    fn_i = int(((pred == 0) & (ys == 1)).sum())
    tn_i = int(((pred == 0) & (ys == 0)).sum())
    img_m = {"threshold": thr_img,
             "accuracy": round((tp + tn_i) / len(ys), 4),
             "sensitivity": round(tp / max(tp + fn_i, 1), 4),
             "specificity": round(tn_i / max(tn_i + fp_i, 1), 4),
             "precision": round(tp / max(tp + fp_i, 1), 4),
             "f1": round(2 * tp / max(2 * tp + fp_i + fn_i, 1), 4),
             "roc_auc": round(float(roc_auc_score(ys, prob)), 4),
             "pr_auc": round(float(average_precision_score(ys, prob)), 4)}
    out = {"frozen": {"thr_std": THR_STD, "thr_large": THR_LARGE,
                      "cascade_tau": TAU, "thr_rescue": THR_RESCUE,
                      "thr_img": thr_img, "s1": "v5", "s2_window": "adaptive+tta_cond"},
           "instance_no_cascade": inst_nocascade,
           "instance_cascade": inst_cascade,
           "rescued_images": n_rescued,
           "image": img_m, "test_n": len(imgs)}
    (OUT / "final_fix4_metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
