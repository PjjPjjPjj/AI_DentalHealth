"""V3 calibration on val (out-of-sample for all base models).

Fits:
  - M3: instance fusion logistic on FCOLS (StandardScaler), fit on val
  - image-level: 2-feature logistic on [max fused prob, imgclf prob], fit on val
Chooses:
  - instance threshold thr_inst: max F1 on val
  - image threshold thr_img: max accuracy on val (also reports max-F1 point)
Saves: outputs/calculus_v3_cls/fusion_v3.npz, chosen_v3.json
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
from sklearn.preprocessing import StandardScaler
from ultralytics import YOLO

from scripts.v3_pipeline_lib import (BUCKETS, DATA, FCOLS, OUT, collect_split,
                                     make_imgclf, make_verifier, s1_pass)

np.random.seed(42)


def select_and_score(im, thr_inst, m3, scaler, nms_iou=0.5):
    from src.inference.fusion import greedy_nms, rec_bin
    feats = im["feats"]
    if not feats:
        return [], 0.0
    X = np.array([[f[c] for c in FCOLS] for f in feats], float)
    fused = m3.predict_proba(scaler.transform(X))[:, 1]
    idxs = np.argwhere(fused >= thr_inst).ravel()
    if not len(idxs):
        return [], float(fused.max())
    recs = []
    for i in idxs:
        rect = im["rects"][i]
        recs.append({"x0": rect["x0"], "y0": rect["y0"], "cw": rect["cw"],
                     "ch": rect["ch"], "bin": im["bins"][i],
                     "score": feats[i]["s2_conf"], "i": int(i)})
    H, W = im["H"], im["W"]
    keep = greedy_nms(recs, "mask_nms", nms_iou, H, W)
    preds = [rec_bin(k, H, W) for k in keep]
    return preds, float(fused.max())


def instance_metrics(imgs, thr_inst, m3, scaler):
    tp = fp = preds_tot = 0
    gtb = {b: 0 for b in BUCKETS}
    tpb = {b: 0 for b in BUCKETS}
    tdice = []
    fp_imgs = 0
    for im in imgs:
        preds, _ = select_and_score(im, thr_inst, m3, scaler)
        preds_tot += len(preds)
        matched_g = set()
        used_p = set()
        for pi, P in enumerate(preds):
            best = (0.0, None)
            for gi, G in enumerate(im["gts"]):
                inter = int((P & G["mask"]).sum())
                uni = int((P | G["mask"]).sum())
                io = inter / uni if uni else 0.0
                if io > best[0]:
                    best = (io, gi)
            if best[0] >= 0.5 and best[1] not in matched_g:
                matched_g.add(best[1]); used_p.add(pi)
                G = im["gts"][best[1]]
                inter = int((P & G["mask"]).sum())
                dc = 2 * inter / (int(P.sum()) + int(G["mask"].sum()))
                tdice.append(dc)
                tpb[G["bucket"]] += 1
        tp += len(matched_g)
        fp += len(preds) - len(used_p)
        fp_imgs += (len(preds) - len(used_p)) > 0
        for G in im["gts"]:
            gtb[G["bucket"]] += 1
    rec = tp / max(sum(gtb.values()), 1)
    prec = tp / max(preds_tot, 1)
    return {"precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
            "r_small": round(tpb["small"] / max(gtb["small"], 1), 4),
            "r_medium": round(tpb["medium"] / max(gtb["medium"], 1), 4),
            "r_large": round(tpb["large"] / max(gtb["large"], 1), 4),
            "fp_img": round(fp / len(imgs), 3)}


def main() -> None:
    from src.inference.fusion import greedy_nms  # noqa
    s1m = YOLO("outputs/calculus_v3_stage1/weights/best.pt")
    s2m = YOLO("outputs/calculus_v3_stage2/weights/best.pt")
    vb = make_verifier()
    clf = make_imgclf()
    man = pd.read_csv(DATA / "split_manifest_v3.csv")
    print("collecting val ...", flush=True)
    imgs = collect_split(s1m, s2m, man, "val", with_verifier=True, vb=vb,
                         with_imgclf=True, clf=clf)

    # instance labels from stored bins vs GT
    all_feats = [f for im in imgs for f in im["feats"]]
    all_y = []
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
            all_y.append(int(best >= 0.5))
    X = np.array([[f[c] for c in FCOLS] for f in all_feats], float)
    y = np.array(all_y)
    print(f"fusion fit: n={len(y)} pos={y.sum()}")
    scaler = StandardScaler().fit(X)
    m3 = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
        scaler.transform(X), y)

    thr_scan = np.arange(0.05, 0.91, 0.05)
    inst_rows = []
    for t in thr_scan:
        inst_rows.append({"thr": round(float(t), 2),
                          **instance_metrics(imgs, float(t), m3, scaler)})
    inst_df = pd.DataFrame(inst_rows)
    inst_df.to_csv(OUT / "val_thr_scan.csv", index=False)
    best_inst = inst_df.loc[inst_df.f1.idxmax()]
    print("best instance thr:", best_inst.to_dict())

    # image-level scores
    imgA, imgB, ys = [], [], []
    for im in imgs:
        feats = im["feats"]
        if feats:
            Xf = np.array([[f[c] for c in FCOLS] for f in feats], float)
            imgA.append(float(m3.predict_proba(scaler.transform(Xf))[:, 1].max()))
        else:
            imgA.append(0.0)
        imgB.append(im["imgclf_prob"])
        ys.append(int(im["has"]))
    imgA, imgB, ys = np.array(imgA), np.array(imgB), np.array(ys)
    Xi = np.stack([imgA, imgB], 1)
    imglr = LogisticRegression(max_iter=1000).fit(Xi, ys)
    img_prob = imglr.predict_proba(Xi)[:, 1]
    auroc = roc_auc_score(ys, img_prob)
    ap = average_precision_score(ys, img_prob)
    scan = []
    for t in np.arange(0.05, 0.96, 0.01):
        pred = (img_prob >= t).astype(int)
        tp = int(((pred == 1) & (ys == 1)).sum())
        fp = int(((pred == 1) & (ys == 0)).sum())
        fn = int(((pred == 0) & (ys == 1)).sum())
        tn = int(((pred == 0) & (ys == 0)).sum())
        scan.append({"thr": round(float(t), 2), "acc": (tp + tn) / len(ys),
                     "sens": tp / max(tp + fn, 1), "spec": tn / max(tn + fp, 1),
                     "precision": tp / max(tp + fp, 1)})
    scan_df = pd.DataFrame(scan)
    scan_df.to_csv(OUT / "val_img_thr_scan.csv", index=False)
    bacc = scan_df.loc[scan_df.acc.idxmax()]
    bf1 = scan_df.assign(f1=2 * scan_df.precision * scan_df.sens /
                         (scan_df.precision + scan_df.sens).clip(lower=1e-9)
                         ).sort_values("f1").iloc[-1]
    print("max-acc point:", bacc.to_dict())
    print("max-f1 point:", bf1.to_dict())

    np.savez(OUT / "fusion_v3.npz", mu=scaler.mean_, sg=scaler.scale_,
             coef=m3.coef_, intercept=m3.intercept_,
             img_coef=imglr.coef_, img_intercept=imglr.intercept_)
    chosen = {
        "thr_inst": float(best_inst["thr"]),
        "thr_img_acc": float(bacc["thr"]),
        "val_image_acc_maxacc": float(bacc["acc"]),
        "val_sens_maxacc": float(bacc["sens"]),
        "val_spec_maxacc": float(bacc["spec"]),
        "val_auroc": round(float(auroc), 4), "val_ap": round(float(ap), 4),
        "val_inst": best_inst.to_dict(),
        "fc_weight_imgA": float(imglr.coef_[0][0]),
        "fc_weight_imgB": float(imglr.coef_[0][1]),
        "n_val": len(imgs), "n_pos": int(ys.sum()),
    }
    (OUT / "chosen_v3.json").write_text(json.dumps(chosen, indent=2))
    print(json.dumps(chosen, indent=2))


if __name__ == "__main__":
    main()
