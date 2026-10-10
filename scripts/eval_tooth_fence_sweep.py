"""Tooth-fence threshold sweep on VAL -> freeze -> TEST (honest protocol).

Fence variants: tooth boxes dilated by 10% of side; candidate survives if
mask∩tooth_union >= FENCE_MIN.  Grid on val: [0.02, 0.05, 0.10, 0.20] vs OFF.
Selection: max val instance F1 subject to recall drop <= 0.05 vs OFF.
Frozen config applied once to test.
Outputs: outputs/eka_toothdet/fence_sweep_val.json, fence_test_final.json
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

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from ultralytics import YOLO

from scripts.eval_ensemble_v3b import effnet_tta_probs, yolocls_probs
from scripts.eval_tooth_fence_ab import load_frozen
from scripts.v3_pipeline_lib import (BUCKETS, DATA, FCOLS, NMS_IOU, OUT,
                                     collect_split, make_verifier)
from src.inference.fusion import greedy_nms, rec_bin

IOU_MATCH = 0.5
THR_INST = 0.6
TOOTH_W = "outputs/eka_toothdet/weights/best.pt"
TOOTH_CONF = 0.25
DILATE = 0.10
GRID = [0.02, 0.05, 0.10, 0.20]


def make_fences(tdet, imgs, conf=TOOTH_CONF):
    fences = []
    stats = []
    for im in imgs:
        res = tdet.predict(im["image_path"], conf=conf, iou=0.6, imgsz=640,
                           device=0, max_det=64, verbose=False)[0]
        H, W = im["H"], im["W"]
        f = np.zeros((H, W), np.uint8)
        nb = 0
        if res.boxes is not None and len(res.boxes):
            for b in res.boxes.xyxy.cpu().numpy():
                bw, bh = b[2] - b[0], b[3] - b[1]
                x0 = max(0, int(b[0] - DILATE * bw))
                y0 = max(0, int(b[1] - DILATE * bh))
                x1 = min(W, int(b[2] + DILATE * bw))
                y1 = min(H, int(b[3] + DILATE * bh))
                if x1 > x0 and y1 > y0:
                    f[y0:y1, x0:x1] = 1
                    nb += 1
        fences.append(f)
        stats.append(nb)
    return fences, stats


def instance_metrics(imgs, fused_all, thr_inst, fences, fence_min):
    tp = fp = preds_tot = 0
    gtb = {b: 0 for b in BUCKETS}
    tpb = {b: 0 for b in BUCKETS}
    td = []
    killed = 0
    for im, fused, fence in zip(imgs, fused_all, fences):
        H, W = im["H"], im["W"]
        idxs = np.argwhere(fused >= thr_inst).ravel() if len(fused) else []
        recs = []
        for i in idxs:
            rect = im["rects"][i]
            P = np.zeros((H, W), np.uint8)
            P[rect["y0"]:rect["y0"] + rect["ch"],
              rect["x0"]:rect["x0"] + rect["cw"]] = im["bins"][i]
            if fence is not None and fence_min is not None:
                ov = int((P & (fence > 0)).sum()) / max(int(P.sum()), 1)
                if ov < fence_min:
                    killed += 1
                    continue
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
            "r_medium": round(tpb["medium"] / max(gtb["medium"], 1), 4),
            "r_large": round(tpb["large"] / max(gtb["large"], 1), 4),
            "fp_img": round(fp / len(imgs), 3),
            "killed": killed}


def image_metrics(imgs, fused_all, m3, mu, sg, lr_img, thr_img, imgB, imgC,
                  ys, fences, fence_min):
    imgA = []
    for im, fused, fence in zip(imgs, fused_all, fences):
        f = im["feats"]
        if not f:
            imgA.append(0.0)
            continue
        X = np.array([[x[c] for c in FCOLS] for x in f], float)
        pr = m3.predict_proba((X - mu) / sg)[:, 1]
        if fence is not None and fence_min is not None:
            ok = []
            for i in range(len(f)):
                rect = im["rects"][i]
                P = np.zeros((im["H"], im["W"]), np.uint8)
                P[rect["y0"]:rect["y0"] + rect["ch"],
                  rect["x0"]:rect["x0"] + rect["cw"]] = im["bins"][i]
                ok.append(int((P & (fence > 0)).sum()) / max(int(P.sum()), 1)
                          >= fence_min)
            imgA.append(float(pr[ok].max()) if any(ok) else 0.0)
        else:
            imgA.append(float(pr.max()))
    prob = lr_img.predict_proba(np.stack([np.array(imgA), imgB, imgC], 1))[:, 1]
    pred = (prob >= thr_img).astype(int)
    tp = int(((pred == 1) & (ys == 1)).sum())
    fp_i = int(((pred == 1) & (ys == 0)).sum())
    fn_i = int(((pred == 0) & (ys == 1)).sum())
    tn_i = int(((pred == 0) & (ys == 0)).sum())
    return {"accuracy": round((tp + tn_i) / len(ys), 4),
            "sensitivity": round(tp / max(tp + fn_i, 1), 4),
            "specificity": round(tn_i / max(tn_i + fp_i, 1), 4),
            "roc_auc": round(float(roc_auc_score(ys, prob)), 4)}


def run_split(m3, mu, sg, lr_img, thr_img, s1m, s2m, vb, tdet, man, split):
    imgs = collect_split(s1m, s2m, man, split, with_verifier=True, vb=vb,
                         log_every=350)
    ys = np.array([int(im["has"]) for im in imgs])
    fences, stats = make_fences(tdet, imgs)
    fused_all = []
    for im in imgs:
        f = im["feats"]
        if f:
            X = np.array([[x[c] for c in FCOLS] for x in f], float)
            fused_all.append(m3.predict_proba((X - mu) / sg)[:, 1])
        else:
            fused_all.append(np.zeros(0))
    imgB = effnet_tta_probs(man, split)
    imgC = yolocls_probs(split)
    rows = {"OFF": {"inst": instance_metrics(imgs, fused_all, THR_INST,
                                             [None] * len(imgs), None)}}
    for fm in GRID:
        rows[f"ON_{fm}"] = {"inst": instance_metrics(imgs, fused_all, THR_INST,
                                                     fences, fm)}
    for k, v in rows.items():
        print(split, k, v["inst"], flush=True)
    return imgs, ys, fences, fused_all, imgB, imgC, rows


def main() -> None:
    m3, (mu, sg), lr_img, thr_img = load_frozen()
    s1m = YOLO("outputs/calculus_v3c_stage1/weights/best.pt")
    s2m = YOLO("outputs/calculus_v3b_stage2/weights/best.pt")
    vb = make_verifier()
    tdet = YOLO(TOOTH_W)
    man = pd.read_csv(DATA / "split_manifest_v3.csv")

    print("=== VAL sweep ===", flush=True)
    vi, vy, vf, vfu, vB, vC, vrows = run_split(
        m3, mu, sg, lr_img, thr_img, s1m, s2m, vb, tdet, man, "val")
    base_f1 = vrows["OFF"]["inst"]["f1"]
    base_rec = vrows["OFF"]["inst"]["recall"]
    cand = []
    for fm in GRID:
        r = vrows[f"ON_{fm}"]["inst"]
        if base_rec - r["recall"] <= 0.05:
            cand.append((r["f1"], fm))
    if cand:
        best_f1, best_fm = max(cand)
        choice = f"ON_{best_fm}" if best_f1 > base_f1 else "OFF"
    else:
        choice = "OFF"
    print("CHOICE:", choice, flush=True)
    (Path("outputs/eka_toothdet/fence_sweep_val.json").write_text(json.dumps(
        {"grid": {k: v["inst"] for k, v in vrows.items()},
         "choice": choice, "rule": "max F1 s.t. recall drop<=0.05"}, indent=2)))

    print("=== TEST (frozen choice) ===", flush=True)
    ti, ty, tf, tfu, tB, tC, trows = run_split(
        m3, mu, sg, lr_img, thr_img, s1m, s2m, vb, tdet, man, "test")
    fm_used = None if choice == "OFF" else float(choice.split("_")[1])
    out = {"choice": choice, "val": {k: v["inst"] for k, v in vrows.items()},
           "test": {k: v["inst"] for k, v in trows.items()}}
    (Path("outputs/eka_toothdet/fence_test_final.json").write_text(
        json.dumps(out, indent=2)))
    print("test OFF:", trows["OFF"]["inst"])
    if choice != "OFF":
        print(f"test {choice}:", trows[choice]["inst"])


if __name__ == "__main__":
    main()
