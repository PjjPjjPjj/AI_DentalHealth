"""Tooth-fence A/B on the frozen Oral-Mamba v3c pipeline.

OFF: M3 (thr_inst=0.6) -> mask-NMS -> instance metrics; image prob = frozen
     ABC logistic on [imgA, imgB, imgC] (thr 0.44).
ON : candidates whose mask overlaps tooth-detector union < 20% are dropped
     BEFORE selection; imgA recomputed post-filter; same frozen heads.
Tooth boxes: outputs/eka_toothdet detector on the test image itself @640.

Outputs: outputs/eka_toothdet/ab_fence_test.json
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
from sklearn.metrics import average_precision_score, roc_auc_score
from ultralytics import YOLO

from scripts.eval_ensemble_v3b import effnet_tta_probs, yolocls_probs
from scripts.v3_pipeline_lib import (BUCKETS, DATA, FCOLS, NMS_IOU, OUT,
                                     collect_split, make_verifier)
from src.inference.fusion import greedy_nms, rec_bin

IOU_MATCH = 0.5
THR_INST = 0.6
TOOTH_W = "outputs/eka_toothdet/weights/best.pt"
TOOTH_CONF = 0.25
FENCE_MIN = 0.20


def load_frozen():
    z = np.load(OUT / "fusion_v3.npz")
    m3 = LogisticRegression(max_iter=1000, class_weight="balanced")
    m3.coef_, m3.intercept_, m3.classes_ = z["coef"], z["intercept"], np.array([0, 1])
    mu, sg = z["mu"], z["sg"]
    e = np.load(OUT / "ensemble_v3b.npz")
    lr = LogisticRegression(max_iter=1000)
    lr.coef_, lr.intercept_, lr.classes_ = e["coef"], e["intercept"], np.array([0, 1])
    return m3, (mu, sg), lr, float(e["thr"])


def instance_metrics(imgs, fused_all, thr_inst, fences=None):
    per = []
    tp = fp = preds_tot = 0
    gtb = {b: 0 for b in BUCKETS}
    tpb = {b: 0 for b in BUCKETS}
    td = []
    killed = 0
    for im, fused, fence in zip(imgs, fused_all,
                                fences if fences is not None
                                else [None] * len(imgs)):
        H, W = im["H"], im["W"]
        idxs = np.argwhere(fused >= thr_inst).ravel() if len(fused) else []
        recs = []
        drop = 0
        for i in idxs:
            if fence is not None:
                rect = im["rects"][i]
                P = np.zeros((H, W), np.uint8)
                P[rect["y0"]:rect["y0"] + rect["ch"],
                  rect["x0"]:rect["x0"] + rect["cw"]] = im["bins"][i]
                ov = int((P & (fence > 0)).sum()) / max(int(P.sum()), 1)
                if ov < FENCE_MIN:
                    drop += 1
                    continue
            rect = im["rects"][i]
            recs.append({"x0": rect["x0"], "y0": rect["y0"], "cw": rect["cw"],
                         "ch": rect["ch"], "bin": im["bins"][i],
                         "score": im["feats"][i]["s2_conf"]})
        killed += drop
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
            dc = 2 * inter / max(int(P.sum()) + int(G["mask"].sum()), 1)
            td.append(dc)
            tpb[G["bucket"]] += 1
        for G in im["gts"]:
            gtb[G["bucket"]] += 1
        tp += len(used_g)
        fp += len(preds) - len(used_p)
        per.append({"tp": len(used_g), "fp": len(preds) - len(used_p),
                    "gt": [G["bucket"] for G in im["gts"]],
                    "gtd": {gi: 1 for _, gi, _ in pairs}})
    rec = tp / max(sum(gtb.values()), 1)
    prec = tp / max(preds_tot, 1)
    return {"precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
            "r_small": round(tpb["small"] / max(gtb["small"], 1), 4),
            "r_medium": round(tpb["medium"] / max(gtb["medium"], 1), 4),
            "r_large": round(tpb["large"] / max(gtb["large"], 1), 4),
            "tp_dice": round(float(np.mean(td)), 4) if td else 0.0,
            "fp_img": round(fp / len(imgs), 3),
            "pred_img": round(preds_tot / len(imgs), 3),
            "killed_candidates": killed}, per


def main() -> None:
    m3, (mu, sg), lr_img, thr_img = load_frozen()
    s1m = YOLO("outputs/calculus_v3c_stage1/weights/best.pt")
    s2m = YOLO("outputs/calculus_v3b_stage2/weights/best.pt")
    tdet = YOLO(TOOTH_W)
    vb = make_verifier()
    man = pd.read_csv(DATA / "split_manifest_v3.csv")
    print("collecting test (v3c pipeline) ...", flush=True)
    imgs = collect_split(s1m, s2m, man, "test", with_verifier=True, vb=vb)
    ys = np.array([int(im["has"]) for im in imgs])

    # tooth boxes per test image
    fences = []
    tooth_stats = []
    for im in imgs:
        res = tdet.predict(im["image_path"], conf=TOOTH_CONF, iou=0.6,
                           imgsz=640, device=0, max_det=64, verbose=False)[0]
        H, W = im["H"], im["W"]
        fence = np.zeros((H, W), np.uint8)
        nb = 0
        if res.boxes is not None and len(res.boxes):
            bx = res.boxes.xyxy.cpu().numpy()
            for b in bx:
                x0, y0 = max(0, int(b[0])), max(0, int(b[1]))
                x1, y1 = min(W, int(b[2])), min(H, int(b[3]))
                if x1 > x0 and y1 > y0:
                    fence[y0:y1, x0:x1] = 1
                    nb += 1
        fences.append(fence)
        tooth_stats.append(nb)
    print(f"tooth boxes/img: median {np.median(tooth_stats):.0f}, "
          f"images with 0 boxes: {int((np.array(tooth_stats) == 0).sum())}")

    # fused probs (pre-filter) for both variants; imgA differs post-filter
    fused_pre = []
    for im in imgs:
        f = im["feats"]
        if f:
            X = np.array([[x[c] for c in FCOLS] for x in f], float)
            fused_pre.append(m3.predict_proba((X - mu) / sg)[:, 1])
        else:
            fused_pre.append(np.zeros(0))

    def imgA_post(fused_all, fence_on):
        a = []
        for im, fused, fence in zip(imgs, fused_all, fences):
            f = im["feats"]
            if not f:
                a.append(0.0); continue
            if fence_on and fence is not None:
                mask_ok = []
                for i in range(len(f)):
                    rect = im["rects"][i]
                    P = np.zeros((im["H"], im["W"]), np.uint8)
                    P[rect["y0"]:rect["y0"] + rect["ch"],
                      rect["x0"]:rect["x0"] + rect["cw"]] = im["bins"][i]
                    mask_ok.append(int((P & (fence > 0)).sum())
                                   / max(int(P.sum()), 1) >= FENCE_MIN)
                X = np.array([[x[c] for c in FCOLS] for x in f], float)
                pr = m3.predict_proba((X - mu) / sg)[:, 1]
                a.append(float(pr[mask_ok].max()) if any(mask_ok) else 0.0)
            else:
                a.append(float(fused.max()) if len(fused) else 0.0)
        return np.array(a)

    imgB = effnet_tta_probs(man, "test")
    imgC = yolocls_probs("test")

    res = {}
    for name, fence_on in (("OFF", False), ("ON", True)):
        fused_use = fused_pre
        inst, per = instance_metrics(imgs, fused_use, THR_INST,
                                     fences=fences if fence_on else None)
        imgA = imgA_post(fused_pre, fence_on)
        prob = lr_img.predict_proba(np.stack([imgA, imgB, imgC], 1))[:, 1]
        pred = (prob >= thr_img).astype(int)
        tp = int(((pred == 1) & (ys == 1)).sum())
        fp_i = int(((pred == 1) & (ys == 0)).sum())
        fn_i = int(((pred == 0) & (ys == 1)).sum())
        tn_i = int(((pred == 0) & (ys == 0)).sum())
        img_m = {"accuracy": round((tp + tn_i) / len(ys), 4),
                 "sensitivity": round(tp / max(tp + fn_i, 1), 4),
                 "specificity": round(tn_i / max(tn_i + fp_i, 1), 4),
                 "precision": round(tp / max(tp + fp_i, 1), 4),
                 "roc_auc": round(float(roc_auc_score(ys, prob)), 4),
                 "pr_auc": round(float(average_precision_score(ys, prob)), 4)}
        res[name] = {"instance": inst, "image": img_m}
        print(f"--- fence {name} ---")
        print(json.dumps({"instance": inst, "image": img_m}, indent=1))

    # Eka test-set mAP of the tooth detector itself
    m = YOLO(TOOTH_W).val(data="datasets/processed/eka_toothdet/toothdet.yaml",
                          split="test", imgsz=640, device=0, verbose=False)
    res["tooth_detector_eka_test"] = {k: round(float(v), 4)
                                      for k, v in m.results_dict.items()
                                      if k.startswith("metrics")}
    out = {"fence_min": FENCE_MIN, "thr_inst": THR_INST, "thr_img": thr_img,
           "results": res}
    (Path("outputs/eka_toothdet/ab_fence_test.json")
     .write_text(json.dumps(out, indent=2)))
    print(json.dumps(res["tooth_detector_eka_test"], indent=2))


if __name__ == "__main__":
    main()
