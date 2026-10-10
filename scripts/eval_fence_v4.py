"""Tooth-mask fence A/B for the v4 pipeline (val sweep -> frozen test).

Fence: candidate mask must overlap predicted tooth-mask union by >= FENCE_MIN.
Tooth masks: IOP-Compass-trained tooth segmenter on the same image @640.
Selection: max val instance F1 s.t. recall drop <= 0.05 vs OFF.
Outputs: outputs/eka_toothdet/fence_v4_val.json, fence_v4_test.json
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
from ultralytics import YOLO

import scripts.v3_pipeline_lib as _lib
from scripts.v3_pipeline_lib import (BUCKETS, DATA, FCOLS, NMS_IOU,
                                     collect_split, make_verifier)
from scripts.eval_tooth_fence_ab import load_frozen
from src.inference.fusion import greedy_nms, rec_bin

_lib.OUT = Path("outputs/calculus_v4_cls")
OUT = _lib.OUT
IOU_MATCH = 0.5
THR_INST = 0.6
TOOTH_W = "outputs/iop_toothseg/weights/best.pt"
GRID = [0.05, 0.10, 0.20]


def make_tooth_masks(tdet, imgs, conf=0.30):
    masks = []
    stats = []
    for im in imgs:
        res = tdet.predict(im["image_path"], conf=conf, iou=0.6, imgsz=640,
                           device=0, retina_masks=True, max_det=48,
                           verbose=False)[0]
        H, W = im["H"], im["W"]
        f = np.zeros((H, W), np.uint8)
        nb = 0
        if res.boxes is not None and len(res.boxes) and res.masks is not None:
            mk = res.masks.data.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            for i in np.where(cls == 0)[0]:
                m = cv2.resize((mk[i] > 0.5).astype(np.uint8), (W, H),
                               interpolation=cv2.INTER_NEAREST)
                f |= m
                nb += 1
        masks.append(f)
        stats.append(nb)
    return masks, stats


def instance_metrics(imgs, fused_all, thr_inst, fences, fence_min):
    tp = fp = preds_tot = killed = 0
    gtb = {b: 0 for b in BUCKETS}
    tpb = {b: 0 for b in BUCKETS}
    td = []
    for im, fused, fence in zip(imgs, fused_all,
                                fences if fences is not None else [None] * len(imgs)):
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
            "pred_img": round(preds_tot / len(imgs), 3),
            "killed": killed}


def img_metrics(imgs, fused_all, m3, mu, sg, lr_img, thr_img, imgB, imgC, ys,
                fences, fence_min):
    from sklearn.metrics import roc_auc_score
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


def run(m3, mu, sg, lr_img, thr_img, s1m, s2m, vb, tdet, man, split):
    imgs = collect_split(s1m, s2m, man, split, with_verifier=True, vb=vb,
                         log_every=350)
    ys = np.array([int(im["has"]) for im in imgs])
    fences, stats = make_tooth_masks(tdet, imgs)
    print(f"{split}: tooth masks/img median {np.median(stats):.0f}, "
          f"zeros {int((np.array(stats)==0).sum())}", flush=True)
    fused_all = []
    for im in imgs:
        f = im["feats"]
        if f:
            X = np.array([[x[c] for c in FCOLS] for x in f], float)
            fused_all.append(m3.predict_proba((X - mu) / sg)[:, 1])
        else:
            fused_all.append(np.zeros(0))
    imgB = __import__("scripts.eval_ensemble_v3b",
                      fromlist=["effnet_tta_probs"]).effnet_tta_probs(man, split)
    imgC = __import__("scripts.eval_ensemble_v3b",
                      fromlist=["yolocls_probs"]).yolocls_probs(split)
    rows = {"OFF": {"inst": instance_metrics(imgs, fused_all, THR_INST,
                                             None, None)}}
    for fm in GRID:
        rows[f"ON_{fm}"] = {"inst": instance_metrics(imgs, fused_all, THR_INST,
                                                     fences, fm)}
    for k, v in rows.items():
        print(split, k, v["inst"], flush=True)
    return imgs, ys, fences, fused_all, imgB, imgC, rows


def main() -> None:
    m3, (mu, sg), lr_img, thr_img = load_frozen()
    s1m = YOLO("outputs/calculus_v4_stage1/weights/best.pt")
    s2m = YOLO("outputs/calculus_v3b_stage2/weights/best.pt")
    vb = make_verifier()
    tdet = YOLO(TOOTH_W)
    man = pd.read_csv(DATA / "split_manifest_v3.csv")

    print("=== VAL sweep ===", flush=True)
    vi, vy, vf, vfu, vB, vC, vrows = run(m3, mu, sg, lr_img, thr_img,
                                         s1m, s2m, vb, tdet, man, "val")
    base_f1 = vrows["OFF"]["inst"]["f1"]
    base_rec = vrows["OFF"]["inst"]["recall"]
    cand = [(vrows[f"ON_{fm}"]["inst"]["f1"], fm) for fm in GRID
            if base_rec - vrows[f"ON_{fm}"]["inst"]["recall"] <= 0.05]
    choice = "OFF"
    if cand:
        bf1, bfm = max(cand)
        if bf1 > base_f1:
            choice = f"ON_{bfm}"
    print("CHOICE:", choice, flush=True)

    print("=== TEST (frozen) ===", flush=True)
    ti, ty, tf, tfu, tB, tC, trows = run(m3, mu, sg, lr_img, thr_img,
                                         s1m, s2m, vb, tdet, man, "test")
    out = {"choice": choice,
           "val": {k: v["inst"] for k, v in vrows.items()},
           "test": {k: v["inst"] for k, v in trows.items()},
           "image_val": {}, "image_test": {}}
    fm_used = None if choice == "OFF" else float(choice.split("_")[1])
    f_off = [None] * len(vi)
    f_on = vf if choice != "OFF" else f_off
    fm_ = fm_used if choice != "OFF" else None
    out["image_val"]["OFF"] = img_metrics(vi, vfu, m3, mu, sg, lr_img, thr_img,
                                          vB, vC, vy, f_off, None)
    out["image_val"][choice] = img_metrics(vi, vfu, m3, mu, sg, lr_img, thr_img,
                                           vB, vC, vy, f_on, fm_)
    out["image_test"]["OFF"] = img_metrics(ti, tfu, m3, mu, sg, lr_img, thr_img,
                                           tB, tC, ty, [None] * len(ti), None)
    out["image_test"][choice] = img_metrics(ti, tfu, m3, mu, sg, lr_img, thr_img,
                                            tB, tC, ty, tf if choice != "OFF" else [None] * len(ti),
                                            fm_)
    (OUT / "fence_v4_result.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["image_test"], indent=1))


if __name__ == "__main__":
    main()
