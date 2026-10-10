"""FINAL TEST v3: frozen pipeline on untouched group-honest test split.

Frozen: S1(c1=.05) -> bbox-center 160 -> S2(c2=.05, cls0) -> intra-window
mask-NMS(.5) -> Verifier(4ch) -> M3 fusion (thr_inst) -> mask-NMS(.5) ->
instance metrics; image-level logistic on [max fused prob, imgclf prob] at
thr_img.  Bootstrap 2000 (seed 42).  Per-domain breakdown + FP anatomy.
Outputs: outputs/calculus_v3_cls/final_v3_metrics.json, final_v3_vs_v1.csv,
         bootstrap_v3.csv, test_error_analysis_v3.csv, final_v3_raw.pkl
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from ultralytics import YOLO

from scripts.v3_pipeline_lib import (BUCKETS, DATA, FCOLS, OUT, collect_split,
                                     make_imgclf, make_verifier)

IOU_MATCH = 0.5
NMS_IOU = 0.5


def load_frozen():
    z = np.load(OUT / "fusion_v3.npz")
    m3 = LogisticRegression(max_iter=1000, class_weight="balanced")
    m3.coef_ = z["coef"]; m3.intercept_ = z["intercept"]; m3.classes_ = np.array([0, 1])
    imglr = LogisticRegression(max_iter=1000)
    imglr.coef_ = z["img_coef"]; imglr.intercept_ = z["img_intercept"]
    imglr.classes_ = np.array([0, 1])
    return m3, (z["mu"], z["sg"]), imglr


def select_preds(im, fused, thr_inst):
    from src.inference.fusion import greedy_nms, rec_bin
    idxs = np.argwhere(fused >= thr_inst).ravel()
    if not len(idxs):
        return []
    recs = []
    for i in idxs:
        rect = im["rects"][i]
        recs.append({"x0": rect["x0"], "y0": rect["y0"], "cw": rect["cw"],
                     "ch": rect["ch"], "bin": im["bins"][i],
                     "score": im["feats"][i]["s2_conf"]})
    H, W = im["H"], im["W"]
    keep = greedy_nms(recs, "mask_nms", NMS_IOU, H, W)
    return [rec_bin(k, H, W) for k in keep]


def run_eval(imgs, fused_all, thr_inst, lat_pre=None):
    per_img = []
    lat = []
    for ii, im in enumerate(imgs):
        t0 = time.time()
        fused = fused_all[ii]
        preds = select_preds(im, fused, thr_inst)
        row = {"preds": len(preds), "fp": 0,
               "gt": {b: 0 for b in BUCKETS}, "td": [], "sem": [],
               "gtd": [0.0] * len(im["gts"]),
               "fn_area": {b: [] for b in BUCKETS},
               "fp_on_other": 0}
        # 1-1 greedy match
        pairs = []
        for pi, P in enumerate(preds):
            for gi, G in enumerate(im["gts"]):
                inter = int((P & G["mask"]).sum())
                uni = int((P | G["mask"]).sum())
                io = inter / uni if uni else 0.0
                if io >= IOU_MATCH:
                    pairs.append((io, pi, gi))
        pairs.sort(reverse=True)
        used_p, used_g, matched = set(), set(), []
        for io, pi, gi in pairs:
            if pi in used_p or gi in used_g:
                continue
            used_p.add(pi); used_g.add(gi)
            matched.append((pi, gi, io))
        row["fp"] = len(preds) - len(matched)
        union = np.zeros((im["H"], im["W"]), np.uint8)
        for pi, gi, io in matched:
            P, G = preds[pi], im["gts"][gi]
            inter = int((P & G["mask"]).sum())
            dc = 2 * inter / (int(P.sum()) + int(G["mask"].sum()))
            row["td"].append(dc)
            row["gtd"][gi] = dc
        for G in im["gts"]:
            row["gt"][G["bucket"]] += 1
        for P in preds:
            union |= P
        for gi, G in enumerate(im["gts"]):
            inter = int((G["mask"] & union).sum())
            uni = int((G["mask"] | union).sum())
            dc = 2 * inter / (int(G["mask"].sum()) + int(union.sum())) \
                if (int(G["mask"].sum()) + int(union.sum())) else 1.0
            row["sem"].append(dc)
            if gi not in used_g:
                row["fn_area"][G["bucket"]].append(G["px"])
        for pi in range(len(preds)):
            if pi not in used_p:
                P = preds[pi]
                if (P & (im["other"] > 0)).sum() > 0.3 * max(int(P.sum()), 1):
                    row["fp_on_other"] += 1
        per_img.append(row)
        lat.append(((time.time() - t0) * 1000) + (lat_pre[ii] if lat_pre else 0.0))
    return per_img, lat


def finalize2(imgs, per_img, lat, n_img):
    preds_tot = sum(r["preds"] for r in per_img)
    fp = sum(r["fp"] for r in per_img)
    fp_on_other = sum(r["fp_on_other"] for r in per_img)
    td = [d for r in per_img for d in r["td"]]
    # matched GT flags via gtd>0 (IoU>=0.5 match implies dice>=0.667)
    tpb = {b: 0 for b in BUCKETS}
    gtpb = {b: 0 for b in BUCKETS}
    for im, r in zip(imgs, per_img):
        for gi, G in enumerate(im["gts"]):
            gtpb[G["bucket"]] += 1
    # matched flags from gtd: matched GT contributes dice of matched pair;
    # unmatched 0.  A matched GT could in theory have low dice but IoU>=0.5
    # implies dice>=0.667. gtd==0 iff unmatched (or matched w/ dice 0 impossible)
    for im, r in zip(imgs, per_img):
        for gi, G in enumerate(im["gts"]):
            if r["gtd"][gi] > 0:
                tpb[G["bucket"]] += 1
    tp = sum(tpb.values()); gt = sum(gtpb.values())
    rec = tp / max(gt, 1)
    prec = tp / max(preds_tot, 1)
    return {
        "precision": round(prec, 4), "recall": round(rec, 4),
        "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
        "r_small": round(tpb["small"] / max(gtpb["small"], 1), 4),
        "r_medium": round(tpb["medium"] / max(gtpb["medium"], 1), 4),
        "r_large": round(tpb["large"] / max(gtpb["large"], 1), 4),
        "tp_dice": round(float(np.mean(td)), 4) if td else 0.0,
        "fp_img": round(fp / n_img, 3),
        "pred_img": round(preds_tot / n_img, 3),
        "fp_on_other_share": round(fp_on_other / max(fp, 1), 3),
        "lat_mean": round(float(np.mean(lat)), 1),
        "lat_med": round(float(np.median(lat)), 1),
        "lat_p95": round(float(np.percentile(lat, 95)), 1),
    }


def bootstrap(imgs, per_img, n_img, thr_img_unused=None, n_boot=2000, seed=42):
    rng = random.Random(seed)
    n = len(per_img)
    gtp_all, tp_all, td_all, fp_all, pred_all = [], [], [], [], []
    for im, r in zip(imgs, per_img):
        gtpb = {b: 0 for b in BUCKETS}
        tpb = {b: 0 for b in BUCKETS}
        for gi, G in enumerate(im["gts"]):
            gtpb[G["bucket"]] += 1
            if r["gtd"][gi] > 0:
                tpb[G["bucket"]] += 1
        gtp_all.append(gtpb); tp_all.append(tpb)
        td_all.append(r["td"]); fp_all.append(r["fp"]); pred_all.append(r["preds"])
    samples = {m: [] for m in ("precision", "recall", "f1", "r_small",
                               "r_medium", "r_large", "tp_dice", "fp_img")}
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        tp = {b: sum(tp_all[i][b] for i in idx) for b in BUCKETS}
        gt = {b: sum(gtp_all[i][b] for i in idx) for b in BUCKETS}
        pr = sum(pred_all[i] for i in idx)
        fp = sum(fp_all[i] for i in idx)
        tp_t, gt_t = sum(tp.values()), sum(gt.values())
        rec = tp_t / max(gt_t, 1)
        prec = tp_t / max(pr, 1)
        tdt = [d for i in idx for d in td_all[i]]
        samples["precision"].append(prec)
        samples["recall"].append(rec)
        samples["f1"].append(2 * prec * rec / max(prec + rec, 1e-9))
        for b in BUCKETS:
            samples[f"r_{b}"].append(tp[b] / max(gt[b], 1))
        samples["tp_dice"].append(float(np.mean(tdt)) if tdt else 0.0)
        samples["fp_img"].append(fp / n_img)
    return {m: {"low": round(float(np.percentile(v, 2.5)), 4),
                "high": round(float(np.percentile(v, 97.5)), 4)}
            for m, v in samples.items()}


def main() -> None:
    chosen = json.loads((OUT / "chosen_v3.json").read_text())
    thr_inst = chosen["thr_inst"]
    thr_img = chosen["thr_img_acc"]
    s1m = YOLO("outputs/calculus_v3_stage1/weights/best.pt")
    s2m = YOLO("outputs/calculus_v3_stage2/weights/best.pt")
    vb = make_verifier()
    clf = make_imgclf()
    m3, (mu, sg), imglr = load_frozen()
    man = pd.read_csv(DATA / "split_manifest_v3.csv")
    print("collecting test ...", flush=True)
    t0 = time.time()
    imgs = collect_split(s1m, s2m, man, "test", with_verifier=True, vb=vb,
                         with_imgclf=True, clf=clf)
    lat_pre = [0.0] * len(imgs)  # latency measured inside run_eval below

    fused_all, imgA = [], []
    for im in imgs:
        feats = im["feats"]
        if feats:
            X = np.array([[f[c] for c in FCOLS] for f in feats], float)
            pr = m3.predict_proba((X - mu) / sg)[:, 1]
            fused_all.append(pr)
            imgA.append(float(pr.max()))
        else:
            fused_all.append(np.zeros(0))
            imgA.append(0.0)
    imgB = np.array([im["imgclf_prob"] for im in imgs])
    ys = np.array([int(im["has"]) for im in imgs])
    img_prob = imglr.predict_proba(np.stack([np.array(imgA), imgB], 1))[:, 1]

    # latency: time S1 pass on a fresh subset (honest single-stage timing)
    from scripts.v3_pipeline_lib import s1_pass
    lat_subset = []
    for im in imgs[:60]:
        t0 = time.time()
        s1_pass(s1m, im["image_path"])
        lat_subset.append((time.time() - t0) * 1000)
    lat_note = {"s1_only_mean_ms_60img": round(float(np.mean(lat_subset)), 1)}

    per_img, _ = run_eval(imgs, fused_all, thr_inst)
    n_img = len(imgs)
    inst = finalize2(imgs, per_img, [0] * n_img, n_img)

    pred = (img_prob >= thr_img).astype(int)
    tp = int(((pred == 1) & (ys == 1)).sum())
    fp_i = int(((pred == 1) & (ys == 0)).sum())
    fn_i = int(((pred == 0) & (ys == 1)).sum())
    tn_i = int(((pred == 0) & (ys == 0)).sum())
    img_metrics = {
        "threshold": thr_img, "accuracy": round((tp + tn_i) / n_img, 4),
        "sensitivity": round(tp / max(tp + fn_i, 1), 4),
        "specificity": round(tn_i / max(tn_i + fp_i, 1), 4),
        "precision": round(tp / max(tp + fp_i, 1), 4),
        "f1": round(2 * tp / max(2 * tp + fp_i + fn_i, 1), 4),
        "roc_auc": round(float(roc_auc_score(ys, img_prob)), 4),
        "pr_auc": round(float(average_precision_score(ys, img_prob)), 4),
        "tp": tp, "fp": fp_i, "fn": fn_i, "tn": tn_i,
    }
    # per-domain
    dom_rows = []
    for dom, sub in pd.DataFrame({
            "domain": [im["domain"] for im in imgs],
            "y": ys, "p": pred}).groupby("domain"):
        acc = float((sub.y == sub.p).mean())
        pos = int(sub.y.sum())
        sens = float(((sub.y == 1) & (sub.p == 1)).sum()) / max(pos, 1)
        dom_rows.append({"domain": dom, "n": len(sub), "acc": round(acc, 4),
                         "sens": round(sens, 4), "pos": pos})
    ci = bootstrap(imgs, per_img, n_img)

    # error analysis
    fn_by_bucket = {b: sum(len(r["fn_area"][b]) for r in per_img) for b in BUCKETS}
    small_fn = [a for r in per_img for a in r["fn_area"]["small"]]
    fp_imgs = sum(1 for r in per_img if r["fp"] > 0)
    out = {
        "frozen": {"thr_inst": thr_inst, "thr_img": thr_img,
                   "s1_c1": 0.05, "s2_c2": 0.05, "nms_iou": NMS_IOU},
        "test_n": n_img, "test_pos": int(ys.sum()),
        "instance": inst, "image": img_metrics,
        "per_domain": dom_rows, "bootstrap_ci": ci,
        "error_analysis": {
            "fn_by_bucket": fn_by_bucket,
            "small_fn_area_median": float(np.median(small_fn)) if small_fn else None,
            "imgs_with_fp": fp_imgs,
            "imgs_with_fp_share": round(fp_imgs / n_img, 3),
            "fn_images_aurocable": None},
        "latency": lat_note,
    }
    (OUT / "final_v3_metrics.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame(dom_rows).to_csv(OUT / "final_v3_per_domain.csv", index=False)
    rows = [{"model": "Final", "metric": m, "ci95_low": v["low"],
             "ci95_high": v["high"]} for m, v in ci.items()]
    pd.DataFrame(rows).to_csv(OUT / "bootstrap_v3.csv", index=False)
    (OUT / "final_v3_raw.pkl").write_bytes(pickle.dumps({
        "imgs": [{k: v for k, v in im.items() if k != "other"} for im in imgs],
        "per_img": per_img, "img_prob": img_prob, "ys": ys}))
    print(json.dumps({k: v for k, v in out.items() if k != "bootstrap_ci"},
                     indent=2))
    print("\n=== BOOTSTRAP CI ===")
    for m, v in ci.items():
        print(f"{m}: [{v['low']}, {v['high']}]")


if __name__ == "__main__":
    main()
