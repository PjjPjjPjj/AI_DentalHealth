"""FINAL TEST v3c: tiled-S1 two-stage pipeline + 3-stream image ensemble.

Instance metrics: S1c(tiled, c1=0.02) -> S2b -> verifier -> M3 (thr_inst from
val) -> mask-NMS.  Image decision: logistic on [imgA, imgB(EffNet TTA),
imgC(YOLO-cls)] with val-frozen threshold (ensemble_v3b.npz).
Outputs: outputs/calculus_v3_cls/final_v3c_metrics.json + per-domain csv.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import json
import random
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

IOU_MATCH = 0.5
S1C_W = "outputs/calculus_v3c_stage1/weights/best.pt"
S2B_W = "outputs/calculus_v3b_stage2/weights/best.pt"


def load_frozen():
    z = np.load(OUT / "fusion_v3.npz")
    m3 = LogisticRegression(max_iter=1000, class_weight="balanced")
    m3.coef_, m3.intercept_, m3.classes_ = z["coef"], z["intercept"], np.array([0, 1])
    mu, sg = z["mu"], z["sg"]
    e = np.load(OUT / "ensemble_v3b.npz")
    routed = int(e["routed"][0]) if "routed" in e else 0
    streams = [s for s in e["streams"].tolist()] if "streams" in e \
        else ["A", "B", "C"]
    lr = LogisticRegression(max_iter=1000)
    if routed:
        lr_w = LogisticRegression(max_iter=1000)
        lr_w.coef_, lr_w.intercept_, lr_w.classes_ = \
            e["wide_coef"], e["wide_int"], np.array([0, 1])
        lr_c = LogisticRegression(max_iter=1000)
        lr_c.coef_, lr_c.intercept_, lr_c.classes_ = \
            e["closeup_coef"], e["closeup_int"], np.array([0, 1])
        lr = {"wide": lr_w, "closeup": lr_c,
              "thr": {"wide": float(e["wide_thr"]),
                      "closeup": float(e["closeup_thr"])}}
    else:
        lr.coef_, lr.intercept_, lr.classes_ = e["coef"], e["intercept"], np.array([0, 1])
        lr = {"plain": lr, "thr": float(e["thr"])}
    return m3, (mu, sg), lr, streams, routed


def convnext_probs(man: pd.DataFrame, split: str) -> np.ndarray:
    import torch
    import torch.nn as nn
    from torchvision import models as tv
    from scripts.train_v3c_convnext import prep
    m = tv.convnext_tiny(weights=None)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, 1)
    m.load_state_dict(torch.load(OUT / "convnext" / "best.pt", weights_only=False))
    m = m.to(0).eval()
    rows = man[man.split == split].reset_index(drop=True)
    probs = []
    with torch.no_grad():
        for i in range(0, len(rows), 24):
            batch = rows.iloc[i:i + 24]
            xs = [prep(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))
                  for p in batch.image_path]
            probs += m(torch.stack(xs).to(0)).squeeze(1).sigmoid().cpu().tolist()
    return np.array(probs)


def run_instance_metrics(imgs, fused_all, thr_inst):
    from src.inference.fusion import greedy_nms, rec_bin
    per_img = []
    for im, fused in zip(imgs, fused_all):
        t0 = __import__("time").time()
        preds = []
        if len(fused):
            idxs = np.argwhere(fused >= thr_inst).ravel()
            recs = []
            for i in idxs:
                rect = im["rects"][i]
                recs.append({"x0": rect["x0"], "y0": rect["y0"], "cw": rect["cw"],
                             "ch": rect["ch"], "bin": im["bins"][i],
                             "score": im["feats"][i]["s2_conf"]})
            if recs:
                keep = greedy_nms(recs, "mask_nms", NMS_IOU, im["H"], im["W"])
                preds = [rec_bin(k, im["H"], im["W"]) for k in keep]
        row = {"preds": len(preds), "fp": 0, "td": [], "gtd": [0.0] * len(im["gts"]),
               "fn_area": {b: [] for b in BUCKETS}, "fp_on_other": 0}
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
            dc = 2 * inter / (int(P.sum()) + int(G["mask"].sum()))
            row["td"].append(dc)
            row["gtd"][gi] = dc
        row["fp"] = len(preds) - len(used_p)
        union = np.zeros((im["H"], im["W"]), np.uint8)
        for P in preds:
            union |= P
        for gi, G in enumerate(im["gts"]):
            inter = int((G["mask"] & union).sum())
            uni = int((G["mask"] | union).sum())
            row["sem"] = row.get("sem", []) + [
                2 * inter / (int(G["mask"].sum()) + int(union.sum()))
                if (int(G["mask"].sum()) + int(union.sum())) else 1.0]
            if gi not in used_g:
                row["fn_area"][G["bucket"]].append(G["px"])
        for pi, P in enumerate(preds):
            if pi not in used_p:
                if (P & (im["other"] > 0)).sum() > 0.3 * max(int(P.sum()), 1):
                    row["fp_on_other"] += 1
        row["lat_ms"] = (__import__("time").time() - t0) * 1000
        per_img.append(row)
    return per_img


def bootstrap(imgs, per_img, n_img, n_boot=2000, seed=42):
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
    out = {m: [] for m in ("precision", "recall", "f1", "r_small", "r_medium",
                           "r_large", "tp_dice", "fp_img")}
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
        out["precision"].append(prec)
        out["recall"].append(rec)
        out["f1"].append(2 * prec * rec / max(prec + rec, 1e-9))
        for b in BUCKETS:
            out[f"r_{b}"].append(tp[b] / max(gt[b], 1))
        out["tp_dice"].append(float(np.mean(tdt)) if tdt else 0.0)
        out["fp_img"].append(fp / n_img)
    return {m: {"low": round(float(np.percentile(v, 2.5)), 4),
                "high": round(float(np.percentile(v, 97.5)), 4)}
            for m, v in out.items()}


def main() -> None:
    import torch.nn as nn
    chosen = json.loads((OUT / "chosen_v3.json").read_text())
    thr_inst = chosen["thr_inst"]
    m3, (mu, sg), lr_img, streams, routed = load_frozen()
    s1m = YOLO(S1C_W)
    s2m = YOLO(S2B_W)
    vb = make_verifier()
    man = pd.read_csv(DATA / "split_manifest_v3.csv")
    print("collecting test ...", flush=True)
    imgs = collect_split(s1m, s2m, man, "test", with_verifier=True, vb=vb)
    imgA = []
    for im in imgs:
        feats = im["feats"]
        if feats:
            X = np.array([[f[c] for c in FCOLS] for f in feats], float)
            imgA.append(float(m3.predict_proba((X - mu) / sg)[:, 1].max()))
        else:
            imgA.append(0.0)
    imgA = np.array(imgA)
    stream_map = {"A": imgA,
                  "B": effnet_tta_probs(man, "test"),
                  "C": yolocls_probs("test"),
                  "D": convnext_probs(man, "test")}
    Ximg = np.stack([stream_map[s] for s in streams], 1)
    ys = np.array([int(im["has"]) for im in imgs])
    if routed:
        sizes = np.array([max(im["H"], im["W"]) for im in imgs])
        prob = np.zeros(len(ys))
        pred = np.zeros(len(ys), int)
        for grp in ("wide", "closeup"):
            m = (sizes >= 500) if grp == "wide" else (sizes < 500)
            if m.sum() == 0:
                continue
            p = lr_img[grp].predict_proba(Ximg[m])[:, 1]
            prob[m] = p
            pred[m] = (p >= lr_img["thr"][grp]).astype(int)
        thr_img = lr_img["thr"]
    else:
        prob = lr_img["plain"].predict_proba(Ximg)[:, 1]
        thr_img = lr_img["thr"]
        pred = (prob >= thr_img).astype(int)

    fused_all = []
    for im in imgs:
        feats = im["feats"]
        if feats:
            X = np.array([[f[c] for c in FCOLS] for f in feats], float)
            fused_all.append(m3.predict_proba((X - mu) / sg)[:, 1])
        else:
            fused_all.append(np.zeros(0))
    per_img = run_instance_metrics(imgs, fused_all, thr_inst)

    # instance summary
    preds_tot = sum(r["preds"] for r in per_img)
    fp = sum(r["fp"] for r in per_img)
    td = [d for r in per_img for d in r["td"]]
    tpb = {b: 0 for b in BUCKETS}
    gtpb = {b: 0 for b in BUCKETS}
    for im, r in zip(imgs, per_img):
        for gi, G in enumerate(im["gts"]):
            gtpb[G["bucket"]] += 1
            if r["gtd"][gi] > 0:
                tpb[G["bucket"]] += 1
    tp_i = sum(tpb.values()); gt_i = sum(gtpb.values())
    rec = tp_i / max(gt_i, 1)
    prec = tp_i / max(preds_tot, 1)
    inst = {"precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
            "r_small": round(tpb["small"] / max(gtpb["small"], 1), 4),
            "r_medium": round(tpb["medium"] / max(gtpb["medium"], 1), 4),
            "r_large": round(tpb["large"] / max(gtpb["large"], 1), 4),
            "tp_dice": round(float(np.mean(td)), 4) if td else 0.0,
            "fp_img": round(fp / len(imgs), 3),
            "pred_img": round(preds_tot / len(imgs), 3)}

    tp = int(((pred == 1) & (ys == 1)).sum())
    fp_i = int(((pred == 1) & (ys == 0)).sum())
    fn_i = int(((pred == 0) & (ys == 1)).sum())
    tn_i = int(((pred == 0) & (ys == 0)).sum())
    n = len(ys)
    thr_rep = {k: round(v, 2) for k, v in thr_img.items()} if routed \
        else round(thr_img, 2)
    img_m = {"threshold": thr_rep,
             "accuracy": round((tp + tn_i) / n, 4),
             "sensitivity": round(tp / max(tp + fn_i, 1), 4),
             "specificity": round(tn_i / max(tn_i + fp_i, 1), 4),
             "precision": round(tp / max(tp + fp_i, 1), 4),
             "f1": round(2 * tp / max(2 * tp + fp_i + fn_i, 1), 4),
             "roc_auc": round(float(roc_auc_score(ys, prob)), 4),
             "pr_auc": round(float(average_precision_score(ys, prob)), 4),
             "tp": tp, "fp": fp_i, "fn": fn_i, "tn": tn_i}
    dom_rows = []
    for dom, sub in pd.DataFrame({
            "domain": [im["domain"] for im in imgs], "y": ys,
            "p": pred}).groupby("domain"):
        dom_rows.append({"domain": dom, "n": len(sub),
                         "acc": round(float((sub.y == sub.p).mean()), 4),
                         "sens": round(float(((sub.y == 1) & (sub.p == 1)).sum()
                                             / max(int(sub.y.sum()), 1)), 4),
                         "pos": int(sub.y.sum())})
    ci = bootstrap(imgs, per_img, n)
    fn_by_bucket = {b: sum(len(r["fn_area"][b]) for r in per_img) for b in BUCKETS}
    out = {"frozen": {"thr_inst": thr_inst,
                      "thr_img": {k: round(v, 2) for k, v in thr_img.items()}
                      if routed else round(thr_img, 2),
                      "s1": S1C_W, "tiled_inference": True,
                      "ensemble_streams": streams, "routed": routed},
           "test_n": n, "instance": inst, "image": img_m,
           "per_domain": dom_rows, "bootstrap_ci": ci,
           "fn_by_bucket": fn_by_bucket}
    (OUT / "final_v3c_metrics.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame(dom_rows).to_csv(OUT / "final_v3c_domains.csv", index=False)
    print(json.dumps({k: v for k, v in out.items() if k != "bootstrap_ci"},
                     indent=2))
    print("BOOTSTRAP:", json.dumps(ci, indent=1))


if __name__ == "__main__":
    main()
