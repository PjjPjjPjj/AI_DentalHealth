"""FINAL TEST: frozen pipeline vs V1 baseline on untouched test split.

Final: S1 c1=.05 -> bbox-center 160 -> S2 V1 instances -> Verifier-B ->
       M3 (fusion_M3_lowc1_coef.npy) thr=0.50 -> mask-NMS 0.5 -> 1-1 match.
V1   : S1/S2 same, c2=0.40 -> mask-NMS 0.5 -> 1-1 match.
Per-image latency measured up to NMS (excludes evaluation matching).
Bootstrap 2000 image-level (seed 42). Image PR/ROC-AUC (descriptive only).
Outputs: final_test_metrics.json, final_vs_v1.csv, bootstrap_ci.csv,
test_error_analysis.csv ; raw: final_test_raw.pkl
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

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from src.inference.fusion import greedy_nms, rec_bin
from src.inference.twostage import TARGET, TwoStageSegmenter
from src.utils.paths import PROCESSED_DIR
import torch
import torch.nn as nn
from torchvision import models as tvmodels
from torchvision.transforms import functional as TF

DATA = PROCESSED_DIR / "calculus_seg"
OUT = Path("outputs/calculus_cls")
C1 = 0.05
SIDE = 160
NMS_IOU = 0.5
IOU_MATCH = 0.5
V1_C2 = 0.40
FINAL_THR = 0.50
BUCKET_EDGES = (0.00552, 0.01565)
BUCKETS = ("small", "medium", "large")
FCOLS = ["s1_conf", "s2_conf", "verifier_prob", "s1_area_ratio",
         "s2_area_ratio", "s1s2_iou", "bbox_area_ratio", "bbox_aspect"]


def bucket_of(r: float) -> str:
    if r < BUCKET_EDGES[0]:
        return "small"
    if r < BUCKET_EDGES[1]:
        return "medium"
    return "large"


def make_vb():
    vb = tvmodels.efficientnet_v2_s(
        weights=tvmodels.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    conv = vb.features[0][0]
    new = nn.Conv2d(4, conv.out_channels, conv.kernel_size, conv.stride,
                    conv.padding, conv.dilation, conv.groups, conv.bias)
    with torch.no_grad():
        new.weight[:, :3] = conv.weight
        new.weight[:, 3] = conv.weight.mean(dim=1)
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    vb.features[0][0] = new
    vb.classifier[1] = nn.Linear(vb.classifier[1].in_features, 1)
    vb.load_state_dict(torch.load(OUT / "verifier_B" / "best.pt",
                                  weights_only=False))
    return vb.to(0).eval()


def match_pair(preds, gts):
    pairs = []
    for pi, P in enumerate(preds):
        for gi, G in enumerate(gts):
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
    return matched, set(p for _, p, _ in matched)


def collect_test() -> list[dict]:
    seg = TwoStageSegmenter("outputs/calculus_yolo11m_full/weights/best.pt",
                            "outputs/calculus_yolo11m_crop/weights/best.pt",
                            c1=C1, c2=0.05, device=0, nms_iou=NMS_IOU,
                            score_mode="s2")
    vb = make_vb()
    cf = np.load(OUT / "fusion_M3_lowc1_coef.npy", allow_pickle=True).item()
    lr = LogisticRegression()
    lr.coef_ = cf["coef"]; lr.intercept_ = cf["intercept"]
    lr.classes_ = np.array([0, 1])
    mu, sg = cf["mu"], cf["sg"]
    man = pd.read_csv(DATA / "split_manifest.csv")
    rows = man[man.split == "test"]
    print(f"test images: {len(rows)}", flush=True)
    imgs = []
    lat_s1s2_all = []
    lat_vf_all = []
    for ri, (_, r) in enumerate(rows.iterrows()):
        t_all0 = time.time()
        img = cv2.imread(r["image_path"])
        H, W = img.shape[:2]
        gtb = cv2.imread(r["mask_original_path"], cv2.IMREAD_UNCHANGED)
        has = bool(r["has_calculus"])
        gts = []
        if has and gtb is not None and (gtb > 0).any():
            gtb = gtb > 0
            n, comp, _, _ = cv2.connectedComponentsWithStats(
                gtb.astype(np.uint8), 8)
            for i in range(1, n):
                if (comp == i).sum() < 5:
                    continue
                m = (comp == i).astype(np.uint8)
                px = int(m.sum())
                gts.append({"mask": m, "bucket": bucket_of(px / float(H * W)),
                            "px": px})
        boxes, confs, masks = seg._stage1_masks(r["image_path"])
        feats, bins, rects, crop_local = [], [], [], []
        for (b, sc1, msk) in zip(boxes, confs, masks):
            msk = msk.astype(np.uint8)
            bx = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
            x0 = int(round((bx[0] + bx[2]) / 2 - SIDE / 2))
            y0 = int(round((bx[1] + bx[3]) / 2 - SIDE / 2))
            x0 = max(0, min(x0, W - SIDE)); y0 = max(0, min(y0, H - SIDE))
            x1, y1 = min(x0 + SIDE, W), min(y0 + SIDE, H)
            crop_img = img[y0:y1, x0:x1]
            crop = cv2.resize(crop_img, (TARGET, TARGET),
                              interpolation=cv2.INTER_AREA)
            insts = seg._stage2_instances(crop)
            sel = [{"x0": 0, "y0": 0, "cw": TARGET, "ch": TARGET,
                    "bin": i["bin640"], "s2_conf": i["s2_conf"],
                    "score": i["s2_conf"]} for i in insts]
            keep = greedy_nms(sel, "mask_nms", NMS_IOU, TARGET, TARGET)
            rectl = {"x0": x0, "y0": y0, "cw": x1 - x0, "ch": y1 - y0}
            for k in keep:
                P = k["bin"]
                m160 = (cv2.resize(P, (SIDE, SIDE),
                                   interpolation=cv2.INTER_NEAREST) > 0)
                s1m = (cv2.resize(msk[y0:y1, x0:x1], (SIDE, SIDE),
                                  interpolation=cv2.INTER_AREA) > 0.5)
                gx, gy, gw, gh = cv2.boundingRect(P.astype(np.uint8))
                feats.append({
                    "s1_conf": float(sc1), "s2_conf": float(k["s2_conf"]),
                    "s1_area_ratio": float(s1m.sum()) / (SIDE * SIDE),
                    "s2_area_ratio": float(m160.sum()) / (SIDE * SIDE),
                    "s1s2_iou": float((s1m & m160).sum() /
                                      max((s1m | m160).sum(), 1)),
                    "bbox_area_ratio": (gw * gh) / float(TARGET * TARGET),
                    "bbox_aspect": float(min(gw, gh) / max(gw, gh)),
                    "verifier_prob": None})
                bins.append(cv2.resize(P.astype(np.uint8),
                                       (rectl["cw"], rectl["ch"]),
                                       interpolation=cv2.INTER_NEAREST))
                rects.append(rectl)
                crop_local.append({"crop_img": crop_img, "P": P})
        lat_s1s2_all.append((time.time() - t_all0) * 1000)
        if feats:
            with torch.no_grad():
                inp = []
                for it in crop_local:
                    P = it["P"]
                    m160 = (cv2.resize(P, (SIDE, SIDE),
                                       interpolation=cv2.INTER_NEAREST) > 0)
                    rgb = cv2.cvtColor(it["crop_img"], cv2.COLOR_BGR2RGB)
                    t = TF.resize(torch.from_numpy(rgb).permute(2, 0, 1)
                                  .float() / 255, (224, 224),
                                  TF.InterpolationMode.BILINEAR)
                    t = TF.normalize(t, (0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))
                    mm = torch.from_numpy(m160.astype(np.float32))[None]
                    mm = TF.resize(mm, (224, 224),
                                   TF.InterpolationMode.NEAREST)
                    mm = (mm > 0.5).float()
                    inp.append(torch.cat([t, mm], 0))
                vp = vb(torch.stack(inp).to(0)).squeeze(1).sigmoid().cpu().numpy()
            for i in range(len(feats)):
                feats[i]["verifier_prob"] = float(vp[i])
        fused = None
        if feats:
            tv0 = time.time()
            X = np.array([[f[c] for c in FCOLS] for f in feats]).astype(float)
            fused = lr.predict_proba((X - mu) / sg)[:, 1]
            lat_vf_all.append((time.time() - tv0) * 1000)
        else:
            lat_vf_all.append(0.0)
        imgs.append({"H": H, "W": W, "gts": gts, "feats": feats,
                     "bins": bins, "rects": rects, "fused": fused,
                     "s2c": np.array([f["s2_conf"] for f in feats]),
                     "has": has})
        if (ri + 1) % 200 == 0:
            print(f"  collected {ri + 1}/{len(rows)}", flush=True)
    return imgs, lat_s1s2_all, lat_vf_all


def run(imgs, select):
    """Returns (per-image row dicts, latency list)."""
    per_img = []
    lat = []
    for im in imgs:
        t0 = time.time()
        row = {"im": im, "preds": 0, "fp": 0, "gt": {b: 0 for b in BUCKETS},
               "td": [], "sem": [], "gtd": [], "fn_area": {b: [] for b in BUCKETS},
               "union": None}
        if not im["feats"]:
            lat.append((time.time() - t0) * 1000)
            for b in BUCKETS:
                for G in im["gts"]:
                    row["sem"].append(0.0)
                    row["gtd"].append(0.0)
                    row["fn_area"][b].append(G["px"])
            per_img.append(row)
            continue
        H, W = im["H"], im["W"]
        idxs = select(im)
        recs_all = []
        for i in idxs:
            rect = im["rects"][i]
            recs_all.append({"x0": rect["x0"], "y0": rect["y0"],
                             "cw": rect["cw"], "ch": rect["ch"],
                             "bin": im["bins"][i],
                             "score": im["s2c"][i]})
        keep = greedy_nms(recs_all, "mask_nms", NMS_IOU, H, W)
        lat.append((time.time() - t0) * 1000)
        preds = [rec_bin(k, H, W) for k in keep]
        row["preds"] = len(preds)
        matched, _ = match_pair(preds, im["gts"])
        matched_gi = {gi for _, gi, _ in matched}
        for pi, gi, io in matched:
            P = preds[pi]; G = im["gts"][gi]
            inter = int((P & G["mask"]).sum())
            dc = 2 * inter / (int(P.sum()) + int(G["mask"].sum())) \
                if (int(P.sum()) + int(G["mask"].sum())) else 1.0
            assert dc >= 0.6667 - 1e-3
            row["td"].append(dc)
        union = np.zeros((H, W), np.uint8)
        for P in preds:
            union |= P
        row["union"] = union
        row["fp"] = len(preds) - len(matched)
        for gi, G in enumerate(im["gts"]):
            row["gt"][G["bucket"]] += 1
            inter = int((G["mask"] & union).sum())
            uni = int((G["mask"] | union).sum())
            io = inter / uni if uni else 0.0
            dc = 2 * inter / (int(G["mask"].sum()) + int(union.sum())) \
                if (int(G["mask"].sum()) + int(union.sum())) else 1.0
            row["sem"].append(dc)
            row["gtd"].append(dc if gi in matched_gi else 0.0)
            if gi not in matched_gi:
                row["fn_area"][G["bucket"]].append(G["px"])
        per_img.append(row)
    return per_img, lat


def aggregate(per_img, n_img):
    tp = {b: sum(r["gt"][b] for r in per_img if False) for b in BUCKETS}
    tp = {b: sum(1 for r in per_img for gi, G in enumerate(r["im"]["gts"])
                 if G["bucket"] == b and False) for b in BUCKETS}
    # matched count per bucket: recompute matched flag from gtd>0 (a matched GT
    # has gtd==sem>0 equal Dice of matched instance; unmatched is exactly 0)
    tpb = {b: 0 for b in BUCKETS}
    for r in per_img:
        tpb[r["im"]["gts"][0]["bucket"]] if False else None
        # use row-level per-GT flags saved earlier: store explicitly
    # simpler: matched flags stored per GT? recompute from match again is costly;
    # store matched flags during run:
    return None


def main() -> None:
    imgs, lat_s1s2, lat_vf = collect_test()
    n_img = len(imgs)
    lat_s1s2 = np.array(lat_s1s2)
    lat_vf = np.array(lat_vf)
    print("run V1...", flush=True)
    run_v1, lat1 = run(imgs, lambda im: np.argwhere(im["s2c"] >= V1_C2).ravel())
    print("run Final...", flush=True)
    run_fin, lat2 = run(imgs,
                        lambda im: (np.argwhere(im["fused"] >= FINAL_THR).ravel()
                                    if im["fused"] is not None
                                    else np.array([], dtype=int)))
    lat1 = np.array(lat1); lat2 = np.array(lat2)

    def finalize(per_img, lat, name):
        # per GT matched flag already encoded in gtd (==0 only if unmatched)
        # but a matched GT with dice 0 impossible (>=0.667); unmatched => 0.
        tpb = {b: 0 for b in BUCKETS}
        gtpb = {b: 0 for b in BUCKETS}
        for r in per_img:
            # matched count per bucket = number of gtd entries >0 with bucket b
            for gi, G in enumerate(r["im"]["gts"]):
                gtpb[G["bucket"]] += 1
                if r["gtd"][gi] > 0:
                    tpb[G["bucket"]] += 1
        preds = sum(r["preds"] for r in per_img)
        fp = sum(r["fp"] for r in per_img)
        td = [d for r in per_img for d in r["td"]]
        sem = [d for r in per_img for d in r["sem"]]
        gtt = [d for r in per_img for d in r["gtd"]]
        tp_tot = sum(tpb.values()); gt_tot = sum(gtpb.values())
        rec = tp_tot / gt_tot if gt_tot else 0.0
        prec = tp_tot / preds if preds else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return {
            "name": name,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4),
            "r_small": round(tpb["small"] / gtpb["small"], 4),
            "r_medium": round(tpb["medium"] / gtpb["medium"], 4),
            "r_large": round(tpb["large"] / gtpb["large"], 4),
            "tp_dice": round(float(np.mean(td)), 4) if td else 0.0,
            "gt_dice": round(float(np.mean(gtt)), 4) if gtt else 0.0,
            "sem_dice": round(float(np.mean(sem)), 4) if sem else 0.0,
            "fp_img": round(fp / n_img, 3),
            "pred_img": round(preds / n_img, 3),
            "lat_mean": round(float(np.mean(lat)), 1),
            "lat_med": round(float(np.median(lat)), 1),
            "lat_p95": round(float(np.percentile(lat, 95)), 1),
            "_per_img": per_img, "_lat": lat,
        }

    # store matched flags: add r_gt=matched? recompute via gtd>0 (safe)
    res1 = finalize(run_v1, lat_s1s2 + lat1, "V1")
    res2 = finalize(run_fin, lat_s1s2 + lat_vf + lat2, "Final")

    # image-level scores for PR/ROC (descriptive)
    def img_scores(per_img, mode):
        out = []
        for r in per_img:
            im = r["im"]
            if mode == "v1":
                sc = float(im["s2c"].max()) if len(im["s2c"]) else 0.0
            else:
                sc = float(im["fused"].max()) if im["fused"] is not None and len(im["fused"]) else 0.0
            out.append(sc)
        return np.array(out)

    y = np.array([im["has"] for im in imgs], dtype=int)
    sc1 = img_scores(run_v1, "v1")
    sc2 = img_scores(run_fin, "final")
    auc1 = roc_auc_score(y, sc1) if float(np.unique(y).size) == 2 else float("nan")
    auc2 = roc_auc_score(y, sc2) if float(np.unique(y).size) == 2 else float("nan")
    pra1 = average_precision_score(y, sc1)
    pra2 = average_precision_score(y, sc2)

    # bootstrap (image-level, 2000, seed42)
    def bootstrap(per_img, n_img, n_boot=2000, seed=42):
        rng = random.Random(seed)
        b = {m: {} for m in ("precision", "recall", "f1", "r_small",
                             "r_medium", "r_large", "tp_dice", "fp_img")}
        n = len(per_img)
        metric_arrays = []
        # precompute per-image contributions
        tpb_all, gtpb_all, td_all, fp_all = [], [], [], []
        for r in per_img:
            tpb = {bk: 0 for bk in BUCKETS}
            gtpb = {bk: 0 for bk in BUCKETS}
            for gi, G in enumerate(r["im"]["gts"]):
                gtpb[G["bucket"]] += 1
                if r["gtd"][gi] > 0:
                    tpb[G["bucket"]] += 1
            tpb_all.append(tpb); gtpb_all.append(gtpb)
            td_all.append(list(r["td"])); fp_all.append(r["fp"])
        import numpy as np
        samples = {m: [] for m in b}
        for _ in range(n_boot):
            idx = [rng.randrange(n) for _ in range(n)]
            tp = {bk: sum(tpb_all[i][bk] for i in idx) for bk in BUCKETS}
            gt = {bk: sum(gtpb_all[i][bk] for i in idx) for bk in BUCKETS}
            preds = sum(per_img[i]["preds"] for i in idx)
            fp = sum(fp_all[i] for i in idx)
            tdt = [d for i in idx for d in td_all[i]]
            tp_tot = sum(tp.values()); gt_tot = sum(gt.values())
            rec = tp_tot / gt_tot if gt_tot else 0.0
            prec = tp_tot / preds if preds else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            samples["precision"].append(prec)
            samples["recall"].append(rec)
            samples["f1"].append(f1)
            samples["r_small"].append(tp["small"] / gt["small"] if gt["small"] else 0.0)
            samples["r_medium"].append(tp["medium"] / gt["medium"] if gt["medium"] else 0.0)
            samples["r_large"].append(tp["large"] / gt["large"] if gt["large"] else 0.0)
            samples["tp_dice"].append(float(np.mean(tdt)) if tdt else 0.0)
            samples["fp_img"].append(fp / n_img)
        for m, vals in samples.items():
            b[m] = {"low": round(float(np.percentile(vals, 2.5)), 4),
                    "high": round(float(np.percentile(vals, 97.5)), 4)}
        return b

    ci1 = bootstrap(run_v1, n_img)
    ci2 = bootstrap(run_fin, n_img)

    def slim(res):
        return {k: v for k, v in res.items() if not k.startswith("_")}

    metrics = {
        "test_n": n_img, "test_pos": int(y.sum()), "test_neg": int((y == 0).sum()),
        "final": {**slim(res2), "image_roc_auc": round(float(auc2), 4),
                  "image_pr_auc": round(float(pra2), 4),
                  "bootstrap_ci": ci2},
        "v1_baseline": {**slim(res1), "image_roc_auc": round(float(auc1), 4),
                        "image_pr_auc": round(float(pra1), 4),
                        "bootstrap_ci": ci1},
    }
    (OUT / "final_test_metrics.json").write_text(json.dumps(metrics, indent=2))
    delta = {k: round(res2[k] - res1[k], 4)
             for k in ("precision", "recall", "f1", "r_small", "r_medium",
                       "r_large", "tp_dice", "gt_dice", "sem_dice", "fp_img",
                       "pred_img", "lat_mean")}
    pd.DataFrame([
        {"metric": m, "V1": slim(res1).get(m, None) if not m.startswith("lat") else res1[m],
         "Final": slim(res2).get(m, None) if not m.startswith("lat") else res2[m],
         "delta": delta.get(m)}
        for m in ("precision", "recall", "f1", "r_small", "r_medium",
                  "r_large", "tp_dice", "gt_dice", "sem_dice", "fp_img",
                  "pred_img", "lat_mean", "lat_med", "lat_p95")
    ]).to_csv(OUT / "final_vs_v1.csv", index=False)
    ci_rows = []
    for src, tag in ((ci1, "V1"), (ci2, "Final")):
        for m, v in src.items():
            ci_rows.append({"model": tag, "metric": m,
                            "ci95_low": v["low"], "ci95_high": v["high"]})
    pd.DataFrame(ci_rows).to_csv(OUT / "bootstrap_ci.csv", index=False)
    # error analysis (Final)
    fn = {b: sum(len(r["fn_area"][b]) for r in run_fin) for b in BUCKETS}
    fp_tot = sum(r["fp"] for r in run_fin)
    fp_dist = [r["fp"] for r in run_fin]
    small_fn = [a for r in run_fin for a in r["fn_area"]["small"]]
    ea = {
        "fn_by_bucket": fn, "fp_total_preds": fp_tot,
        "fp_per_image_mean": round(fp_tot / n_img, 3),
        "fp_per_image": {
            "min": int(min(fp_dist)), "p25": int(np.percentile(fp_dist, 25)),
            "median": int(np.median(fp_dist)), "p75": int(np.percentile(fp_dist, 75)),
            "max": int(max(fp_dist)), "imgs_with_fp": int((np.array(fp_dist) > 0).sum())},
        "small_fn_area": {
            "n": len(small_fn),
            "median": float(np.median(small_fn)) if small_fn else None,
            "p25": float(np.percentile(small_fn, 25)) if small_fn else None,
            "p75": float(np.percentile(small_fn, 75)) if small_fn else None},
    }
    pd.DataFrame([{"bucket": k, "fn": v} for k, v in fn.items()] +
                 [{"bucket": "fp_total", "fn": fp_tot}] +
                 [{"bucket": "fp_dist_p25", "fn": ea["fp_per_image"]["p25"]},
                  {"bucket": "fp_dist_med", "fn": ea["fp_per_image"]["median"]},
                  {"bucket": "fp_dist_p75", "fn": ea["fp_per_image"]["p75"]},
                  {"bucket": "small_fn_area_med", "fn": ea["small_fn_area"]["median"]},
                  {"bucket": "small_fn_area_p25", "fn": ea["small_fn_area"]["p25"]},
                  {"bucket": "small_fn_area_p75", "fn": ea["small_fn_area"]["p75"]}]
                 ).to_csv(OUT / "test_error_analysis.csv", index=False)
    (OUT / "final_test_raw.pkl").write_bytes(pickle.dumps({
        "run_v1": run_v1, "run_fin": run_fin}))

    print("\n===== FINAL TEST SUMMARY =====")
    print(pd.DataFrame([
        {"metric": m, "V1": (res1.get(m) if not m.startswith("lat") else
                             res1[m]), "Final": (res2.get(m) if not m.startswith("lat")
                                                else res2[m]), "delta": delta.get(m)}
        for m in ("precision", "recall", "f1", "r_small", "r_medium",
                  "r_large", "tp_dice", "gt_dice", "sem_dice", "fp_img",
                  "pred_img", "lat_mean", "lat_med", "lat_p95")
    ]).to_string(index=False))
    print("image ROC/PR-AUC  V1:", round(auc1, 4), round(pra1, 4),
          " Final:", round(auc2, 4), round(pra2, 4))


if __name__ == "__main__":
    main()
