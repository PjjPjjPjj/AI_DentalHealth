"""EXP21: strict M0/M1/M2/M3 fusion ablation on main VAL (no retraining).

Same single-pass collection, unified threshold grid 0.02..0.90.
Constraints: smallR>=0.40, mediumR>=0.60, largeR>=0.60, TP-Dice>=0.78
-> pick best F1 per model. Fine scan 0.44..0.48 only for the competing
candidate models (M1/M3; M2 if it wins). Decision rule:
  M3-M1 F1 <0.005 and Recall diff <0.01 -> drop verifier, freeze M1
  0.005..0.01 -> prefer simpler M1 unless smallR/FP/latency strongly say else
  >=0.01 and Recall unharmed -> keep M3
Outputs: outputs/calculus_cls/exp21_ablation.csv + exp21_decision.json
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.inference.fusion import greedy_nms, rec_bin
from src.inference.twostage import TARGET, TwoStageSegmenter
from src.utils.paths import PROCESSED_DIR

DATA = PROCESSED_DIR / "calculus_seg"
OUT = Path("outputs/calculus_cls")
C1 = 0.05
SIDE = 160
NMS_IOU = 0.5
IOU_MATCH = 0.5
BUCKET_EDGES = (0.00552, 0.01565)
BUCKETS = ("small", "medium", "large")
MODELS = ("M0", "M1", "M2", "M3")
FCOLS = {"M0": ["s2_conf"],
         "M1": ["s1_conf", "s2_conf"],
         "M2": ["s1_conf", "s2_conf", "verifier_prob"],
         "M3": ["s1_conf", "s2_conf", "verifier_prob", "s1_area_ratio",
                "s2_area_ratio", "s1s2_iou", "bbox_area_ratio",
                "bbox_aspect"]}
GRID = [round(x, 2) for x in np.arange(0.02, 0.92, 0.02)]
FINE = [0.44, 0.45, 0.46, 0.47, 0.48]


def bucket_of(r: float) -> str:
    if r < BUCKET_EDGES[0]:
        return "small"
    if r < BUCKET_EDGES[1]:
        return "medium"
    return "large"


def main() -> None:
    import torch
    import torch.nn as nn
    from torchvision import models as tvmodels
    from torchvision.transforms import functional as TF
    seg = TwoStageSegmenter("outputs/calculus_yolo11m_full/weights/best.pt",
                            "outputs/calculus_yolo11m_crop/weights/best.pt",
                            c1=C1, c2=0.05, device=0, nms_iou=NMS_IOU,
                            score_mode="s2")
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
    vb = vb.to(0).eval()
    cfgs = {m: np.load(OUT / f"fusion_{m}_coef.npy", allow_pickle=True).item()
            for m in MODELS}
    lrs = {}
    for m, cf in cfgs.items():
        lr = LogisticRegression()
        lr.coef_ = cf["coef"]; lr.intercept_ = cf["intercept"]
        lr.classes_ = np.array([0, 1])
        lrs[m] = (lr, cf["mu"], cf["sg"], cf["cols"])
    man = pd.read_csv(DATA / "split_manifest.csv")
    rows = man[man.split == "val"]
    t0 = time.time()
    imgs = []
    for _, r in rows.iterrows():
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
                gts.append({"mask": m, "bucket": bucket_of(px / float(H * W))})
        boxes, confs, masks = seg._stage1_masks(r["image_path"])
        feats, bins, rects, s2c = [], [], [], []
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
            for k in keep:
                P = k["bin"]
                m160 = cv2.resize(P, (SIDE, SIDE),
                                  interpolation=cv2.INTER_NEAREST) > 0
                s1m = cv2.resize(msk[y0:y1, x0:x1], (SIDE, SIDE),
                                 interpolation=cv2.INTER_AREA) > 0.5
                gx, gy, gw, gh = cv2.boundingRect(P.astype(np.uint8))
                rect = {"x0": x0, "y0": y0, "cw": x1 - x0, "ch": y1 - y0}
                feats.append({
                    "s1_conf": float(sc1), "s2_conf": float(k["s2_conf"]),
                    "verifier_prob": None,  # filled after batch VB inference
                    "s1_area_ratio": float(s1m.sum()) / (SIDE * SIDE),
                    "s2_area_ratio": float(m160.sum()) / (SIDE * SIDE),
                    "s1s2_iou": float((s1m & m160).sum() /
                                      max((s1m | m160).sum(), 1)),
                    "bbox_area_ratio": (gw * gh) / float(TARGET * TARGET),
                    "bbox_aspect": float(min(gw, gh) / max(gw, gh)),
                })
                bins.append(cv2.resize(P.astype(np.uint8),
                                       (rect["cw"], rect["ch"]),
                                       interpolation=cv2.INTER_NEAREST))
                rects.append(rect)
                s2c.append(float(k["s2_conf"]))
        # real verifier-B batch inference for this image's instances
        if feats:
            with torch.no_grad():
                inp = []
                for i in range(len(feats)):
                    P = bins[i]
                    if P.shape[0] != SIDE or P.shape[1] != SIDE:
                        P = cv2.resize(P, (SIDE, SIDE),
                                       interpolation=cv2.INTER_NEAREST)
                    m160 = (P > 0)
                    rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
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
        imgs.append({"H": H, "W": W, "gts": gts, "feats": feats,
                     "bins": bins, "rects": rects, "fused": {}})
    print(f"collect done {time.time()-t0:.0f}s", flush=True)
    # fused per model
    for m, (lr, mu, sg, cols) in lrs.items():
        for im in imgs:
            if not im["feats"]:
                continue
            X = np.array([[f[c] for c in cols] for f in im["feats"]]).astype(float)
            im["fused"][m] = lr.predict_proba((X - mu) / sg)[:, 1]
    n_img = len(imgs)

    def evaluate(model: str, thr: float) -> dict:
        tp = {b: 0 for b in BUCKETS}
        gt_n = {b: 0 for b in BUCKETS}
        tp_dice, gt_dice, sem = [], [], []
        n_preds = n_unm = 0
        for im in imgs:
            H, W = im["H"], im["W"]
            if not im["feats"] or model not in im["fused"]:
                continue
            fus = im["fused"][model]
            recs_all = []
            for i in range(len(im["feats"])):
                if fus[i] < thr:
                    continue
                rect = im["rects"][i]
                recs_all.append({"x0": rect["x0"], "y0": rect["y0"],
                                 "cw": rect["cw"], "ch": rect["ch"],
                                 "bin": im["bins"][i],
                                 "score": im["feats"][i]["s2_conf"]})
            keep = greedy_nms(recs_all, "mask_nms", NMS_IOU, H, W)
            preds = [rec_bin(k, H, W) for k in keep]
            n_preds += len(preds)
            gts = im["gts"]
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
            n_unm += len(preds) - len(matched)
            for pi, gi, io in matched:
                P = preds[pi]; G = gts[gi]
                inter = int((P & G["mask"]).sum())
                dc = 2 * inter / (int(P.sum()) + int(G["mask"].sum())) \
                    if (int(P.sum()) + int(G["mask"].sum())) else 1.0
                assert dc >= 0.6667 - 1e-3
                tp[G["bucket"]] += 1
                tp_dice.append(dc)
            union = np.zeros((H, W), np.uint8)
            for P in preds:
                union |= P
            for gi, G in enumerate(gts):
                gt_n[G["bucket"]] += 1
                inter = int((G["mask"] & union).sum())
                uni = int((G["mask"] | union).sum())
                io = inter / uni if uni else 0.0
                dc = 2 * inter / (int(G["mask"].sum()) + int(union.sum())) \
                    if (int(G["mask"].sum()) + int(union.sum())) else 1.0
                sem.append(dc)
                gt_dice.append(dc if gi in [g for _, g, _ in matched] else 0.0)
        tp_tot = sum(tp.values()); gt_tot = sum(gt_n.values())
        rec = tp_tot / gt_tot if gt_tot else 0.0
        prec = tp_tot / n_preds if n_preds else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return {"model": model, "thr": thr, "precision": round(prec, 4),
                "recall": round(rec, 4), "f1": round(f1, 4),
                "tp_dice": round(float(np.mean(tp_dice)), 4),
                "gt_dice": round(float(np.mean(gt_dice)), 4),
                "sem_dice": round(float(np.mean(sem)), 4),
                "fp_img": round(n_unm / n_img, 3),
                "pred_img": round(n_preds / n_img, 3),
                "r_small": round(tp["small"] / gt_n["small"], 4),
                "r_medium": round(tp["medium"] / gt_n["medium"], 4),
                "r_large": round(tp["large"] / gt_n["large"], 4)}

    rows = []
    for m in MODELS:
        for t in GRID:
            rows.append(evaluate(m, t))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exp21_ablation.csv", index=False)

    def constraint_ok(r):
        return (r["r_small"] >= 0.40 and r["r_medium"] >= 0.60
                and r["r_large"] >= 0.60 and r["tp_dice"] >= 0.78)

    print("== coarse best-F1 (constraint-passing) per model ==")
    bests = {}
    for m in MODELS:
        sub = df[(df.model == m)].copy()
        sub = sub[sub.apply(constraint_ok, axis=1)]
        if not len(sub):
            print(m, "no constraint point")
            continue
        b = sub.loc[sub.f1.idxmax()]
        bests[m] = b
        print({k: b[k] for k in ("thr", "precision", "recall", "f1",
                                 "tp_dice", "fp_img", "r_small",
                                 "r_medium", "r_large")})
    # fine scan for final candidates
    cand = [m for m in ("M1", "M3") if m in bests]
    if "M2" in bests:
        best_by_f1 = max(bests.items(), key=lambda kv: kv[1]["f1"])[0]
        if best_by_f1 == "M2":
            cand.append("M2")
    fine_rows = []
    for m in cand:
        for t in FINE:
            fine_rows.append(evaluate(m, t))
    fdf = pd.DataFrame(fine_rows)
    fdf.to_csv(OUT / "exp21_fine_scan.csv", index=False)
    print("\n== fine scan (0.44-0.48) ==")
    print(fdf.round(4).to_string(index=False))
    # decision M1 vs M3
    if "M1" in bests and "M3" in bests:
        b1 = bests["M1"]; b3 = bests["M3"]
        d_f1 = b3["f1"] - b1["f1"]
        d_rec = abs(b3["recall"] - b1["recall"])
        decision = "M3" if (d_f1 >= 0.01 and b3["recall"] >= b1["recall"] - 0.001) \
            else "M1"
        rule = ""
        if d_f1 < 0.005 and d_rec < 0.01:
            rule = "drop-verifier M1 (ΔF1<0.005 & ΔR<0.01)"
        elif d_f1 < 0.01:
            rule = "prefer-M1 simpler (0.005<=ΔF1<0.01)"
        else:
            rule = "M3 kept only if ΔF1>=0.01 & R unharmed"
        print(f"\nΔF1(M3-M1)={d_f1:.4f} Δ|R|={d_rec:.4f} -> {rule}")
        dec = {"best_per_model": {m: {k: bests[m][k] for k in
                                      ("thr", "precision", "recall", "f1",
                                       "r_small", "r_medium", "r_large",
                                       "tp_dice", "fp_img")} for m in bests},
               "d_f1_M3_M1": round(float(d_f1), 4),
               "d_rec_M3_M1": round(float(d_rec), 4),
               "decision": decision, "rule": rule,
               "frozen_model": decision,
               "frozen_thr": float(bests[decision]["thr"])}
        pd.Series(json.dumps(dec), dtype=object).to_json(OUT / "exp21_decision.json")
        print("decision:", decision, "@thr", bests[decision]["thr"])


if __name__ == "__main__":
    main()
