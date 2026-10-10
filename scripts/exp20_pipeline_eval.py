"""EXP20 main-validation end-to-end evaluation with M3 soft fusion.

Pipeline: Stage1 c1=0.05 -> 160 crop -> Stage2 V1 instances -> per-instance
Verifier-B score -> LR M3 fusion score (coefs from calibration, saved
fusion_M3_coef.npy) -> per-instance fusion threshold -> global mask-NMS ->
instance matching metrics. Threshold scan over fusion score.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from torchvision import models as tvmodels
import torch.nn as nn

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
FCOL = ["s1_conf", "s2_conf", "verifier_prob", "s1_area_ratio",
        "s2_area_ratio", "s1s2_iou", "bbox_area_ratio", "bbox_aspect"]


def bucket_of(r: float) -> str:
    if r < BUCKET_EDGES[0]:
        return "small"
    if r < BUCKET_EDGES[1]:
        return "medium"
    return "large"


def load_vb():
    m = tvmodels.efficientnet_v2_s(
        weights=tvmodels.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    conv = m.features[0][0]
    new = nn.Conv2d(4, conv.out_channels, conv.kernel_size, conv.stride,
                    conv.padding, conv.dilation, conv.groups, conv.bias)
    with torch.no_grad():
        new.weight[:, :3] = conv.weight
        new.weight[:, 3] = conv.weight.mean(dim=1)
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    m.features[0][0] = new
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    m.load_state_dict(torch.load(OUT / "verifier_B" / "best.pt",
                                 weights_only=False))
    return m.to(0).eval()


def main() -> None:
    import time
    seg = TwoStageSegmenter("outputs/calculus_yolo11m_full/weights/best.pt",
                            "outputs/calculus_yolo11m_crop/weights/best.pt",
                            c1=C1, c2=0.05, device=0, nms_iou=NMS_IOU,
                            score_mode="s2")
    vb = load_vb()
    cf = np.load(OUT / "fusion_M3_coef.npy", allow_pickle=True).item()
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.coef_ = cf["coef"]; lr.intercept_ = cf["intercept"]
    lr.classes_ = np.array([0, 1])
    mu = cf["mu"]; sg = cf["sg"]
    man = pd.read_csv(DATA / "split_manifest.csv")
    rows = man[man.split == "val"]
    # collect instances + features per image
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
                    "score": i["s2_conf"], "raw": i} for i in insts]
            keep = greedy_nms(sel, "mask_nms", NMS_IOU, TARGET, TARGET)
            for k in keep:
                P = k["bin"]
                m160 = (cv2.resize(P, (SIDE, SIDE),
                                   interpolation=cv2.INTER_NEAREST) > 0)
                s1m = (cv2.resize(msk[y0:y1, x0:x1], (SIDE, SIDE),
                                  interpolation=cv2.INTER_AREA) > 0.5)
                rect = {"x0": x0, "y0": y0, "cw": x1 - x0, "ch": y1 - y0}
                gx, gy, gw, gh = cv2.boundingRect(P.astype(np.uint8))
                feats.append({
                    "s1_conf": float(sc1), "s2_conf": float(k["s2_conf"]),
                    "s1_area_ratio": float(s1m.sum()) / (SIDE * SIDE),
                    "s2_area_ratio": float(m160.sum()) / (SIDE * SIDE),
                    "s1s2_iou": float((s1m & m160).sum() /
                                      max((s1m | m160).sum(), 1)),
                    "bbox_area_ratio": (gw * gh) / float(TARGET * TARGET),
                    "bbox_aspect": float(min(gw, gh) / max(gw, gh)),
                })
                bins.append(cv2.resize(P.astype(np.uint8), (x1 - x0, y1 - y0),
                                       interpolation=cv2.INTER_NEAREST))
                rects.append(rect)
                s2c.append(float(k["s2_conf"]))
        vp = []
        if feats:
            with torch.no_grad():
                inp = []
                for idx, P in enumerate(bins):
                    m160 = (cv2.resize(P, (SIDE, SIDE),
                                       interpolation=cv2.INTER_NEAREST) > 0)
                    rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB) if "crop_img" in dir() else None
                    t = TF.resize(torch.from_numpy(
                        cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
                        .permute(2, 0, 1).float() / 255, (224, 224),
                        TF.InterpolationMode.BILINEAR)
                    t = TF.normalize(t, (0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))
                    mm = torch.from_numpy(m160.astype(np.float32))[None]
                    mm = TF.resize(mm, (224, 224),
                                   TF.InterpolationMode.NEAREST)
                    mm = (mm > 0.5).float()
                    inp.append(torch.cat([t, mm], 0))
                vp = vb(torch.stack(inp).to(0)).squeeze(1).sigmoid().cpu().numpy()
        imgs.append({"H": H, "W": W, "gts": gts, "feats": feats,
                     "bins": bins, "rects": rects, "s2c": s2c, "vp": vp,
                     "fused": None})
    print(f"collect done {time.time()-t0:.0f}s", flush=True)
    # attach verifier prob into features
    for im in imgs:
        for i, f in enumerate(im["feats"]):
            f["verifier_prob"] = float(im["vp"][i]) if i < len(im["vp"]) else 0.0
    # fused scores
    for im in imgs:
        if im["feats"]:
            X = np.array([[f[c] for c in FCOL] for f in im["feats"]]).astype(float)
            Xs = (X - mu) / sg
            pr = lr.predict_proba(Xs)[:, 1]
            im["fused"] = pr
        else:
            im["fused"] = np.zeros(0)

    thr_list = list(np.round(np.linspace(0.02, 0.90, 23), 3))
    rows = []
    n_img = len(imgs)
    for t in thr_list:
        tp = {b: 0 for b in BUCKETS}
        gt_n = {b: 0 for b in BUCKETS}
        tp_dice, gt_dice, sem = [], [], []
        n_preds = n_unm = 0
        for im in imgs:
            H, W = im["H"], im["W"]
            recs_all = []
            for i in range(len(im["feats"])):
                if im["fused"][i] < t:
                    continue
                rect = im["rects"][i]
                b = cv2.resize(im["bins"][i], (rect["cw"], rect["ch"]),
                               interpolation=cv2.INTER_NEAREST)
                recs_all.append({"x0": rect["x0"], "y0": rect["y0"],
                                 "cw": rect["cw"], "ch": rect["ch"],
                                 "bin": b,
                                 "score": im["s2c"][i]})
            try:
                keep = greedy_nms(recs_all, "mask_nms", NMS_IOU, H, W)
            except ValueError:
                for r2 in recs_all:
                    print("rect", {k: r2[k] for k in ("x0", "y0", "cw", "ch")},
                          "bin", r2["bin"].shape)
                raise SystemExit
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
        rows.append({"thr": t, "precision": round(prec, 4),
                     "recall": round(rec, 4), "f1": round(f1, 4),
                     "tp_dice": round(float(np.mean(tp_dice)), 4),
                     "gt_dice": round(float(np.mean(gt_dice)), 4),
                     "sem_dice": round(float(np.mean(sem)), 4),
                     "fp_img": round(n_unm / n_img, 3),
                     "pred_img": round(n_preds / n_img, 3),
                     "r_small": round(tp["small"] / gt_n["small"], 4),
                     "r_medium": round(tp["medium"] / gt_n["medium"], 4),
                     "r_large": round(tp["large"] / gt_n["large"], 4)})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "exp20_main_pipeline_scan.csv", index=False)
    print("EXP20 fusion (M3) main-val scan:")
    print(d.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
