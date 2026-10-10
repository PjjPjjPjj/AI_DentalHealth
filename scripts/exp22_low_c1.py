"""EXP22: low-c1 + M3 final tuning (no deep-model retraining).

STEP1 calibration (single c1=0.02 pass, same 350-image group as EXP20:
rebuilt with identical code+seed from train split minus verifier-train/val
images -> verifier-train pool, sample 350 seed 42):
  Stage1 c1=0.02 -> 160 crop -> Stage2 V1 -> Verifier B -> per-instance
  features (same set as EXP21 M3) + label (IoU>=0.5 pos, <0.1 neg,
  amb excluded) -> refit M3 LR (same feature list) -> fusion_M3_lowc1_coef.npy
STEP2 main-val REAL runs for c1 0.05/0.03/0.02 (each full Stage1 inference)
with the new M3 params, M3 threshold scan, full metric set.
STEP3 engineering constraints: P>=0.44, medR>=0.62, largeR>=0.70,
TP-D>=0.80, FP<=0.30 -> max F1; watch smallR.
STEP4 freeze rule: adopt new c1 only if ΔF1>=0.01 or (ΔsmallR>=0.02 AND all
hard constraints hold); else keep c1=0.05 M3 thr=0.56.
No test-touch. Outputs: outputs/calculus_cls/exp22_*.csv
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
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
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

from src.inference.fusion import greedy_nms, rec_bin
from src.inference.twostage import TARGET, TwoStageSegmenter
from src.utils.paths import PROCESSED_DIR
import torch
import torch.nn as nn
from torchvision import models as tvmodels
from torchvision.transforms import functional as TF

DATA = PROCESSED_DIR / "calculus_seg"
V2 = PROCESSED_DIR / "calculus_v2"
OUT = Path("outputs/calculus_cls")
SIDE = 160
NMS_IOU = 0.5
IOU_MATCH = 0.5
BUCKET_EDGES = (0.00552, 0.01565)
BUCKETS = ("small", "medium", "large")
FCOLS = ["s1_conf", "s2_conf", "verifier_prob", "s1_area_ratio",
         "s2_area_ratio", "s1s2_iou", "bbox_area_ratio", "bbox_aspect"]
GRID = [round(x, 2) for x in np.arange(0.02, 0.92, 0.02)]
C1S = (0.05, 0.03, 0.02)


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


def collect(rows, seg, vb, c1, need_labels=True):
    """Return list of per-instance feature dicts (with label if need_labels)."""
    seg.c1 = c1
    recs = []
    for _, r in rows.iterrows():
        img = cv2.imread(r["image_path"])
        H, W = img.shape[:2]
        gtb = cv2.imread(r["mask_original_path"], cv2.IMREAD_UNCHANGED)
        gtb = (gtb > 0) if gtb is not None else np.zeros((H, W), np.uint8)
        boxes, confs, masks = seg._stage1_masks(r["image_path"])
        batch_rgb = None
        inst_local = []
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
            gtc = gtb[y0:y1, x0:x1].astype(np.uint8)
            ng, comp, _, _ = cv2.connectedComponentsWithStats(gtc, 8)
            gts640 = [(cv2.resize((comp == i).astype(np.uint8), (TARGET, TARGET),
                                  interpolation=cv2.INTER_AREA) > 0.5).astype(np.uint8)
                      for i in range(1, ng) if (comp == i).sum() >= 5]
            insts = seg._stage2_instances(crop)
            sel = [{"x0": 0, "y0": 0, "cw": TARGET, "ch": TARGET,
                    "bin": i["bin640"], "s2_conf": i["s2_conf"],
                    "score": i["s2_conf"]} for i in insts]
            keep = greedy_nms(sel, "mask_nms", NMS_IOU, TARGET, TARGET)
            for k in keep:
                P = k["bin"]
                best_iou = 0.0
                for gm in gts640:
                    inter = float((P & gm).sum())
                    uni = float((P | gm).sum())
                    best_iou = max(best_iou, inter / uni if uni else 0.0)
                m160 = (cv2.resize(P, (SIDE, SIDE),
                                   interpolation=cv2.INTER_NEAREST) > 0)
                s1m = (cv2.resize(msk[y0:y1, x0:x1], (SIDE, SIDE),
                                  interpolation=cv2.INTER_AREA) > 0.5)
                gx, gy, gw, gh = cv2.boundingRect(P.astype(np.uint8))
                inst_local.append({
                    "s1_conf": float(sc1), "s2_conf": float(k["s2_conf"]),
                    "s1_area_ratio": float(s1m.sum()) / (SIDE * SIDE),
                    "s2_area_ratio": float(m160.sum()) / (SIDE * SIDE),
                    "s1s2_iou": float((s1m & m160).sum() /
                                      max((s1m | m160).sum(), 1)),
                    "bbox_area_ratio": (gw * gh) / float(TARGET * TARGET),
                    "bbox_aspect": float(min(gw, gh) / max(gw, gh)),
                    "label": (1 if best_iou >= 0.5 else
                              (0 if best_iou < 0.1 else np.nan)),
                    "best_gt_iou": best_iou,
                    "_rgb": crop_img, "_mask": P})
                batch_rgb = crop_img
        if inst_local:
            with torch.no_grad():
                inp = []
                for it in inst_local:
                    P = it["_mask"]
                    m160 = (cv2.resize(P, (SIDE, SIDE),
                                       interpolation=cv2.INTER_NEAREST) > 0)
                    rgb = cv2.cvtColor(it["_rgb"], cv2.COLOR_BGR2RGB)
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
            for i in range(len(inst_local)):
                inst_local[i]["verifier_prob"] = float(vp[i])
        for it in inst_local:
            recs.append({k: v for k, v in it.items()
                         if not k.startswith("_")})
    return recs


def evaluate_scan(fits, lrs, mu, sg, imgs_list, n_img, tgrid):
    rows = []
    for c1, imgs in imgs_list.items():
        # fused per image
        for im in imgs:
            if not im["feats"]:
                continue
            X = np.array([[f[c] for c in FCOLS] for f in im["feats"]]).astype(float)
            im["fused"] = lrs.predict_proba((X - mu) / sg)[:, 1]
        for t in tgrid:
            tp = {b: 0 for b in BUCKETS}
            gt_n = {b: 0 for b in BUCKETS}
            tp_dice, gt_dice, sem = [], [], []
            n_preds = n_unm = 0
            for im in imgs:
                if not im["feats"]:
                    continue
                H, W = im["H"], im["W"]
                recs_all = []
                for i in range(len(im["feats"])):
                    if im["fused"][i] < t:
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
            rows.append({"c1": c1, "thr": t, "precision": round(prec, 4),
                         "recall": round(rec, 4), "f1": round(f1, 4),
                         "tp_dice": round(float(np.mean(tp_dice)), 4),
                         "gt_dice": round(float(np.mean(gt_dice)), 4),
                         "sem_dice": round(float(np.mean(sem)), 4),
                         "fp_img": round(n_unm / n_img, 3),
                         "pred_img": round(n_preds / n_img, 3),
                         "r_small": round(tp["small"] / gt_n["small"], 4),
                         "r_medium": round(tp["medium"] / gt_n["medium"], 4),
                         "r_large": round(tp["large"] / gt_n["large"], 4)})
    return pd.DataFrame(rows)


def collect_main_val(seg, vb, c1):
    """Full main-val collection with per-config timing."""
    man = pd.read_csv(DATA / "split_manifest.csv")
    rows = man[man.split == "val"]
    seg.c1 = c1
    t0 = time.time()
    n_prop = 0
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
        n_prop += len(boxes)
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
        imgs.append({"H": H, "W": W, "gts": gts, "feats": feats,
                     "bins": bins, "rects": rects, "fused": None})
    ms = (time.time() - t0) / len(rows) * 1000
    return imgs, len(rows), n_prop * 1.0 / len(rows), ms


def main() -> None:
    seg = TwoStageSegmenter("outputs/calculus_yolo11m_full/weights/best.pt",
                            "outputs/calculus_yolo11m_crop/weights/best.pt",
                            c1=0.02, c2=0.05, device=0, nms_iou=NMS_IOU,
                            score_mode="s2")
    vb = make_vb()
    # ---- STEP1 calibration rebuild (same 350 image group as EXP20) ----
    man = pd.read_csv(DATA / "split_manifest.csv")
    rows = man[man.split == "train"]
    vmeta = pd.read_csv(V2 / "verifier_dataset" / "metadata.csv")
    v_train_imgs = set(vmeta[vmeta.split == "train"].image_id)
    v_val_imgs = set(vmeta[vmeta.split == "val"].image_id)
    # NOTE: calibration MUST exclude verifier TRAIN images (in-sample verifier
    # probs leak); use the verifier-VAL image group (early-stop only, not
    # trained on) = same group that produced the frozen EXP21 M3 coefs.
    calib_pool = rows[rows.image_id.isin(v_val_imgs)]
    sel = calib_pool
    print(f"STEP1: calibration images={len(sel)} "
          f"(verifier-val group, main val/test 0 overlap)", flush=True)
    recs = collect(sel, seg, vb, c1=0.02)
    df = pd.DataFrame(recs)
    df = df[df.label.notna()].reset_index(drop=True)
    df.to_csv(OUT / "exp22_calib_features.csv", index=False)
    print(f"  calib samples={len(df)} pos={int((df.label==1).sum())} "
          f"neg={int((df.label==0).sum())}", flush=True)
    X = df[FCOLS].values.astype(float)
    y = df.label.values.astype(int)
    mu = StandardScaler().fit(X).mean_
    sg = StandardScaler().fit(X).scale_
    lr = LogisticRegression(max_iter=5000).fit(
        StandardScaler().fit(X).transform(X), y)
    prA = average_precision_score(y, lr.predict_proba(
        StandardScaler().fit(StandardScaler().fit(X).transform(X) if False else
                             ((X - mu) / sg)))[:, 1]) if False else None
    lr = LogisticRegression(max_iter=5000)
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    lr.fit(Xs, y)
    p = lr.predict_proba(Xs)[:, 1]
    print(f"  calib LR: pos-ratio={y.mean():.3f} PR-AUC="
          f"{average_precision_score(y, p):.4f}", flush=True)
    np.save(OUT / "fusion_M3_lowc1_coef.npy", {
        "coef": lr.coef_, "intercept": lr.intercept_, "cols": FCOLS,
        "mu": sc.mean_, "sg": sc.scale_}, allow_pickle=True)

    # ---- STEP2 main-val real runs ----
    imgs_list, prop_per_img, ms_data = {}, {}, {}
    for c1 in C1S:
        imgs, n_img, npp, ms = collect_main_val(seg, vb, c1)
        imgs_list[c1] = imgs
        prop_per_img[c1] = npp
        ms_data[c1] = ms
        print(f"  collected main val c1={c1}: proposals/img={npp:.3f} "
              f"ms/img={ms:.0f}", flush=True)
    sc2 = StandardScaler().fit(np.array([[0.0] * len(FCOLS)]))
    cf = np.load(OUT / "fusion_M3_lowc1_coef.npy", allow_pickle=True).item()
    lr2 = LogisticRegression()
    lr2.coef_ = cf["coef"]; lr2.intercept_ = cf["intercept"]
    lr2.classes_ = np.array([0, 1])
    scan = evaluate_scan(None, lr2, cf["mu"], cf["sg"], imgs_list,
                         n_img, GRID)
    scan["proposals_img"] = scan.c1.map(prop_per_img)
    scan["ms_img"] = scan.c1.map(ms_data)
    scan.to_csv(OUT / "exp22_main_scan.csv", index=False)

    print("\nEXP22 scan (coarse). constraints: P>=0.44 medR>=0.62 "
          "largeR>=0.70 TP>=0.80 FP<=0.30")
    for c1 in C1S:
        sub = scan[scan.c1 == c1]
        ok = sub[(sub.precision >= 0.44) & (sub.r_medium >= 0.62) &
                 (sub.r_large >= 0.70) & (sub.tp_dice >= 0.80) &
                 (sub.fp_img <= 0.30)]
        if len(ok):
            b = ok.loc[ok.f1.idxmax()]
            print(f"c1={c1} best-constraint: ", {k: b[k] for k in
                  ("thr", "precision", "recall", "f1", "tp_dice", "fp_img",
                   "r_small", "r_medium", "r_large", "proposals_img",
                   "ms_img")})
        else:
            print(f"c1={c1}: no constraint-passing point; global best: ",
                  {k: sub.loc[sub.f1.idxmax(), k] for k in
                   ("thr", "precision", "recall", "f1", "r_small",
                    "r_medium", "r_large", "fp_img")})


if __name__ == "__main__":
    main()
