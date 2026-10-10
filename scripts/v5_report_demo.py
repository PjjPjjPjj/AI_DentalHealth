"""Render V5 report demo panels + dataset montage from v5_demo_picks.json.

Panels per bucket (outputs/demo/v5_report/):
  demo_<bucket>_<id>_compare.jpg  [GT overlay | TP(green)/FN(red)/FP(blue) map]
  demo_miss_<id>_compare.jpg      failure case (limitation slide)
  dataset_montage.jpg             6 train examples with GT overlay (red)
  v5_demo_summary.json            per-image numbers for slides
All panels reuse the frozen V5 decision logic (thr_std/thr_large/cascade).
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

import scripts.v3_pipeline_lib as _lib
_lib.OUT = Path("outputs/calculus_v5_cls")
_lib.C1 = 0.005
from scripts.v3_pipeline_lib import (DATA, collect_split, greedy_nms,
                                     image_gts, make_verifier, rec_bin)
from ultralytics import YOLO

OUT_DIR = Path("outputs/demo/v5_report")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PICKS = json.load(open(OUT_DIR / "v5_demo_picks.json"))
THR_STD, THR_LARGE, LARGE_EDGE = 0.6, 0.45, 0.01565
NMS_IOU = 0.5

s1m = YOLO("outputs/calculus_v5_stage1/weights/best.pt")
s2m = YOLO("outputs/calculus_v3b_stage2/weights/best.pt")
vb = make_verifier()
man = pd.read_csv(DATA / "split_manifest_v3.csv")
ids = [str(PICKS[k]["id"]) for k in ("small", "medium", "large", "miss")]
mc = man[man.image_id.astype(str).isin(ids) & (man.split == "test")].copy()
print("rows:", mc.image_id.tolist(), flush=True)

imgs = collect_split(s1m, s2m, mc, "test", with_verifier=True, vb=vb)
mc = mc.reset_index(drop=True)
by_id = {str(mc.iloc[i].image_id): imgs[i] for i in range(len(imgs))}

fz = np.load(_lib.OUT / "fusion_v3.npz")
lr = LogisticRegression()
lr.coef_, lr.intercept_, lr.classes_ = fz["coef"], fz["intercept"], np.array([0, 1])

summary = []
for key in ("small", "medium", "large", "miss"):
    pick = PICKS[key]
    im = by_id[str(pick["id"])]
    H, W = im["H"], im["W"]
    f = im["feats"]
    preds, confs = [], []
    fused_max = 0.0
    if f:
        X = np.array([[x[c] for c in _lib.FCOLS] for x in f], float)
        fused = lr.predict_proba((X - fz["mu"]) / fz["sg"])[:, 1]
        s1c = np.array([x["s1_conf"] for x in f])
        ar = np.array([x["s2_area_ratio"] for x in f])
        thr = np.where(ar >= LARGE_EDGE, THR_LARGE, THR_STD)
        fused_max = float(fused.max())
        ok = (s1c >= 0.02) & (fused >= thr)
        if ok.any():
            recs = []
            for j in np.argwhere(ok).ravel():
                rect = im["rects"][j]
                recs.append({"x0": rect["x0"], "y0": rect["y0"], "cw": rect["cw"],
                             "ch": rect["ch"], "bin": im["bins"][j],
                             "score": im["feats"][j]["s2_conf"]})
            kept = greedy_nms(recs, "mask_nms", NMS_IOU, H, W)
            for k in kept:
                preds.append(rec_bin(k, H, W))
                confs.append(k["score"])
    gt_bin = np.zeros((H, W), np.uint8)
    for g in im["gts"]:
        gt_bin |= g["mask"]
    pred_bin = np.zeros((H, W), np.uint8)
    for p in preds:
        pred_bin |= p

    tp = int((pred_bin & gt_bin).sum())
    fp_px = int((pred_bin & (1 - gt_bin)).sum())
    fn_px = int(((1 - pred_bin) & gt_bin).sum())
    dice = 2 * tp / max(int(pred_bin.sum()) + int(gt_bin.sum()), 1)

    img = cv2.imread(pick["img"])
    # panel 1: GT
    p1 = img.copy()
    p1[gt_bin > 0] = (0.45 * np.array([0, 0, 255]) + 0.55 * p1[gt_bin > 0]).astype(np.uint8)
    cnts, _ = cv2.findContours(gt_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(p1, cnts, -1, (0, 0, 255), 2)
    # panel 2: TP/FN/FP
    p2 = img.copy()
    m_tp = (pred_bin > 0) & (gt_bin > 0)
    m_fn = (pred_bin == 0) & (gt_bin > 0)
    m_fp = (pred_bin > 0) & (gt_bin == 0)
    p2[m_tp] = (0.45 * np.array([0, 180, 0]) + 0.55 * p2[m_tp]).astype(np.uint8)
    p2[m_fn] = (0.55 * np.array([0, 0, 255]) + 0.45 * p2[m_fn]).astype(np.uint8)
    p2[m_fp] = (0.45 * np.array([255, 80, 0]) + 0.55 * p2[m_fp]).astype(np.uint8)
    pc, _ = cv2.findContours(pred_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(p2, pc, -1, (0, 255, 0), 1)
    scale = 560 / max(H, W)
    p1 = cv2.resize(p1, (int(W * scale), int(H * scale)))
    p2 = cv2.resize(p2, (int(W * scale), int(H * scale)))
    gap = np.full((p1.shape[0], 14, 3), 255, np.uint8)
    canvas = np.hstack([p1, gap, p2])
    out_path = OUT_DIR / (f"demo_{key}_{pick['id']}_compare.jpg")
    cv2.imwrite(str(out_path), canvas)

    summary.append(dict(demo=key, image_id=str(pick["id"]), file=str(out_path),
                        W=W, H=H, gt_instances=len(im["gts"]),
                        gt_px=int(gt_bin.sum()), pred_px=int(pred_bin.sum()),
                        n_pred=len(preds), confs=[round(float(c), 3) for c in confs],
                        pixel_dice=round(dice, 4),
                        precision_px=round(tp / max(tp + fp_px, 1), 4),
                        recall_px=round(tp / max(tp + fn_px, 1), 4),
                        fused_max=round(fused_max, 4)))
    print(f"[{key}] {pick['id']}: dice={dice:.3f} pred={len(preds)} confs={[round(c,2) for c in confs]}", flush=True)

# --- dataset montage: 6 calculus-positive train examples ---
tr = man[(man.split == "train") & (man.has_calculus == True)].reset_index(drop=True)
cells = []
seen = set()
for dom_quota in [("plain", "caries", "gingivitis")] * 2 + [("caries", "plain")]:
    for dom in dom_quota:
        got = 0
        for _, r in tr.iterrows():
            if got >= 1:
                break
            if r.image_id in seen:
                continue
            d = "caries" if str(r.image_id).startswith("caries") else \
                "gingivitis" if str(r.image_id).startswith("gingivitis") else "plain"
            if d != dom:
                continue
            img = cv2.imread(r.image_path)
            if img is None:
                continue
            H, W = img.shape[:2]
            gts, _ = image_gts(r.image_path, str(_lib.RAW / f"{r.image_id}.png"), H, W)
            gtb = np.zeros((H, W), np.uint8)
            for g in gts:
                gtb |= g["mask"]
            if gtb.sum() == 0:
                continue
            img = img.copy()
            img[gtb > 0] = (0.45 * np.array([0, 0, 255]) + 0.55 * img[gtb > 0]).astype(np.uint8)
            cnts, _ = cv2.findContours(gtb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, (0, 0, 255), 1)
            cells.append((f"{r.image_id} [{d}]", img))
            seen.add(r.image_id)
            got += 1
CW, CH, LBL = 300, 215, 20
cells = cells[:6]
rows_n = (len(cells) + 2) // 3
grid = np.full((rows_n * (CH + LBL + 8) + 4, 3 * CW + 24, 3), 255, np.uint8)
for i, (lbl, img) in enumerate(cells[:6]):
    rr, cc = divmod(i, 3)
    y0, x0 = rr * (CH + LBL + 8) + 4, cc * (CW + 8) + 4
    sc = min(CW / img.shape[1], CH / img.shape[0])
    rs = cv2.resize(img, (int(img.shape[1] * sc), int(img.shape[0] * sc)))
    grid[y0:y0 + rs.shape[0], x0:x0 + rs.shape[1]] = rs
    cv2.putText(grid, lbl, (x0, y0 + CH + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1)
cv2.imwrite(str(OUT_DIR / "dataset_montage.jpg"), grid)
print("montage cells:", len(cells))

json.dump(dict(params=dict(thr_std=THR_STD, thr_large=THR_LARGE, large_edge=LARGE_EDGE,
                           s1="outputs/calculus_v5_stage1/weights/best.pt",
                           s2="outputs/calculus_v3b_stage2/weights/best.pt")),
          open(OUT_DIR / "v5_demo_params.json", "w"), indent=1)
json.dump(summary, open(OUT_DIR / "v5_demo_summary.json", "w"), ensure_ascii=False, indent=1)
print("summary saved")
