"""Scan V5 test split: find per-bucket (small/medium/large) clean TP demo images.

Caches collect_split results to outputs/demo/v5_report/test_cache.npz
Selection per bucket: GT instance of that bucket matched (IoU>=0.5) by a
prediction, highest pixel dice, prefer single-GT images and larger images.
Also keeps one representative failure case (GT present, nothing selected).
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

import scripts.v3_pipeline_lib as _lib
_lib.OUT = Path("outputs/calculus_v5_cls")
_lib.C1 = 0.005
from scripts.v3_pipeline_lib import (BUCKETS, DATA, collect_split, greedy_nms,
                                     make_verifier, rec_bin)
from ultralytics import YOLO
from sklearn.linear_model import LogisticRegression

OUT_DIR = Path("outputs/demo/v5_report")
CACHE = OUT_DIR / "test_cache.npz"
IOU_MATCH = 0.5
THR_STD, THR_LARGE, LARGE_EDGE = 0.6, 0.45, 0.01565
NMS_IOU = 0.5

s1m = YOLO("outputs/calculus_v5_stage1/weights/best.pt")
s2m = YOLO("outputs/calculus_v3b_stage2/weights/best.pt")
vb = make_verifier()
man = pd.read_csv(DATA / "split_manifest_v3.csv")
tc = man[(man.split == "test") & (man.has_calculus == True)].reset_index(drop=True)
print(f"test calculus images: {len(tc)}", flush=True)

imgs = collect_split(s1m, s2m, tc, "test", with_verifier=True, vb=vb, log_every=50)

fz = np.load(_lib.OUT / "fusion_v3.npz")
lr = LogisticRegression()
lr.coef_, lr.intercept_, lr.classes_ = fz["coef"], fz["intercept"], np.array([0, 1])

rows = []
for i, im in enumerate(imgs):
    r = tc.iloc[i]
    H, W = im["H"], im["W"]
    f = im["feats"]
    preds = []
    fused_max = 0.0
    if f:
        X = np.array([[x[c] for c in _lib.FCOLS] for x in f], float)
        fused = lr.predict_proba((X - fz["mu"]) / fz["sg"])[:, 1]
        s1c = np.array([x["s1_conf"] for x in f])
        ar = np.array([x["s2_area_ratio"] for x in f])
        thr = np.where(ar >= LARGE_EDGE, THR_LARGE, THR_STD)
        ok = (s1c >= 0.02) & (fused >= thr)
        fused_max = float(fused.max()) if len(fused) else 0.0
        if ok.any():
            recs = []
            for j in np.argwhere(ok).ravel():
                rect = im["rects"][j]
                recs.append({"x0": rect["x0"], "y0": rect["y0"], "cw": rect["cw"],
                             "ch": rect["ch"], "bin": im["bins"][j],
                             "score": im["feats"][j]["s2_conf"]})
            preds = [rec_bin(k, H, W) for k in greedy_nms(recs, "mask_nms", NMS_IOU, H, W)]
    for G in im["gts"]:
        best_iou, best_dice = 0.0, 0.0
        for P in preds:
            inter = int((P & G["mask"]).sum())
            uni = int((P | G["mask"]).sum())
            if uni:
                iou = inter / uni
                best_iou = max(best_iou, iou)
                best_dice = max(best_dice, 2 * inter / max(int(P.sum()) + int(G["mask"].sum()), 1))
        rows.append(dict(image_id=str(r.image_id), img=str(r.image_path),
                         W=W, H=H, n_gt=len(im["gts"]), n_pred=len(preds),
                         bucket=G["bucket"], gt_px=G["px"], fused_max=round(fused_max, 4),
                         iou=round(best_iou, 4), dice=round(best_dice, 4),
                         matched=best_iou >= IOU_MATCH))
    if not im["gts"]:
        rows.append(dict(image_id=str(r.image_id), img=str(r.image_path), W=W, H=H,
                         n_gt=0, n_pred=len(preds), bucket="none", gt_px=0,
                         fused_max=round(fused_max, 4), iou=0.0, dice=0.0, matched=False))

d = pd.DataFrame(rows)
d.to_csv(OUT_DIR / "v5_test_scan.csv", index=False)

picks = {}
for b in ["small", "medium", "large"]:
    sub = d[(d.bucket == b) & d.matched].copy()
    sub = sub.sort_values(["dice", "n_gt", "W"], ascending=[False, True, False])
    if len(sub):
        r = sub.iloc[0]
        picks[b] = dict(id=r.image_id, img=r.img, px=int(r.gt_px), H=int(r.H), W=int(r.W),
                        dice=float(r.dice), iou=float(r.iou), fused=float(r.fused_max),
                        n_gt=int(r.n_gt), n_pred=int(r.n_pred))
miss = d[(d.bucket != "none") & (~d.matched) & (d.n_pred == 0)]
miss = miss.sort_values(["bucket", "W"], ascending=[True, False])
if len(miss):
    r = miss.iloc[0]
    picks["miss"] = dict(id=r.image_id, img=r.img, px=int(r.gt_px), H=int(r.H), W=int(r.W),
                         bucket=r.bucket)
json.dump(picks, open(OUT_DIR / "v5_demo_picks.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(picks, ensure_ascii=False, indent=1))
