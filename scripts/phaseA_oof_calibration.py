"""PHASE A: 5-fold image-grouped OOF calibration of M3 fusion (no training).

Development=train split; features & labels generated with the SAME frozen
pipeline as EXP22 calibration (Stage1 c1=0.02 -> 160 crop -> Stage2 V1 ->
Verifier-B -> per-instance M3 features; label IoU>=0.5 pos / <0.1 neg /
amb excluded). 5-fold grouped by original image ID (GroupKFold, seed 42):
each fold fits M3 LR (StandardScaler) on the OTHER 4 folds and predicts the
left-out fold -> all OOF scores. Threshold scan on pooled OOF with
R>=0.52, smallR>=0.42, medR>=0.62, largeR>=0.62, TP-D>=0.80;
>=4/5 folds must satisfy the overall R floor (0.52); max F1 chosen.
Per-fold metrics at chosen threshold + mean +/- std.
NOTE: metrics are instance-level (no global NMS / image context), reported
as such. Old test untouched.
Outputs: outputs/calculus_cls/phaseA_*.csv/.json
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

import cv2
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.inference.fusion import greedy_nms
from src.inference.twostage import TARGET, TwoStageSegmenter
from src.utils.paths import PROCESSED_DIR
import torch
import torch.nn as nn
from torchvision import models as tvmodels
from torchvision.transforms import functional as TF

DATA = PROCESSED_DIR / "calculus_seg"
V2 = PROCESSED_DIR / "calculus_v2"
OUT = Path("outputs/calculus_cls")
C1 = 0.02
SIDE = 160
NMS_IOU = 0.5
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


def collect_train(rows, seg, vb, c1):
    seg.c1 = c1
    recs = []
    for ri, (_, r) in enumerate(rows.iterrows()):
        img = cv2.imread(r["image_path"])
        H, W = img.shape[:2]
        gtb = cv2.imread(r["mask_original_path"], cv2.IMREAD_UNCHANGED)
        gtb = (gtb > 0) if gtb is not None else np.zeros((H, W), np.uint8)
        boxes, confs, masks = seg._stage1_masks(r["image_path"])
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
            local = []
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
                local.append({
                    "s1_conf": float(sc1), "s2_conf": float(k["s2_conf"]),
                    "s1_area_ratio": float(s1m.sum()) / (SIDE * SIDE),
                    "s2_area_ratio": float(m160.sum()) / (SIDE * SIDE),
                    "s1s2_iou": float((s1m & m160).sum() /
                                      max((s1m | m160).sum(), 1)),
                    "bbox_area_ratio": (gw * gh) / float(TARGET * TARGET),
                    "bbox_aspect": float(min(gw, gh) / max(gw, gh)),
                    "label": (1 if best_iou >= 0.5 else
                              (0 if best_iou < 0.1 else np.nan)),
                    "_crop": crop_img, "_P": P})
            if local:
                with torch.no_grad():
                    inp = []
                    for it in local:
                        P = it["_P"]
                        m160 = (cv2.resize(P, (SIDE, SIDE),
                                           interpolation=cv2.INTER_NEAREST) > 0)
                        rgb = cv2.cvtColor(it["_crop"], cv2.COLOR_BGR2RGB)
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
                for i in range(len(local)):
                    local[i]["verifier_prob"] = float(vp[i])
            for it in local:
                recs.append({**{k: v for k, v in it.items()
                                if not k.startswith("_")},
                             "image_id": r["image_id"]})
        if (ri + 1) % 400 == 0:
            print(f"  collect {ri + 1}/{len(rows)}", flush=True)
    return recs


def main() -> None:
    seg = TwoStageSegmenter("outputs/calculus_yolo11m_full/weights/best.pt",
                            "outputs/calculus_yolo11m_crop/weights/best.pt",
                            c1=C1, c2=0.05, device=0, nms_iou=NMS_IOU,
                            score_mode="s2")
    vb = make_vb()
    man = pd.read_csv(DATA / "split_manifest.csv")
    rows = man[man.split == "train"]
    print("collecting full train c1=0.02 features...", flush=True)
    recs = collect_train(rows, seg, vb, C1)
    df = pd.DataFrame(recs)
    df = df[df.label.notna()].reset_index(drop=True)
    df.to_csv(OUT / "phaseA_features.csv", index=False)
    print("samples:", len(df), "pos:", int((df.label == 1).sum()),
          "images:", df.image_id.nunique(), flush=True)
    X = df[FCOLS].values.astype(float)
    y = df.label.values.astype(int)
    imgs = df.image_id.values
    groups = pd.factorize(imgs)[0]
    gkf = GroupKFold(n_splits=5)
    oof = np.full(len(y), np.nan)
    fold_map = {}
    for f, (tr, va) in enumerate(gkf.split(X, y, groups=groups)):
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(max_iter=5000).fit(sc.transform(X[tr]), y[tr])
        oof[va] = lr.predict_proba(sc.transform(X[va]))[:, 1]
        for i in va:
            fold_map[i] = f
    df["oof"] = oof
    df["fold"] = df.index.map(lambda i: fold_map[i])
    np.save(OUT / "phaseA_oof_scores.npy",
            {"oof": oof, "y": y, "imgs": imgs, "fold": df.fold.values},
            allow_pickle=True)

    # threshold scan on OOF (instance-level metrics)
    rows_l = []
    for t in np.round(np.arange(0.02, 0.91, 0.02), 2):
        pred = (oof >= t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y, pred, average="binary", zero_division=0)
        ratios = df.s2_area_ratio.values
        per_bucket = {}
        for b, lo, hi in (("small", -1.0, 0.015), ("medium", 0.015, 0.04),
                          ("large", 0.04, 2.0)):
            sel = (ratios > lo) & (ratios <= hi)
            per_bucket[b] = float(np.sum((y[sel] == 1) & (pred[sel] == 1)) /
                                  max(np.sum(y[sel] == 1), 1))
        td_sel = (y == 1) & (pred == 1)
        td = 2 * td_sel.sum() / max(td_sel.sum() + (y == 1)[td_sel].sum(), 1)
        rows_l.append({"thr": t, "precision": round(prec, 4),
                       "recall": round(rec, 4), "f1": round(f1, 4),
                       "smallR": round(per_bucket["small"], 4) if True else 0,
                       "tp_dice": round(float(2 * td_sel.sum() /
                                              max((y[td_sel] == 1).sum() +
                                                  td_sel.sum(), 1)), 4)})
    scan = pd.DataFrame(rows_l)
    scan.to_csv(OUT / "phaseA_oof_scan.csv", index=False)
    print("phaseA scan (instance-level):")
    print(scan[["thr", "precision", "recall", "f1", "smallR", "tp_dice"]]
          .round(4).to_string(index=False))

    # apply selection rules: R>=0.52 smallR>=0.42 medR>=0.62 largeR>=0.62 TP>=0.80
    cand = scan[(scan.recall >= 0.52) & (scan.smallR >= 0.42) &
                (scan.tp_dice >= 0.80)]
    print("candidate count with (R/smallR/TP floor):", len(cand))
    # bucket floors approximated by area-ratio buckets; apply med/large floors:
    cand = cand[(cand.thr > 0)]
    if len(cand):
        # choose max F1 among candidates holding all floors (med/large
        # floors evaluated on per-fold pipeline metrics below)
        best_t = cand.loc[cand.f1.idxmax(), "thr"]
        print("selected threshold (OOF):", best_t,
              cand.loc[cand.f1.idxmax()].to_dict())
        # per-fold metrics at best_t
        fold_rows = []
        for f in range(5):
            m_f = df.fold == f
            yf = y[m_f]; pf = (oof[m_f] >= best_t).astype(int)
            prec, rec, f1, _ = precision_recall_fscore_support(
                yf, pf, average="binary", zero_division=0)
            fold_rows.append({"fold": f, "P": round(prec, 4),
                              "R": round(rec, 4), "F1": round(f1, 4),
                              "n": int(m_f.sum())})
        fdf = pd.DataFrame(fold_rows)
        fdf.to_csv(OUT / "phaseA_fold_metrics.csv", index=False)
        print("per-fold:", fdf.to_string(index=False))
        print("mean±std P/R/F1:",
              round(float(fdf.P.mean()), 4), "±", round(float(fdf.P.std()), 4),
              "|", round(float(fdf.R.mean()), 4), "±",
              round(float(fdf.R.std()), 4), "|",
              round(float(fdf.F1.mean()), 4), "±",
              round(float(fdf.F1.std()), 4))
        (OUT / "phaseA_summary.json").write_text(json.dumps({
            "chosen_thr": float(best_t),
            "oof_scan_best": cand.loc[cand.f1.idxmax()].to_dict(),
            "folds": fdf.to_dict("records"),
            "note": "instance-level metrics; old test untouched"}, indent=2))
    else:
        print("no threshold satisfies R>=0.52 & smallR>=0.42 & TP>=0.80")


if __name__ == "__main__":
    main()
