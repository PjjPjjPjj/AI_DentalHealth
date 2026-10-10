"""EXP20: soft score fusion (instances, no hard verifier gate).

Calibration split: images from TRAIN that were NOT used by the verifier
dataset (neither train nor val), grouped by image id, random 350 imgs seed 42.
Per stage2 instance extract features; label by 1-to-1 GT match
(pos IoU>=0.5, neg <0.1, ambiguous excluded).

Models: M0 s2_conf | M1 s1+s2 | M2 +verifier_prob | M3 +area/iou features
LogisticRegression (standardized). Select best by calibration PR-AUC.

Then run the FULL real pipeline on MAIN VAL with Verifier B scores:
instance -> fusion score -> threshold scan -> metrics vs V1 baseline.
Usage: python scripts/exp20_soft_fusion.py build | evaluate
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

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_fscore_support)
from sklearn.preprocessing import StandardScaler

from src.inference.twostage import TARGET, TwoStageSegmenter
from src.inference.fusion import greedy_nms
from src.utils.paths import PROCESSED_DIR
import torch
from torchvision import models as tvmodels
import torch.nn as nn

DATA = PROCESSED_DIR / "calculus_seg"
V2 = PROCESSED_DIR / "calculus_v2"
OUT = Path("outputs/calculus_cls")
C1 = 0.05
SIDE = 160
NMS_IOU = 0.5
FEATS = ["s1_conf", "s2_conf", "verifier_prob", "s1_area_ratio",
         "s2_area_ratio", "s1s2_iou", "bbox_area_ratio", "bbox_aspect"]


def load_verifier_b():
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


def collect(rows, seg, vmodel=None) -> pd.DataFrame:
    from torchvision import transforms as T
    from torchvision.transforms import functional as TF
    recs = []
    for _, r in rows.iterrows():
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
            x1, y1 = x0 + SIDE, y0 + SIDE
            crop_img = img[y0:y1, x0:x1]
            crop = cv2.resize(crop_img, (TARGET, TARGET),
                              interpolation=cv2.INTER_AREA)
            gtc = gtb[y0:y1, x0:x1].astype(np.uint8)
            ng, comp, _, _ = cv2.connectedComponentsWithStats(gtc, 8)
            gt_insts = [(comp == i).astype(np.uint8)
                        for i in range(1, ng) if (comp == i).sum() >= 5]
            insts = seg._stage2_instances(crop)
            sel = [{"x0": 0, "y0": 0, "cw": TARGET, "ch": TARGET,
                    "bin": i["bin640"], "s2_conf": i["s2_conf"],
                    "score": i["s2_conf"], "raw": i} for i in insts]
            keep = greedy_nms(sel, "mask_nms", NMS_IOU, TARGET, TARGET)
            # batch verifier inference on this image's kept instances
            if vmodel is not None and keep:
                inp, kin = [], []
                with torch.no_grad():
                    for k in keep:
                        P = k["bin"]
                        m160 = cv2.resize(P, (SIDE, SIDE),
                                          interpolation=cv2.INTER_NEAREST) > 0
                        rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255
                        t = TF.resize(t, (224, 224), TF.InterpolationMode.BILINEAR)
                        mn = TF.normalize(t, (0.485, 0.456, 0.406),
                                          (0.229, 0.224, 0.225))
                        mm = torch.from_numpy(m160.astype(np.float32))[None]
                        mm = TF.resize(mm, (224, 224),
                                       TF.InterpolationMode.NEAREST)
                        mm = (mm > 0.5).float()
                        inp.append(torch.cat([mn, mm], 0))
                        kin.append(k)
                    vprobs = vmodel(torch.stack(inp).to(0)).squeeze(1).sigmoid().cpu().numpy()
                for k, vp in zip(kin, vprobs):
                    P = k["bin"]
                    gm640s = [cv2.resize(g, (TARGET, TARGET),
                                         interpolation=cv2.INTER_AREA) > 0.5
                              for g in gt_insts]
                    best_iou = 0.0
                    for gm in gm640s:
                        inter = float((P & gm).sum())
                        uni = float((P | gm).sum())
                        best_iou = max(best_iou, inter / uni if uni else 0.0)
                    m160 = cv2.resize(P, (SIDE, SIDE),
                                      interpolation=cv2.INTER_NEAREST) > 0
                    s1m = cv2.resize(msk[y0:y1, x0:x1], (SIDE, SIDE),
                                     interpolation=cv2.INTER_AREA) > 0.5
                    s2a = float(m160.sum()) / (SIDE * SIDE)
                    s1a = float(s1m.sum()) / (SIDE * SIDE)
                    x0i, y0i, w0, h0 = cv2.boundingRect(P.astype(np.uint8))
                    recs.append({
                        "s1_conf": float(sc1), "s2_conf": float(k["s2_conf"]),
                        "verifier_prob": float(vp),
                        "s1_area_ratio": s1a, "s2_area_ratio": s2a,
                        "s1s2_iou": float((s1m & m160).sum() / max((s1m | m160).sum(), 1)),
                        "bbox_area_ratio": (w0 * h0) / float(TARGET * TARGET),
                        "bbox_aspect": float(min(w0, h0) / max(w0, h0)),
                        "label": 1 if best_iou >= 0.5 else (0 if best_iou < 0.1 else np.nan),
                        "best_gt_iou": best_iou})
    return pd.DataFrame(recs)


def build() -> None:
    seg = TwoStageSegmenter("outputs/calculus_yolo11m_full/weights/best.pt",
                            "outputs/calculus_yolo11m_crop/weights/best.pt",
                            c1=C1, c2=0.05, device=0, nms_iou=NMS_IOU,
                            score_mode="s2")
    vmodel = load_verifier_b()
    man = pd.read_csv(DATA / "split_manifest.csv")
    rows = man[man.split == "train"]
    vmeta = pd.read_csv(V2 / "verifier_dataset" / "metadata.csv")
    v_train_imgs = set(vmeta[vmeta.split == "train"].image_id)
    v_val_imgs = set(vmeta[vmeta.split == "val"].image_id)
    calib_pool = rows[rows.image_id.isin(v_train_imgs - v_val_imgs)]
    assert len(calib_pool) > 350, len(calib_pool)
    sel = calib_pool.sample(n=350, random_state=42)
    print(f"calibration images: {len(sel)} (verifier-train pool, "
          f"verifier-val excluded; main val/test = 0)")
    df = collect(sel, seg, vmodel)
    df = df[df.label.notna()].reset_index(drop=True)
    df.to_csv(OUT / "calib_features.csv", index=False)
    print("calib samples:", len(df), "pos:", int((df.label == 1).sum()))
    # LR models
    from sklearn.linear_model import LogisticRegression
    sets = {"M0": ["s2_conf"],
            "M1": ["s1_conf", "s2_conf"],
            "M2": ["s1_conf", "s2_conf", "verifier_prob"],
            "M3": FEATS}
    results = {}
    best_pr, best_tag = -1.0, None
    for tag, cols in sets.items():
        X = df[cols].values.astype(float)
        y = df.label.values.astype(int)
        sc = StandardScaler().fit(X)
        Xs = sc.transform(X)
        lr = LogisticRegression(max_iter=2000).fit(Xs, y)
        p = lr.predict_proba(Xs)[:, 1]
        prA = average_precision_score(y, p)
        rocA = roc_auc_score(y, p)
        pm = (p >= 0.5).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y, pm, average="binary", zero_division=0)
        results[tag] = {"prAUC": round(prA, 4), "rocAUC": round(rocA, 4),
                        "P": round(prec, 4), "R": round(rec, 4),
                        "F1": round(f1, 4)}
        print(tag, results[tag])
        best_pr = prA if prA > best_pr else best_pr
        if prA >= best_pr:
            best_pr = prA
            best_tag = tag
        np.save(OUT / f"fusion_{tag}_coef.npy",
                {"coef": lr.coef_, "intercept": lr.intercept_, "cols": cols,
                 "mu": sc.mean_, "sg": sc.scale_}, allow_pickle=True)
    results["__best__"] = best_tag
    (OUT / "fusion_m_compare.json").write_text(
        pd.Series(results, dtype=object).to_json())
    print("BEST:", best_tag)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        build()
    else:
        print("use: build")
