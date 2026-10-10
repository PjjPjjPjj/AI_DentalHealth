"""Trace the 13 large-lesion FNs through the v5 cascade stages.

For each GT: stage1 best covering candidate (conf/cover) -> s2 instance IoU
-> verifier prob -> fused score -> selected(thr_large=0.45) or not.
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
import torch
from ultralytics import YOLO

import scripts.v3_pipeline_lib as _lib
_lib.C1 = 0.005
from scripts.v3_pipeline_lib import (BUCKETS, FCOLS, SIDE, TARGET, collect_split,
                                     make_verifier, s1_pass, s2_instances)
from src.inference.fusion import greedy_nms, rec_bin

IDS = ["216", "74", "caries_0_115", "caries_0_1652", "caries_0_2747",
       "caries_0_3129", "caries_0_4307", "caries_0_5873", "caries_0_5959",
       "caries_0_8180", "caries_0_835", "caries_0_7630", "caries_0_7816"]
THR_LARGE = 0.45
IOU_MATCH = 0.5


def gt_iou(P, G):
    inter = int((P & G).sum())
    uni = int((P | G).sum())
    return inter / uni if uni else 0.0


def main() -> None:
    s1m = YOLO("outputs/calculus_v5_stage1/weights/best.pt")
    s2m = YOLO("outputs/calculus_v3b_stage2/weights/best.pt")
    vb = make_verifier()
    fz = np.load(_lib.OUT / "fusion_v3.npz")
    mu, sg, coef, b0 = fz["mu"], fz["sg"], fz["coef"][0], float(fz["intercept"][0])
    man = pd.read_csv(_lib.DATA / "split_manifest_v3.csv")
    rows = man[(man.split == "test") & (man.image_id.isin(IDS))].set_index("image_id")

    report = {}
    for iid in IDS:
        r = rows.loc[iid]
        img = cv2.imread(r["image_path"])
        H, W = img.shape[:2]
        m = cv2.imread(r["mask_original_path"], cv2.IMREAD_UNCHANGED)
        if m is None:
            continue
        if m.ndim == 3:
            m = m[:, :, 0]
        calc = (m > 0).astype(np.uint8)
        n, comp = cv2.connectedComponents(calc, 8)
        gts = []
        for i in range(1, n):
            px = int((comp == i).sum())
            if px >= 5:
                gts.append({"mask": (comp == i).astype(np.uint8), "px": px,
                            "bucket": _lib.bucket_of(px / float(H * W))})
        boxes, confs, masks = s1_pass(s1m, r["image_path"])
        # per-candidate: s2 instances placed, verifier, fused
        cands = []
        for ci, (b, sc1, msk) in enumerate(zip(boxes, confs, masks)):
            bx = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
            x0 = int(round((bx[0] + bx[2]) / 2 - SIDE / 2))
            y0 = int(round((bx[1] + bx[3]) / 2 - SIDE / 2))
            x0 = max(0, min(x0, W - SIDE)); y0 = max(0, min(y0, H - SIDE))
            x1, y1 = min(x0 + SIDE, W), min(y0 + SIDE, H)
            crop_img = img[y0:y1, x0:x1]
            crop = cv2.resize(crop_img, (TARGET, TARGET), interpolation=cv2.INTER_AREA)
            insts = s2_instances(s2m, crop)
            sx, sy = (x1 - x0) / TARGET, (y1 - y0) / TARGET
            for it in insts:
                P = np.zeros((H, W), np.uint8)
                m6 = cv2.resize(it["bin640"], (int(TARGET * sx), int(TARGET * sy)),
                                interpolation=cv2.INTER_NEAREST)
                P[y0:y0 + m6.shape[0], x0:x0 + m6.shape[1]] = m6
                rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255
                from torchvision.transforms import functional as TF
                from torchvision.transforms.functional import InterpolationMode
                t = TF.normalize(TF.resize(t, (224, 224), InterpolationMode.BILINEAR),
                                 (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                mm = TF.resize(torch.from_numpy(it["bin640"].astype(np.float32))[None],
                               (224, 224), InterpolationMode.NEAREST)
                inp = torch.cat([t, (mm > 0.5).float()], 0)[None]
                with torch.no_grad():
                    vp = float(vb(inp.to(0)).sigmoid())
                feat = {"s1_conf": float(sc1), "s2_conf": it["s2_conf"], "verifier_prob": vp,
                        "s1_area_ratio": float(msk.sum()) / (H * W),
                        "s2_area_ratio": float(m6.sum()) / (H * W),
                        "s1s2_iou": gt_iou(msk, (P > 0).astype(np.uint8)),
                        "bbox_area_ratio": (bx[2] - bx[0]) * (bx[3] - bx[1]) / (H * W),
                        "bbox_aspect": (bx[2] - bx[0]) / max(bx[3] - bx[1], 1)}
                fused = float(1 / (1 + np.exp(-(np.dot(
                    (np.array([feat[c] for c in FCOLS]) - mu) / sg, coef) + b0))))
                cands.append({"ci": ci, "s1_conf": feat["s1_conf"],
                              "s2_conf": feat["s2_conf"], "verifier": vp,
                              "fused": round(fused, 3), "area_ratio": feat["s2_area_ratio"],
                              "mask": P})
        rep = {"W": W, "H": H, "n_s1": len(boxes), "n_cands": len(cands), "gts": []}
        for g in gts:
            # best candidate by final pred IoU
            best = None
            for cd in cands:
                io = gt_iou(cd["mask"], g["mask"])
                if best is None or io > best["iou"]:
                    best = {"iou": round(io, 3), "s1_conf": round(cd["s1_conf"], 3),
                            "s2_conf": round(cd["s2_conf"], 3),
                            "verifier": round(cd["verifier"], 3), "fused": cd["fused"]}
            s1_cover = 0.0
            for b in boxes:
                x0, y0, x1, y1 = [int(round(v)) for v in b]
                inter = int(g["mask"][max(0, y0):y1, max(0, x0):x1].sum())
                s1_cover = max(s1_cover, inter / max(g["px"], 1))
            sel = None
            if best and best["iou"] >= 0.5:
                thr = THR_LARGE if g["bucket"] == "large" else 0.6
                sel = best["fused"] >= thr
            rep["gts"].append({"bucket": g["bucket"], "px": g["px"],
                               "s1_cover": round(float(s1_cover), 3),
                               "best_s2": best, "selected": sel})
        report[iid] = rep
        print(iid, json.dumps(rep), flush=True)
    Path("outputs/calculus_v5_cls/large_fn_trace.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
