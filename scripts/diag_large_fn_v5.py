"""Diagnose large-lesion FNs of the v5 cascade on the calculus test split.

For each FN id (from outputs/calculus_v5_cls/large_lesion_diag_v5.json):
  image size, GT bucket/px/ratio, stage1 candidates under current s1_pass,
  and stage1 candidates with an extra zoom-out (whole-image) pass.
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
from ultralytics import YOLO

import scripts.v3_pipeline_lib as _lib
_lib.C1 = 0.005
from scripts.v3_pipeline_lib import RAW, image_gts, s1_pass

IDS = ["216", "74", "caries_0_115", "caries_0_1652", "caries_0_2747",
       "caries_0_3129", "caries_0_4307", "caries_0_5873", "caries_0_5959",
       "caries_0_8180", "caries_0_835", "caries_0_7630", "caries_0_7816"]


def image_gts_bin(mask_path: str, H: int, W: int, min_px: int = 5):
    m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if m is None:
        m = np.zeros((H, W), np.uint8)
    if m.ndim == 3:
        m = m[:, :, 0]
    calc = (m > 0).astype(np.uint8)
    gts = []
    if calc.any():
        n, comp = cv2.connectedComponents(calc, 8)
        for i in range(1, n):
            px = int((comp == i).sum())
            if px < min_px:
                continue
            gts.append({"mask": (comp == i).astype(np.uint8), "px": px,
                        "bucket": _lib.bucket_of(px / float(H * W))})
    return gts


def box_gt_overlap(b, g):
    x0, y0, x1, y1 = [int(round(v)) for v in b]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(g.shape[1], x1), min(g.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = int(g[y0:y1, x0:x1].sum())
    gt = int(g.sum())
    return inter / max(gt, 1)   # fraction of GT covered by box


def main() -> None:
    s1m = YOLO("outputs/calculus_v5_stage1/weights/best.pt")
    man = pd.read_csv(_lib.DATA / "split_manifest_v3.csv")
    rows = man[man.split == "test"].set_index("image_id")
    out = []
    for iid in IDS:
        r = rows.loc[iid]
        img = cv2.imread(r["image_path"])
        H, W = img.shape[:2]
        gts = image_gts_bin(r["mask_original_path"], H, W)
        big = max(gts, key=lambda g: g["px"]) if gts else None
        b_std, c_std, m_std = s1_pass(s1m, r.image_path)
        # zoom-out pass: whole image directly (ultralytics resizes max side to 640)
        res = s1m.predict(r.image_path, conf=0.005, iou=0.6, imgsz=640, device=0,
                          retina_masks=True, max_det=150, verbose=False)[0]
        cls = res.boxes.cls.cpu().numpy().astype(int)
        keep = np.where(cls == 0)[0]
        b_zoom = res.boxes.xyxy.cpu().numpy()[keep] if len(keep) else np.zeros((0, 4), np.float32)
        c_zoom = res.boxes.conf.cpu().numpy()[keep] if len(keep) else np.zeros(0, np.float32)
        rec = {"id": iid, "W": W, "H": H, "wide": bool(max(W, H) >= _lib.WIDE_MIN),
               "gt_px": int(big["px"]) if big else 0,
               "gt_ratio": round(float(big["px"]) / (H * W), 5) if big else 0}
        for tag, bb, cc in [("std", b_std, c_std), ("zoom", b_zoom, c_zoom)]:
            if big is None or len(bb) == 0:
                rec[f"{tag}_cover"] = 0.0
                rec[f"{tag}_best_conf"] = 0.0
                continue
            covers = [box_gt_overlap(b, big["mask"]) for b in bb]
            j = int(np.argmax(covers))
            rec[f"{tag}_cover"] = round(float(covers[j]), 3)
            rec[f"{tag}_best_conf"] = round(float(cc[j]), 3)
        out.append(rec)
        print(json.dumps(rec), flush=True)
    Path("outputs/calculus_v5_cls/large_fn_probe.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
