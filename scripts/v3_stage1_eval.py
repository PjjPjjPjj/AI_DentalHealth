"""V3 Stage1 evaluation on val: proposal recall by bucket, c1 sweep, FP anatomy.

Reports:
  - instance recall@0.5 of calculus proposals (1-1 greedy) per size bucket
  - image-level AUROC at max calculus conf
  - FP anatomy: share of false proposals overlapping other-lesion GT
Outputs: outputs/calculus_v3_stage1/eval_val.json
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
from sklearn.metrics import roc_auc_score

from ultralytics import YOLO

from scripts.v3_pipeline_lib import BUCKETS, DATA, bucket_of

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

OUT = Path("outputs/calculus_v3_stage1")
IOU_MATCH = 0.5


def greedy_match(preds, gts):
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
    return matched


def main() -> None:
    model = YOLO("outputs/calculus_v3_stage1/weights/best.pt")
    man = pd.read_csv(DATA / "split_manifest_v3.csv")
    rows = man[man.split == "val"].reset_index(drop=True)
    raw_dir = Path(f"{_ROOT}/datasets/raw/oral_mamba/lables")

    c1_list = [0.02, 0.05, 0.10, 0.20]
    agg = {c: {"tp": 0, "fp": 0, "proposals": 0, "fp_on_other": 0}
           for c in c1_list}
    gtb = {b: 0 for b in BUCKETS}
    tp_b = {c: {b: 0 for b in BUCKETS} for c in c1_list}
    img_conf = {c: [] for c in c1_list}
    ys = []
    per_c1_fp_imgs = {c: 0 for c in c1_list}

    for ri, r in rows.iterrows():
        img = cv2.imread(r.image_path)
        H, W = img.shape[:2]
        m = cv2.imread(str(raw_dir / f"{r.image_id}.png"), cv2.IMREAD_UNCHANGED)
        if m is None:
            m = np.zeros((H, W), np.uint8)
        if m.ndim == 3:
            m = m[:, :, 0]
        calc = (m == 3)
        other = np.isin(m, [1, 2, 4]).astype(np.uint8)
        gts = []
        if calc.any():
            n, comp = cv2.connectedComponents(calc.astype(np.uint8), 8)
            for i in range(1, n):
                px = int((comp == i).sum())
                if px < 5:
                    continue
                gts.append({"mask": (comp == i).astype(np.uint8), "px": px,
                            "bucket": bucket_of(px / float(H * W))})
        for G in gts:
            gtb[G["bucket"]] += 1
        ys.append(int(r.has_calculus))

        res = model.predict(r.image_path, conf=c1_list[0], iou=0.6, imgsz=640,
                            device=0, retina_masks=True, max_det=100,
                            verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            for c in c1_list:
                img_conf[c].append(0.0)
            continue
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        mk = res.masks.data.cpu().numpy()
        pmasks = [(mk[i] > 0.5) for i in range(mk.shape[0])]
        calc_idx = [i for i in range(len(boxes)) if clss[i] == 0]
        for c in c1_list:
            preds = [pmasks[i] for i in calc_idx if confs[i] >= c]
            confs_c = [confs[i] for i in calc_idx if confs[i] >= c]
            img_conf[c].append(float(max(confs_c) if confs_c else 0.0))
            matched = greedy_match(preds, gts)
            mg = {gi for _, gi, _ in matched}
            agg[c]["tp"] += len(matched)
            agg[c]["proposals"] += len(preds)
            fp = len(preds) - len(matched)
            agg[c]["fp"] += fp
            if fp:
                per_c1_fp_imgs[c] += 1
            for b in BUCKETS:
                tp_b[c][b] += sum(1 for _, gi, _ in matched
                                  if gts[gi]["bucket"] == b)
            # FP anatomy on the strictest c1 only (report once at 0.05)
            if abs(c - 0.05) < 1e-6:
                mp = {pi for _, pi, _ in matched}
                for pi, P in enumerate(preds):
                    if pi in mp:
                        continue
                    if (P & (other > 0)).sum() > 0.3 * P.sum():
                        agg[c]["fp_on_other"] += 1

    rep = {"n_val": len(rows)}
    for c in c1_list:
        pr = tp_b[c]
        rep[f"c1_{c}"] = {
            "proposals_img": round(agg[c]["proposals"] / len(rows), 3),
            "fp_img": round(agg[c]["fp"] / len(rows), 3),
            "instance_recall": round(agg[c]["tp"] / max(sum(gtb.values()), 1), 4),
            "recall_small": round(pr["small"] / max(gtb["small"], 1), 4),
            "recall_medium": round(pr["medium"] / max(gtb["medium"], 1), 4),
            "recall_large": round(pr["large"] / max(gtb["large"], 1), 4),
            "img_fp_rate": round(per_c1_fp_imgs[c] / len(rows), 3),
            "fp_on_other_share": round(agg[c]["fp_on_other"] / max(agg[c]["fp"], 1), 3),
            "img_auroc": round(float(roc_auc_score(ys, img_conf[c])), 4),
        }
    rep["gt_buckets"] = gtb
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "eval_val.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
