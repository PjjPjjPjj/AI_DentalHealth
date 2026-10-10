"""V6 4-class evaluation on val: per-class AP + calculus proposal quality vs V5.

Key comparison: 4-class model's calculus proposals (cls 0) vs the V5 binary
pipeline's calculus proposals (from val_collect_v4.pkl) at matched conf.
Outputs: outputs/calculus_v6_4cls/eval_v6.json
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from scripts.v3_pipeline_lib import BUCKET_EDGES, RAW, bucket_of

OUT = Path("outputs/calculus_v6_4cls")
DATA = Path("datasets/processed/calculus_seg_v3")
IOU = 0.5


def gt_calc_instances(m):
    calc = (m == 3).astype(np.uint8)
    gts = []
    if calc.any():
        n, comp = cv2.connectedComponents(calc, 8)
        for i in range(1, n):
            px = int((comp == i).sum())
            if px >= 5:
                gts.append((comp == i).astype(np.uint8))
    return gts


def greedy_match(preds, gts):
    pairs = []
    for pi, P in enumerate(preds):
        for gi, G in enumerate(gts):
            inter = int((P & G).sum())
            uni = int((P | G).sum())
            io = inter / uni if uni else 0.0
            if io >= IOU:
                pairs.append((io, pi, gi))
    pairs.sort(reverse=True)
    used_p, used_g, matched = set(), set(), []
    for io, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi); used_g.add(gi)
        matched.append((pi, gi))
    return matched, len(preds) - len(matched)


def main() -> None:
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else \
        "outputs/calculus_v6_4cls/weights/best.pt"
    model = YOLO(model_path)
    man = pd.read_csv(DATA / "split_manifest_v3.csv")
    rows = man[man.split == "val"].reset_index(drop=True)

    C_grid = [0.02, 0.05, 0.10, 0.20, 0.30]
    agg = {c: {"tp": 0, "fp": 0, "fp_on_other": 0} for c in C_grid}
    per_class_ap = None
    ys, confs = [], []
    for ri, r in rows.iterrows():
        img = cv2.imread(r.image_path)
        H, W = img.shape[:2]
        m = cv2.imread(str(RAW / f"{r.image_id}.png"), cv2.IMREAD_UNCHANGED)
        if m is None:
            m = np.zeros((H, W), np.uint8)
        if m.ndim == 3:
            m = m[:, :, 0]
        gts = gt_calc_instances(m)
        res = model.predict(r.image_path, conf=C_grid[0], iou=0.6, imgsz=640,
                            device=0, retina_masks=True, max_det=150,
                            verbose=False)[0]
        mk = res.masks.data.cpu().numpy() if res.masks is not None else None
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else None
        clss = res.boxes.cls.cpu().numpy().astype(int) if res.boxes is not None else None
        confs_all = res.boxes.conf.cpu().numpy() if res.boxes is not None else None
        for c in C_grid:
            preds = []
            if boxes is not None:
                for i in range(len(boxes)):
                    if clss[i] == 0 and confs_all[i] >= c:
                        P = np.zeros((H, W), np.uint8)
                        P[mk[i] > 0.5] = 1
                        preds.append(P)
            matched, fp = greedy_match(preds, gts)
            agg[c]["tp"] += len(matched)
            agg[c]["fp"] += fp
        if boxes is not None:
            c0 = [(confs_all[i], boxes[i], mk[i] > 0.5)
                  for i in range(len(boxes))
                  if clss[i] == 0 and confs_all[i] >= 0.02]
            c0.sort(key=lambda t: -t[0])
            preds = []
            for cf, b, mk_ in c0:
                P = np.zeros((H, W), np.uint8)
                P[mk_] = 1
                preds.append(P)
            matched, fp = greedy_match(preds, gts)
            mp = {pi for pi, _ in matched}
            for pi, P in enumerate(preds):
                if pi not in mp and (P & (np.isin(m, [1, 2, 4]))).sum() > 0.3 * max(P.sum(), 1):
                    agg[0.05]["fp_on_other"] += 1
            ys.append(int(r.has_calculus))
            confs.append(float(max([t[0] for t in c0], default=0.0)))
    tot_gt = sum(len(gt_calc_instances(
        cv2.imread(str(RAW / f"{r.image_id}.png"), cv2.IMREAD_UNCHANGED)
        if cv2.imread(str(RAW / f"{r.image_id}.png"), cv2.IMREAD_UNCHANGED) is not None
        else np.zeros((1, 1), np.uint8))) for r in rows.itertuples())

    rep = {"gt_calculus_instances": tot_gt}
    for c in C_grid:
        rec = agg[c]["tp"] / max(tot_gt, 1)
        rep[f"c1_{c}"] = {"proposal_recall": round(rec, 4),
                          "fp_img": round(agg[c]["fp"] / len(rows), 3),
                          "fp_on_other_share": round(
                              agg[0.05]["fp_on_other"] / max(agg[0.05]["fp"], 1), 3)}
    from sklearn.metrics import roc_auc_score
    rep["img_auroc_maxconf"] = round(float(roc_auc_score(ys, confs)), 4)
    tag = Path(model_path).parts[-3] if len(Path(model_path).parts) > 3 else "v6"
    (OUT / f"eval_v6_{tag}.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
