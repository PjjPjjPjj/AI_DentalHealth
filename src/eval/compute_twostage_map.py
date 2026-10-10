"""Instance-level mAP50 / mAP50-95 for the two-stage pipeline on test.

Instances are connected components of the final predicted mask; confidence =
max stage-2 conf across candidates. GT instances come from the original
binary calculus mask. AP is computed per IoU threshold 0.50..0.95 step .05.
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

from src.utils.logger import get_logger
from src.utils.paths import PROCESSED_DIR

logger = get_logger("twostage_map")

DATA = PROCESSED_DIR / "calculus_seg"
RUN = Path("outputs/calculus_yolo11m_twostage")


def main() -> None:
    thr = json.loads((RUN / "thresholds.json").read_text())
    c1, c2 = float(thr["stage1_conf"]), float(thr["selected_conf"])
    from src.inference.twostage import TwoStageSegmenter
    seg = TwoStageSegmenter("outputs/calculus_yolo11m_full/weights/best.pt",
                            "outputs/calculus_yolo11m_crop/weights/best.pt",
                            c1=c1, c2=c2, device=0)
    man = pd.read_csv(DATA / "split_manifest.csv")
    rows = man[man.split == "test"]

    dets = []  # (image_id, conf, mask_hw) predicted instances
    gt_inst = []  # (image_id, mask_hw)
    for _, r in rows.iterrows():
        img = cv2.imread(r["image_path"])
        H, W = img.shape[:2]
        gt = cv2.imread(r["mask_original_path"], cv2.IMREAD_UNCHANGED) > 0
        img2 = img
        boxes = seg._stage1(r["image_path"])
        if boxes is not None:
            for b in boxes:
                x0, y0, x1, y1 = seg._window(b, H, W)
                crop = cv2.resize(img2[y0:y1, x0:x1], (640, 640), interpolation=cv2.INTER_AREA)
                st_mask, sc = seg._stage2(crop)
                if st_mask is None:
                    continue
                m = cv2.resize(st_mask, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
                buf = np.zeros((H, W), np.uint8)
                buf[y0:y1, x0:x1] = m
                n, comp, _, _ = cv2.connectedComponentsWithStats(buf, 8)
                for i in range(1, n):
                    if (comp == i).sum() < 5:
                        continue
                    dets.append({"image_id": r["image_id"], "conf": sc,
                                 "mask": (comp == i).astype(np.uint8)})
        if gt.any():
            n, comp, _, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8), 8)
            for i in range(1, n):
                if (comp == i).sum() < 5:
                    continue
                gt_inst.append({"image_id": r["image_id"], "mask": (comp == i).astype(np.uint8)})

    ious = []
    for d in dets:
        imghash = d["image_id"]
        best = 0.0
        for g in gt_inst:
            if g["image_id"] != imghash:
                continue
            inter = int((d["mask"] & g["mask"]).sum())
            union = int((d["mask"] | g["mask"]).sum())
            best = max(best, inter / union if union else 0.0)
        ious.append(best)

    n_gt = len(gt_inst)
    n_pred = len(dets)
    logger.info(f"GT instances={n_gt} pred instances={n_pred}")

    order = np.argsort(-np.array([d["conf"] for d in dets]))
    results = {}
    for t in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        tp_count = 0.0
        used_gt = set()
        precs = []
        for idx in order:
            ov = ious[idx]
            ok = ov >= t
            # greedy one-to-one matching via mask overlap with a distinct GT
            matched = None
            if ok:
                imghash = dets[idx]["image_id"]
                for gi, g in enumerate(gt_inst):
                    if gi in used_gt or g["image_id"] != imghash:
                        continue
                    inter = int((dets[idx]["mask"] & g["mask"]).sum())
                    union = int((dets[idx]["mask"] | g["mask"]).sum())
                    iou = inter / union if union else 0.0
                    if iou >= t:
                        matched = gi
                        break
            if matched is not None:
                used_gt.add(matched)
                tp_count += 1
            precs.append(tp_count / (len(precs) + 1))
        recs_cum = tp_count / n_gt if n_gt else 0.0
        results[t] = {"AP": float(np.trapz(precs, np.linspace(0, 1, len(precs)))),
                      "recall@1": float(recs_cum)}
    map50 = results[0.5]["AP"]
    map5095 = float(np.mean([results[t]["AP"] for t in results]))
    (RUN / "final_metrics.json").write_text(json.dumps({
        "mask_mAP50": round(map50, 4), "mask_mAP50_95": round(map5095, 4),
        "instance_recall@IoU0.5": round(results[0.5]["recall@1"], 4),
        "gt_instances": n_gt, "pred_instances": n_pred,
    }, indent=2, default=float))
    logger.info(f"mAP50(M)={map50:.4f} mAP50-95(M)={map5095:.4f} rec@0.5={results[0.5]['recall@1']:.4f}")


if __name__ == "__main__":
    main()
