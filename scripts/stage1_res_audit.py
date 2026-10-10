"""PHASE C1: Stage1 inference-resolution diagnostic (development=val split).

Frozen Stage1 checkpoint, c1=0.05, NMS/IoU same as baseline; only Stage1
inference imgsz varies 640/768/896/1024. Stage2/Verifier/Fusion NOT used.
Per imgsz, per size bucket: GT n, proposal Recall (bbox-overlap>0),
proposal Recall (mask-IoU>=0.1), no-proposal rate, confidence
median/P25/P75, proposals/image, proposal precision (>=IoU0.1 with any GT),
latency mean/median/P95, GPU peak MB.
Small GT additionally stratified by bbox short-side: <10 / 10-15 / 15-20 /
>=20 px -> proposal Recall.
Outputs: outputs/calculus_cls/stage1_resolution_audit.csv
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

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from src.utils.paths import PROCESSED_DIR

DATA = PROCESSED_DIR / "calculus_seg"
OUT = Path("outputs/calculus_cls")
C1 = 0.05
BUCKET_EDGES = (0.00552, 0.01565)
BUCKETS = ("small", "medium", "large")
RES = (640, 768, 896, 1024)


def bucket_of(r: float) -> str:
    if r < BUCKET_EDGES[0]:
        return "small"
    if r < BUCKET_EDGES[1]:
        return "medium"
    return "large"


def main() -> None:
    model = YOLO("outputs/calculus_yolo11m_full/weights/best.pt")
    man = pd.read_csv(DATA / "split_manifest.csv")
    rows = man[man.split == "val"]
    rows_l = []
    for res in RES:
        gts_by_bucket = {b: [] for b in BUCKETS}
        prop_cnt = 0
        prop_hit = 0
        time_l = []
        n_prop_img = []
        torch.cuda.reset_peak_memory_stats()
        for _, r in rows.iterrows():
            img = cv2.imread(r["image_path"])
            H, W = img.shape[:2]
            gtb = cv2.imread(r["mask_original_path"], cv2.IMREAD_UNCHANGED)
            gtb = (gtb > 0) if gtb is not None else np.zeros((H, W), np.uint8)
            n, comp, _, _ = cv2.connectedComponentsWithStats(
                gtb.astype(np.uint8), 8)
            gts = []
            for i in range(1, n):
                if (comp == i).sum() < 5:
                    continue
                m = (comp == i).astype(np.uint8)
                gx, gy, gw, gh = cv2.boundingRect(m)
                px = int(m.sum())
                ratio = px / float(H * W)
                gts.append({"mask": m, "px": px, "box": (gx, gy, gw, gh),
                            "bucket": bucket_of(ratio),
                            "short": min(gw, gh)})
            t0 = time.time()
            rp = model.predict(r["image_path"], conf=C1, iou=0.6, imgsz=res,
                               device=0, retina_masks=True, max_det=100,
                               verbose=False)[0]
            time_l.append((time.time() - t0) * 1000)
            boxes = (rp.boxes.xyxy.cpu().numpy()
                     if rp.boxes is not None and len(rp.boxes)
                     else np.zeros((0, 4)))
            confs = (rp.boxes.conf.cpu().numpy()
                     if rp.boxes is not None and len(rp.boxes)
                     else np.zeros((0,)))
            mk = (rp.masks.data.cpu().numpy()
                  if rp.masks is not None else None)
            n_prop_img.append(len(boxes))
            prop_cnt += len(boxes)
            for g in gts:
                gx, gy, gw, gh = g["box"]
                hit_bb = False
                mask_iou = 0.0
                best_conf = 0.0
                for bi, bs in enumerate(boxes):
                    inter = (max(0, min(gx + gw, int(bs[2])) - max(gx, int(bs[0]))) *
                             max(0, min(gy + gh, int(bs[3])) - max(gy, int(bs[1]))))
                    if inter > 0:
                        hit_bb = True
                        best_conf = max(best_conf, float(confs[bi]))
                    if mk is not None:
                        msk = (mk[bi] > 0.5).astype(np.uint8)
                        i2 = int((msk & g["mask"]).sum())
                        u = int((msk | g["mask"]).sum())
                        mask_iou = max(mask_iou, i2 / u if u else 0.0)
                gts_by_bucket[g["bucket"]].append({
                    "hit_bb": int(hit_bb), "mid_hit": int(mask_iou >= 0.1),
                    "conf": best_conf, "short": g["short"], "px": g["px"]})
                if mk is not None:
                    for bi in range(len(boxes)):
                        msk = (mk[bi] > 0.5).astype(np.uint8)
                        for g2 in gts:
                            i2 = int((msk & g2["mask"]).sum())
                            u = int((msk | g2["mask"]).sum())
                            if u and i2 / u >= 0.1:
                                prop_hit += 1
                                break
        lat = np.array(time_l)
        mem = torch.cuda.max_memory_allocated() / 1e6
        for b in BUCKETS:
            v = gts_by_bucket[b]
            if not v:
                continue
            hitc = [x["conf"] for x in v if x["hit_bb"] == 1]
            rows_l.append({
                "imgsz": res, "bucket": b, "n_gt": len(v),
                "recall_bbox": round(float(np.mean([x["hit_bb"] for x in v])), 4),
                "recall_mIoU01": round(float(np.mean([x["mid_hit"] for x in v])), 4),
                "no_proposal": round(float(1 - np.mean([x["hit_bb"] for x in v])), 4),
                "conf_med": round(float(np.median(hitc)), 4) if hitc else None,
                "conf_p25": round(float(np.percentile(hitc, 25)), 4) if hitc else None,
                "conf_p75": round(float(np.percentile(hitc, 75)), 4) if hitc else None,
                "proposals_img": round(float(np.mean(n_prop_img)), 3),
                "proposal_precision": round(prop_hit / max(prop_cnt, 1), 4),
                "lat_mean": round(float(np.mean(lat)), 1),
                "lat_med": round(float(np.median(lat)), 1),
                "lat_p95": round(float(np.percentile(lat, 95)), 1),
                "gpu_peak_mb": round(mem, 0)})
        # small stratification
        small = gts_by_bucket["small"]
        for lo, hi, tag in ((0, 10, "<10"), (10, 15, "10-15"),
                            (15, 20, "15-20"), (20, 100000, ">=20")):
            sel = [x for x in small if lo <= x["short"] < hi]
            if sel:
                rows_l.append({"imgsz": res, "bucket": f"small:{tag}",
                               "n_gt": len(sel),
                               "recall_bbox": round(float(np.mean(
                                   [x["hit_bb"] for x in sel])), 4),
                               "recall_mIoU01": round(float(np.mean(
                                   [x["mid_hit"] for x in sel])), 4),
                               "no_proposal": round(float(1 - np.mean(
                                   [x["hit_bb"] for x in sel])), 4),
                               "conf_med": None, "conf_p25": None,
                               "conf_p75": None,
                               "proposals_img": None,
                               "proposal_precision": None,
                               "lat_mean": None, "lat_med": None,
                               "lat_p95": None, "gpu_peak_mb": None})
    df = pd.DataFrame(rows_l)
    df.to_csv(OUT / "stage1_resolution_audit.csv", index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
