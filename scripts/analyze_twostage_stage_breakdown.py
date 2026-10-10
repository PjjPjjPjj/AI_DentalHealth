"""Stage-level breakdown of missed GT instances on test (comp-level IoU).

Two IoU views are computed for every GT instance:
  - det_iou  : max IoU over single predicted connected components (same
               matching criterion as compute_twostage_map / mAP50)
  - union_iou: IoU against the UNION of all stage-2 outputs on the image
               (what the pipeline emits as the final mask)

Stage-1 coverage: GT instance bbox vs candidate bboxes
  - s1_overlap  : bbox overlap > 0 with any candidate
  - s1_center_in: GT bbox center inside some candidate

Buckets by gt_ratio (instance px / image px), boundaries from
lesion_size_analysis.csv. Cache of inference stored in stage_cache.pkl.

Outputs:
  outputs/calculus_yolo11m_twostage/stage_breakdown.csv
  outputs/calculus_yolo11m_twostage/stage_breakdown_summary.txt
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

import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.inference.twostage import TwoStageSegmenter
from src.utils.paths import PROCESSED_DIR

DATA = PROCESSED_DIR / "calculus_seg"
RUN = Path("outputs/calculus_yolo11m_twostage")
CACHE = RUN / "stage_cache.pkl"
BUCKETS = ("small", "medium", "large")


def size_bucket(ratio: float) -> str:
    if ratio < 0.00552:
        return "small"
    if ratio < 0.01565:
        return "medium"
    return "large"


def run_inference() -> dict:
    thr = json.loads((RUN / "thresholds.json").read_text())
    c1, c2 = float(thr["stage1_conf"]), float(thr["selected_conf"])
    seg = TwoStageSegmenter("outputs/calculus_yolo11m_full/weights/best.pt",
                            "outputs/calculus_yolo11m_crop/weights/best.pt",
                            c1=c1, c2=c2, device=0)
    man = pd.read_csv(DATA / "split_manifest.csv")
    rows = man[(man.split == "test") & man.has_calculus]
    out = {}
    for i, (_, r) in enumerate(rows.iterrows()):
        img = cv2.imread(r["image_path"])
        H, W = img.shape[:2]
        gt = cv2.imread(r["mask_original_path"], cv2.IMREAD_UNCHANGED) > 0
        boxes = seg._stage1(r["image_path"])
        cand = np.zeros((0, 4)) if boxes is None else boxes

        dets, union, best_conf = [], np.zeros((H, W), np.uint8), 0.0
        for b in cand:
            x0, y0, x1, y1 = seg._window(b, H, W)
            ch, cw = y1 - y0, x1 - x0
            crop = cv2.resize(img[y0:y1, x0:x1], (640, 640),
                              interpolation=cv2.INTER_AREA)
            st_mask, sc = seg._stage2(crop)
            if st_mask is None:
                continue
            best_conf = max(best_conf, sc)
            m = cv2.resize(st_mask, (cw, ch), interpolation=cv2.INTER_NEAREST)
            buf = np.zeros((H, W), np.uint8)
            buf[y0:y1, x0:x1] = m
            union = np.maximum(union, buf)
            n, comp, _, _ = cv2.connectedComponentsWithStats(buf, 8)
            for k in range(1, n):
                if (comp == k).sum() < 5:
                    continue
                dets.append((comp == k).astype(np.uint8))
        out[r["image_id"]] = {
            "H": H, "W": W, "cand": cand.copy(), "dets": dets,
            "union": union > 0, "best_conf": best_conf,
            "gt": (gt, [m for m in (None,)]),
        }
        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1} images", flush=True)
    return out


def main() -> None:
    if CACHE.exists():
        print("loading cache", flush=True)
        cache = pickle.loads(CACHE.read_bytes())
    else:
        cache = run_inference()
        CACHE.write_bytes(pickle.dumps(cache))

    rows_l = []
    for image_id, c in cache.items():
        H, W = c["H"], c["W"]
        gt = c["gt"][0]
        cand = c["cand"]
        if not gt.any():
            continue
        n, comp, _, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8), 8)
        unionb = c["union"]
        for i in range(1, n):
            if (comp == i).sum() < 5:
                continue
            gm = (comp == i).astype(np.uint8)
            gx, gy, gw, gh = cv2.boundingRect(gm)
            px1, py1 = gx + gw, gy + gh
            gpx = int(gm.sum())
            ratio = gpx / float(H * W)
            s1_overlap, s1_center_in, s1_max_iou = False, False, 0.0
            if len(cand):
                cxs, cys = float(gx + gw / 2), float(gy + gh / 2)
                best = 0.0
                for bx0, by0, bx1, by1 in cand:
                    ix0, iy0 = max(gx, int(bx0)), max(gy, int(by0))
                    ix1, iy1 = min(px1, int(bx1)), min(py1, int(by1))
                    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
                    if inter <= 0:
                        continue
                    s1_overlap = True
                    iou = inter / float(gw * gh + float((bx1 - bx0) * (by1 - by0)) - inter)
                    best = max(best, iou)
                    if bx0 <= cxs <= bx1 and by0 <= cys <= by1:
                        s1_center_in = True
                s1_max_iou = best
            det_iou, det_hit50 = 0.0, False
            for dm in c["dets"]:
                inter = int((gm & dm).sum())
                uni = int((gm | dm).sum())
                iou = inter / uni if uni else 0.0
                if iou > det_iou:
                    det_iou = iou
            det_hit50 = det_iou >= 0.5
            inter_u = int((gm & unionb).sum())
            uni_u = int((gm | unionb).sum())
            iou_u = inter_u / uni_u if uni_u else 0.0
            rows_l.append({
                "image_id": image_id, "inst_id": i, "gt_px": gpx,
                "gt_ratio": round(ratio, 6), "bucket": size_bucket(ratio),
                "s1_max_bbox_iou": round(s1_max_iou, 4), "s1_overlap": s1_overlap,
                "s1_center_in": s1_center_in,
                "det_iou": round(det_iou, 4), "det_hit50": det_hit50,
                "union_iou": round(iou_u, 4), "union_hit50": iou_u >= 0.5,
                "max_conf": round(c["best_conf"], 4),
            })
    df = pd.DataFrame(rows_l)
    df.to_csv(RUN / "stage_breakdown.csv", index=False)

    lines = [f"GT instances: {len(df)} (comp-level)  images: {len(cache)}", ""]
    lines.append("bucket      n   s1_overlap  s1_center  det_iou>=.5  det|s1cov  union_iou>=.5  mean_det_iou")
    for b in BUCKETS:
        sub = df[df.bucket == b]
        n = len(sub)
        s1ov = float(sub.s1_overlap.mean()) if n else 0.0
        s1c = float(sub.s1_center_in.mean()) if n else 0.0
        hit = float(sub.det_hit50.mean()) if n else 0.0
        cov = sub[sub.s1_overlap]
        cond = float(cov.det_hit50.mean()) if len(cov) else 0.0
        uh = float(sub.union_hit50.mean()) if n else 0.0
        mi = float(sub.det_iou.mean()) if n else 0.0
        lines.append(f"{b:8s} {n:5d}  {s1ov:10.3f}  {s1c:9.3f}  {hit:10.3f}  "
                     f"{cond:10.3f}  {uh:14.3f}  {mi:14.3f}")
    lines.insert(0, f"Images (test positives): {len(cache)}")
    report = "\n".join(lines)
    print(report)
    (RUN / "stage_breakdown_summary.txt").write_text(report)

    miss = df[~(df.det_hit50 | df.union_hit50)]
    print("\n-- Missed instances in BOTH views, top 12 by det_iou --")
    print(miss.sort_values("det_iou", ascending=False)
          [["image_id", "bucket", "gt_px", "s1_overlap", "det_iou"]]
          .head(12).to_string())
    print("\n-- s1 coverage vs det success by bucket (miss attribution) --")
    for b in BUCKETS:
        sub = df[df.bucket == b]
        no_s1 = sub[~sub.s1_overlap]
        s1_no_det = sub[sub.s1_overlap & ~sub.det_hit50]
        ok = sub[sub.det_hit50]
        print(f"{b:8s}: n={len(sub):3d}  s1-miss={len(no_s1):3d} "
              f"({len(no_s1)/max(len(sub),1)*100:4.1f}%)  s1-captured-but-det-iou<.5={len(s1_no_det):3d} "
              f"({len(s1_no_det)/max(len(sub),1)*100:4.1f}%)  detected={len(ok):3d}")


if __name__ == "__main__":
    main()
