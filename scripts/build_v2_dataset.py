"""Stage2 V2 training-data build (STEP 1-4): TRAIN split only, no val/test.

STEP1: real Stage1 (c1=0.05 frozen) on train split -> predicted bbox ->
       bbox-centered 160x160 crop (clamped, same as inference; NO 1.8 margin).
       Match crop labels against GT instances (per instance, >=5 px):
         positive : at least one GT instance with inclusion>=0.8
         negative : zero calculus GT pixels in the crop
         ambiguous: some GT but max inclusion<0.8  (excluded from training)
       Multi-instance crops keep ALL independent instance labels (no union).
STEP2: run current Stage2 V1 (V1 model, base conf 0.05) on every proposal
       crop -> classify positives:
         normal_positive : exists GT with best-instance IoU>=0.5 & conf>=0.40
         hard_positive   : otherwise
       negatives:
         hard_negative   : max V1 s2_conf >= 0.20
         easy_negative   : < 0.20
STEP3: balanced sampling plan (weighted, no physical copies; symlinks).
STEP4: leakage audit + note that Stage1 & Stage2 V1 were both trained on the
       same train split (proposal may be optimistic; TODO out-of-fold).

Outputs (crop 160x160 PNG + YOLO-seg txt labels, normalized to crop w/h):
  datasets/processed/calculus_v2/crops/{images,labels,crops_meta.csv}
  datasets/processed/calculus_v2/balance_plan.md
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

import csv
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.inference.twostage import TARGET, TwoStageSegmenter
from src.utils.paths import PROCESSED_DIR

DATA = PROCESSED_DIR / "calculus_seg"
OUT = PROCESSED_DIR / "calculus_v2"
CROPS = OUT / "crops"
C1 = 0.05
V1_C2_MINING = 0.05   # base conf for mining (record all responses)
C2_FROZEN = 0.40
SIDE = 160
AREA_MIN = 5
BUCKET_EDGES = (0.00552, 0.01565)  # kept for reporting consistency


def bucket_of(ratio: float) -> str:
    if ratio < BUCKET_EDGES[0]:
        return "small"
    if ratio < BUCKET_EDGES[1]:
        return "medium"
    return "large"


def instance_masks(mask: np.ndarray) -> list[np.ndarray]:
    n, comp, _, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        if (comp == i).sum() < AREA_MIN:
            continue
        out.append((comp == i).astype(np.uint8))
    return out


def mask_to_polygon(m: np.ndarray, cw: int, ch: int) -> list[float]:
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return []
    c = max(cs, key=cv2.contourArea)
    eps = 1.5
    c = cv2.approxPolyDP(c, eps, True)
    pts = c.reshape(-1, 2).astype(float)
    if len(pts) < 3:
        return []
    flat = []
    for x, y in pts:
        flat.append(round(float(x) / cw, 6) if cw else 0.0)
        flat.append(round(float(y) / ch, 6) if ch else 0.0)
    return flat


def main() -> None:
    seg = TwoStageSegmenter("outputs/calculus_yolo11m_full/weights/best.pt",
                            "outputs/calculus_yolo11m_crop/weights/best.pt",
                            c1=C1, c2=V1_C2_MINING, device=0, fusion="mask_nms",
                            nms_iou=0.5, score_mode="s2")
    if CROPS.exists():
        shutil.rmtree(CROPS)
    (CROPS / "images").mkdir(parents=True)
    (CROPS / "labels").mkdir(parents=True)

    man = pd.read_csv(DATA / "split_manifest.csv")
    rows = man[man.split == "train"]
    seen_ids = set()
    with open(CROPS / "crops_meta.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["crop_id", "type", "src_image", "n_gt_inst", "gt_buckets",
                    "max_s2_conf", "best_inst_iou", "ambiguous_no_gt"])
        idx = 0
        for ri, (_, r) in enumerate(rows.iterrows()):
            img = cv2.imread(r["image_path"])
            H, W = img.shape[:2]
            gt = cv2.imread(r["mask_original_path"], cv2.IMREAD_UNCHANGED)
            gtb = (gt > 0) if gt is not None else np.zeros((H, W), np.uint8)
            boxes, confs = seg._stage1_full(r["image_path"])
            for b, sc1 in zip(boxes, confs):
                cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
                x0 = int(round(cx - SIDE / 2)); y0 = int(round(cy - SIDE / 2))
                x0 = max(0, min(x0, W - SIDE)); y0 = max(0, min(y0, H - SIDE))
                x1, y1 = min(x0 + SIDE, W), min(y0 + SIDE, H)
                ch, cw = y1 - y0, x1 - x0
                crop_img = img[y0:y1, x0:x1]
                gc = gtb[y0:y1, x0:x1]
                gts_in = instance_masks(gc)
                has_gt = gts_in and True
                # inclusion per GT (relative to its FULL mask at image level)
                full_masks = instance_masks(gtb)
                # map crop instances to full masks by overlap
                incs = []
                for gm in gts_in:
                    best_frac = 0.0
                    for fm in full_masks:
                        inter = int((gm & fm[y0:y1, x0:x1]).sum())
                        if inter > 0:
                            frac = inter / float(fm.sum())
                            best_frac = max(best_frac, frac)
                    incs.append(best_frac)
                incs = np.array(incs) if incs else np.array([])
                if len(incs) == 0:
                    ctype = "negative"
                elif incs.max() >= 0.8:
                    ctype = "positive"
                else:
                    ctype = "ambiguous"
                # stage2 V1 mining
                resized = cv2.resize(crop_img, (TARGET, TARGET),
                                     interpolation=cv2.INTER_AREA)
                insts = seg._stage2_instances(resized)
                v1_insts = [{"bin": i["bin640"], "conf": i["s2_conf"]} for i in insts]
                max_conf = max([i["conf"] for i in v1_insts], default=0.0)
                best_iou_all = 0.0
                if v1_insts and ctype == "positive":
                    tc = cv2.resize(gc.astype(np.uint8), (TARGET, TARGET),
                                    interpolation=cv2.INTER_AREA) > 0.5
                    for vi in v1_insts:
                        for gm in gts_in:
                            tg = cv2.resize(gm.astype(np.uint8), (TARGET, TARGET),
                                            interpolation=cv2.INTER_AREA) > 0.5
                            ii = int((tg & vi["bin"]).sum())
                            uu = int((tg | vi["bin"]).sum())
                            best_iou_all = max(best_iou_all, ii / uu if uu else 0.0)
                if ctype == "positive":
                    hitted = best_iou_all >= 0.5 and max_conf >= C2_FROZEN
                    final_type = "normal_positive" if hitted else "hard_positive"
                elif ctype == "negative":
                    final_type = ("hard_negative" if max_conf >= 0.20
                                  else "easy_negative")
                else:
                    final_type = "ambiguous_excluded"
                if final_type != "ambiguous_excluded":
                    cid = f"tr{idx:06d}"
                    cv2.imwrite(str(CROPS / "images" / f"{cid}.png"), crop_img)
                    lab_lines = []
                    bucket_tags = []
                    for i, gm in enumerate(gts_in):
                        poly = mask_to_polygon(gm, cw, ch)
                        if poly:
                            lab_lines.append("0 " + " ".join(map(str, poly)))
                            best_fm, bi = None, 0
                            for fm in full_masks:
                                inter = int((gm & fm[y0:y1, x0:x1]).sum())
                                if inter > bi:
                                    bi, best_fm = inter, fm
                            ratio = (best_fm.sum() / float(H * W)
                                     if best_fm is not None else
                                     int(gm.sum()) / float(H * W))
                            bucket_tags.append(bucket_of(ratio))
                    (CROPS / "labels" / f"{cid}.txt").write_text(
                        "\n".join(lab_lines) + "\n")
                    w.writerow([cid, final_type, r["image_id"],
                                len(gts_in), "|".join(sorted(set(bucket_tags))),
                                round(max_conf, 4), round(best_iou_all, 4),
                                int(len(incs) > 0 and ctype == "ambiguous")])
                idx += 1
            if ri % 500 == 0 and ri:
                print(f"  processed {ri}/{len(rows)}", flush=True)
    print("proposal crops done", flush=True)

    # ---- GT-centered clean positives (train only, per GT instance) ----
    with open(CROPS / "crops_meta.csv", "a", newline="") as f:
        w = csv.writer(f)
        idx2 = 0
        for _, r in rows.iterrows():
            img = cv2.imread(r["image_path"])
            H, W = img.shape[:2]
            gt = cv2.imread(r["mask_original_path"], cv2.IMREAD_UNCHANGED)
            if gt is None or not (gt > 0).any():
                continue
            gtb = gt > 0
            for m in instance_masks(gtb):
                gx, gy, gw, gh = cv2.boundingRect(m)
                cx, cy = gx + gw / 2.0, gy + gh / 2.0
                x0 = int(round(cx - SIDE / 2)); y0 = int(round(cy - SIDE / 2))
                x0 = max(0, min(x0, W - SIDE)); y0 = max(0, min(y0, H - SIDE))
                x1, y1 = min(x0 + SIDE, W), min(y0 + SIDE, H)
                ch, cw = y1 - y0, x1 - x0
                cid = f"gt{idx2:06d}"
                idx2 += 1
                crop_img = img[y0:y1, x0:x1]
                cv2.imwrite(str(CROPS / "images" / f"{cid}.png"), crop_img)
                gc = gtb[y0:y1, x0:x1]
                lines = []
                tagger = []
                for gm in instance_masks(gc):
                    poly = mask_to_polygon(gm, cw, ch)
                    if poly:
                        lines.append("0 " + " ".join(map(str, poly)))
                        tagger.append(bucket_of(int(gm.sum()) / float(H * W)))
                (CROPS / "labels" / f"{cid}.txt").write_text(
                    "\n".join(lines) + "\n")
                w.writerow([cid, "gt_centered_positive", r["image_id"],
                            len(lines), "|".join(sorted(set(tagger))),
                            0.0, 0.0, 0])
    print("gt-centered done", flush=True)

    meta = pd.read_csv(CROPS / "crops_meta.csv")
    print("\n== STEP2 stats (per type) ==")
    print(meta.groupby("type").agg(n=("crop_id", "count")).to_string())
    print("\n== size-bucket distribution (instance buckets inside crop) ==")
    for t in ["normal_positive", "hard_positive", "gt_centered_positive",
              "hard_negative", "easy_negative"]:
        sub = meta[meta.type == t]
        print(t, dict(sub.gt_buckets.value_counts()) if len(sub) else "-")
    print("\n== leakage check ==")
    val_test = set(pd.read_csv(DATA / "split_manifest.csv")[
        pd.read_csv(DATA / "split_manifest.csv").split.isin(["val", "test"])
    ].image_id)
    src_in_split = set(meta.src_image)
    print("val/test images in train crops:", len(src_in_split & val_test))
    print("unique train images contributing:", len(src_in_split))


if __name__ == "__main__":
    main()
