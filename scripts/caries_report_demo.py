"""Caries demo assets for the report PPT.

Outputs (outputs/demo/caries_report/):
  demo_caries_<id>_compare.jpg  [GT boxes | pred boxes vs GT: TP green / FP blue / FN red]
  dataset_montage.jpg           6 train examples with GT boxes
  demo_summary.json             per-image numbers
Model: outputs/caries_c1/weights/best.pt (YOLO11m @960, frozen conf=0.45)
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

DATA = Path("datasets/processed/caries_det_v2")
MAN = pd.read_csv(DATA / "split_manifest.csv")
OUT = Path("outputs/demo/caries_report")
OUT.mkdir(parents=True, exist_ok=True)
CONF, IMGSZ, IOU = 0.45, 960, 0.6

GREEN, RED, BLUE, YELLOW, WHITE = (0, 180, 0), (0, 0, 255), (255, 80, 0), (0, 255, 255), (255, 255, 255)


def load_gt(label_path: str, W: int, H: int):
    boxes = []
    for line in Path(label_path).read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cx, cy, w, h = map(float, p[1:5])
        boxes.append([int((cx - w / 2) * W), int((cy - h / 2) * H),
                      int((cx + w / 2) * W), int((cy + h / 2) * H)])
    return boxes


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


model = YOLO("outputs/caries_c1/weights/best.pt")

test = MAN[MAN.split == "test"].reset_index(drop=True)
# candidates: 1-4 GT boxes, decent size
cand = []
for _, r in test.iterrows():
    img = cv2.imread(r.img)
    if img is None:
        continue
    H, W = img.shape[:2]
    gts = load_gt(r.label, W, H)
    if 1 <= len(gts) <= 4 and W >= 800:
        cand.append((r, img, gts))
print(f"candidates: {len(cand)}")

summary = []
picked = []
# one clean single-box case + one multi-box case
for want in (1, 3):
    for r, img, gts in cand:
        if len(gts) != want:
            continue
        res = model.predict(r.img, conf=CONF, iou=IOU, imgsz=IMGSZ, device=0, verbose=False)[0]
        preds = [list(map(int, b)) for b in res.boxes.xyxy.cpu().numpy()]
        confs = [float(c) for c in res.boxes.conf.cpu().numpy()]
        matched = 0
        used = set()
        for g in gts:
            best, bi = 0.0, None
            for i, p in enumerate(preds):
                if i in used:
                    continue
                v = iou(g, p)
                if v > best:
                    best, bi = v, i
            if best >= 0.5:
                matched += 1
                used.add(bi)
        if matched == len(gts) and len(preds) == len(gts):
            picked.append((r, img, gts, preds, confs, matched))
            break

for r, img, gts, preds, confs, matched in picked[:2]:
    # panel1: GT
    p1 = img.copy()
    for g in gts:
        cv2.rectangle(p1, (g[0], g[1]), (g[2], g[3]), RED, 3)
    # panel2: pred with TP/FP/FN
    p2 = img.copy()
    used = set()
    for g in gts:
        best, bi = 0.0, None
        for i, p in enumerate(preds):
            if i in used:
                continue
            v = iou(g, p)
            if v > best:
                best, bi = v, i
        if best >= 0.5:
            used.add(bi)
            p = preds[bi]
            cv2.rectangle(p2, (p[0], p[1]), (p[2], p[3]), GREEN, 3)
            cv2.putText(p2, f"{confs[bi]:.2f}", (p[0], max(20, p[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
        else:
            cv2.rectangle(p2, (g[0], g[1]), (g[2], g[3]), RED, 2)
            cv2.putText(p2, "FN", (g[0], max(20, g[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
    for i, p in enumerate(preds):
        if i not in used:
            cv2.rectangle(p2, (p[0], p[1]), (p[2], p[3]), BLUE, 3)
            cv2.putText(p2, f"FP {confs[i]:.2f}", (p[0], max(20, p[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLUE, 2)
    scale = 560 / max(img.shape[:2])
    p1 = cv2.resize(p1, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
    p2 = cv2.resize(p2, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
    gap = np.full((p1.shape[0], 14, 3), 255, np.uint8)
    canvas = np.hstack([p1, gap, p2])
    fid = Path(r.img).stem
    out_path = OUT / f"demo_caries_{fid}_compare.jpg"
    cv2.imwrite(str(out_path), canvas)
    tp = len(gts) if matched == len(gts) else matched
    summary.append(dict(image_id=fid, file=str(out_path), n_gt=len(gts), n_pred=len(preds),
                        matched=matched, confs=[round(c, 3) for c in confs]))
    print(f"{fid}: gt={len(gts)} pred={len(preds)} matched={matched} confs={[round(c,2) for c in confs]}")

# dataset montage: 6 train images with GT boxes
tr = MAN[MAN.split == "train"].reset_index(drop=True)
cells = []
for _, r in tr.iterrows():
    if len(cells) >= 6:
        break
    img = cv2.imread(r.img)
    if img is None:
        continue
    H, W = img.shape[:2]
    gts = load_gt(r.label, W, H)
    if not (1 <= len(gts) <= 3):
        continue
    for g in gts:
        cv2.rectangle(img, (g[0], g[1]), (g[2], g[3]), RED, 3)
    cells.append((Path(r.img).name, img))
CW, CH, LBL = 300, 215, 20
rows_n = (len(cells) + 2) // 3
grid = np.full((rows_n * (CH + LBL + 8) + 4, 3 * CW + 24, 3), 255, np.uint8)
for i, (lbl, img) in enumerate(cells):
    rr, cc = divmod(i, 3)
    y0, x0 = rr * (CH + LBL + 8) + 4, cc * (CW + 8) + 4
    sc = min(CW / img.shape[1], CH / img.shape[0])
    rs = cv2.resize(img, (int(img.shape[1] * sc), int(img.shape[0] * sc)))
    grid[y0:y0 + rs.shape[0], x0:x0 + rs.shape[1]] = rs
    cv2.putText(grid, lbl[:40], (x0, y0 + CH + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1)
cv2.imwrite(str(OUT / "dataset_montage.jpg"), grid)
print("montage cells:", len(cells))

json.dump(dict(conf=CONF, imgsz=IMGSZ, model="outputs/caries_c1/weights/best.pt",
               test_metrics=dict(P=0.8889, R=0.8737, F1=0.8813,
                                 mAP50=0.9335, mAP50_95=0.7135, conf=0.45),
               baseline=dict(model="yolo11s@640", P=0.8507, R=0.9018, F1=0.8755,
                             mAP50=0.9277, mAP50_95=0.7190)),
          open(OUT / "demo_summary.json", "w"), indent=1)
json.dump(summary, open(OUT / "demo_images.json", "w"), ensure_ascii=False, indent=1)
print("done")
