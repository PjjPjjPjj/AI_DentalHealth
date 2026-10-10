"""Mine hard negatives from Mendeley TRAIN base images (val/test untouched).

Protocol:
  tooth-seg crop (iop_toothseg, same as inference) on train-split base images
  -> keep crops with max IoU < 0.1 vs ALL GT boxes (unannotated regions)
  -> v2_sano P(plaque) in [0.70, 0.95]  (borderline = model's uncertain band)
  -> stratified review sheets; visually confirmed negatives become v3 train data

Output: datasets/processed/plaque_mined_negs/ + mined_cands.csv + review sheets
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

ROOT = Path(f"{_ROOT}")
P1 = ROOT / "datasets/raw/Plaque/mendeley-dataset-materials_Part_1"
P2 = ROOT / "datasets/raw/Plaque/mendeley-dataset-materials_Part_2"
PARTS = [P1, P2]
DATA1 = ROOT / "datasets/processed/plaque_bin_v1"
OUT = ROOT / "datasets/processed/plaque_mined_negs"
V2 = ROOT / "outputs/plaque_bin_v2_sano/weights/best.pt"
TOOTH_W = ROOT / "outputs/iop_toothseg/weights/best.pt"
PMIN, PMAX, IOU_MAX = 0.70, 0.95, 0.10


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-9)


def main() -> None:
    sys.path.insert(0, str(ROOT))
    from src.inference.models import ToothModel, PlaqueModel

    rows = pd.read_csv(DATA1 / "plaque_crops.csv")
    tr = rows[(rows.split == "train") & (rows.is_aug == 0) & (rows.is_dup == 0)]
    imgs = tr[["image_id", "part"]].drop_duplicates()
    print("train base images:", len(imgs))

    # index label/image dirs by patient prefix (case-insensitive, mirrors build v1 audit fix)
    lbl_idx = {}
    for part in PARTS:
        for d in (part / "data" / "labels").iterdir():
            if d.is_dir():
                lbl_idx[d.name.lower()] = d
    img_idx = {}
    for part in PARTS:
        for d in (part / "data" / "images").iterdir():
            if d.is_dir():
                img_idx[d.name.lower()] = d

    def patient_of(image_id: str) -> str:
        return image_id.split("_")[0].lower()

    def gt_boxes(image_id: str, H, W):
        pd_ = patient_of(image_id)
        lf = lbl_idx.get(pd_)
        if lf is None:
            return []
        f = lf / f"{image_id}.txt"
        if not f.exists():
            f = next((x for x in lf.iterdir() if x.stem.lower() == image_id.lower()), None)
            if f is None:
                return []
        boxes = []
        for line in f.read_text().splitlines():
            t = line.split()
            if len(t) != 6:
                continue
            cx, cy, w, h = map(float, t[1:5])
            x1 = int(np.clip((cx - w / 2) * W, 0, W)); x2 = int(np.clip((cx + w / 2) * W, 0, W))
            y1 = int(np.clip((cy - h / 2) * H, 0, H)); y2 = int(np.clip((cy + h / 2) * H, 0, H))
            boxes.append((x1, y1, x2, y2))
        return boxes

    tm = ToothModel(TOOTH_W)
    pm = PlaqueModel(V2)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "crops").mkdir(exist_ok=True)
    recs = []
    for k, (_, r) in enumerate(imgs.iterrows()):
        image_id = r.image_id
        pd_ = patient_of(image_id)
        d = img_idx.get(pd_)
        if d is None:
            continue
        f = next((x for x in d.iterdir() if x.stem.lower() == image_id.lower()), None)
        if f is None:
            continue
        img = cv2.imread(str(f))
        if img is None:
            continue
        H, W = img.shape[:2]
        boxes = gt_boxes(image_id, H, W)
        teeth = tm.predict(img)
        cands = []
        for ti, t in enumerate(teeth):
            x0, y0, x1, y1 = [int(max(0.0, v)) for v in t["bbox"]]
            x1, y1 = min(W, x1), min(H, y1)
            if x1 - x0 < 24 or y1 - y0 < 24:   # mined negs: keep informative sizes
                continue
            bb = (x0, y0, x1, y1)
            if boxes and max(iou(bb, g) for g in boxes) >= IOU_MAX:
                continue
            cands.append((ti, bb))
        if not cands:
            continue
        crops = [img[b[1]:b[3], b[0]:b[2]].copy() for _, b in cands]
        probs = pm._proba_array(crops)
        for (ti, bb), p in zip(cands, probs):
            if PMIN <= p <= PMAX:
                cid = f"mine_{image_id}__t{ti:02d}.jpg"
                cv2.imwrite(str(OUT / "crops" / cid),
                            img[bb[1]:bb[3], bb[0]:bb[2]].copy(), [cv2.IMWRITE_JPEG_QUALITY, 95])
                recs.append({"crop": cid, "image_id": image_id, "p_v2": round(float(p), 4),
                             "box": bb})
        if (k + 1) % 100 == 0:
            print(f"{k+1}/{len(imgs)} mined so far: {len(recs)}")

    df = pd.DataFrame(recs)
    df.to_csv(OUT / "mined_cands.csv", index=False)
    print("total candidates:", len(df), "from", df.image_id.nunique() if len(df) else 0, "images")

    # review sheets: stratified by P (equal from 0.70-0.78 / 0.78-0.86 / 0.86-0.95)
    import math
    q = ROOT / "annotation_queue/mined_neg_review"
    q.mkdir(parents=True, exist_ok=True)
    if len(df):
        rng = np.random.RandomState(0)
        bins = [df[(df.p_v2 >= a) & (df.p_v2 < b)] for a, b in
                [(0.70, 0.78), (0.78, 0.86), (0.86, 0.951)]]
        sel = pd.concat([b.sample(min(len(b), 90), random_state=0) for b in bins])
        TH, COLS = 150, 12
        rows_n = math.ceil(len(sel) / COLS)
        sheet = np.full((rows_n * TH, COLS * TH, 3), 30, np.uint8)
        order = sel.sort_values("p_v2", ascending=False).reset_index(drop=True)
        for i, r in order.iterrows():
            im = cv2.imread(str(OUT / "crops" / r.crop))
            if im is None:
                continue
            im = cv2.resize(im, (TH - 6, TH - 24))
            rr, cc = divmod(i, COLS)
            y, x = rr * TH + 2, cc * TH + 2
            sheet[y:y + im.shape[0], x:x + im.shape[1]] = im
            cv2.putText(sheet, f"{r.p_v2:.2f}", (x, y + im.shape[0] + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imwrite(str(q / "mined_review_00.jpg"), sheet)
        print("review sheet ->", q, f"({len(order)} crops)")


if __name__ == "__main__":
    main()
