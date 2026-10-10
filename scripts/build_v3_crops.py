"""V3 Stage2 crop dataset: S1 proposals + GT-centered windows, 3-class labels.

Windows: side=160 (clipped to image bounds), resized to 640.
Labels (640-space): class0 = calculus(px3) components in window,
                    class1 = other_lesion(px1|2|4) components in window.
Windows with no calculus label are downsampled to 60% (deployment realism
keeps most negatives, avoids drowning positives).
Output: datasets/processed/calculus_seg_v3_crops/{images,labels}/{train,val}
        + crops_meta.csv
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from src.utils.paths import PROCESSED_DIR

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

DATA = PROCESSED_DIR / "calculus_seg_v3"
OUT = PROCESSED_DIR / "calculus_seg_v3_crops"
S1_W = "outputs/calculus_v3b_stage1/weights/best.pt"
C1 = 0.02
SIDE = 160
TARGET = 640
MIN_PX = 5
NEG_KEEP = 0.6
SEED = 42


def window_for(cx: float, cy: float, W: int, H: int) -> tuple[int, int, int, int]:
    side = min(SIDE, max(W, H))
    x0 = int(round(cx - side / 2)); y0 = int(round(cy - side / 2))
    x0 = max(0, min(x0, W - side)); y0 = max(0, min(y0, H - side))
    return x0, y0, min(x0 + side, W), min(y0 + side, H)


def win_iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


def lines_in_window(full: np.ndarray, win, vals: list[int], cls_id: int) -> list[str]:
    x0, y0, x1, y1 = win
    cw, chh = x1 - x0, y1 - y0
    m = np.isin(full[y0:y1, x0:x1], vals).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    sx, sy = TARGET / cw, TARGET / chh
    for c in contours:
        if cv2.contourArea(c) < MIN_PX:
            continue
        pts = c.reshape(-1, 2)
        lines.append(f"{cls_id} " + " ".join(
            f"{min(max(x * sx / TARGET, 0), 1):.6f} "
            f"{min(max(y * sy / TARGET, 0), 1):.6f}" for x, y in pts))
    return lines


def main() -> None:
    rng = np.random.default_rng(SEED)
    if OUT.exists():
        shutil.rmtree(OUT)
    for sp in ("train", "val"):
        (OUT / "images" / sp).mkdir(parents=True)
        (OUT / "labels" / sp).mkdir(parents=True)
    model = YOLO(S1_W)
    man = pd.read_csv(DATA / "split_manifest_v3.csv")
    raw_dir = Path(f"{_ROOT}/datasets/raw/oral_mamba/lables")
    meta = []
    for sp in ("train", "val"):
        rows = man[man.split == sp].reset_index(drop=True)
        for ri, r in rows.iterrows():
            img = cv2.imread(r.image_path)
            H, W = img.shape[:2]
            full = cv2.imread(str(raw_dir / f"{r.image_id}.png"), cv2.IMREAD_UNCHANGED)
            if full is None:
                full = np.zeros((H, W), np.uint8)
            if full.ndim == 3:
                full = full[:, :, 0]
            # GT calculus component centers
            centers: list[tuple[float, float]] = []
            calc = (full == 3).astype(np.uint8)
            if calc.any():
                n, comp, stats, cents = cv2.connectedComponentsWithStats(calc, 8)
                for i in range(1, n):
                    if stats[i, cv2.CC_STAT_AREA] < MIN_PX:
                        continue
                    centers.append((cents[i][0], cents[i][1]))
            # S1 proposal centers
            res = model.predict(r.image_path, conf=C1, iou=0.6, imgsz=640,
                                device=0, retina_masks=True, max_det=100,
                                verbose=False)[0]
            if res.boxes is not None and len(res.boxes):
                bx = res.boxes.xyxy.cpu().numpy()
                keep_cls = res.boxes.cls.cpu().numpy().astype(int) == 0
                bx = bx[keep_cls]
                for b in bx:
                    centers.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
            # dedup windows
            wins: list[tuple[int, int, int, int]] = []
            for cx, cy in centers:
                w = window_for(cx, cy, W, H)
                if all(win_iou(w, u) < 0.9 for u in wins):
                    wins.append(w)
            recs = []
            for wi, win in enumerate(wins):
                l0 = lines_in_window(full, win, [3], 0)
                l1 = lines_in_window(full, win, [1, 2, 4], 1)
                recs.append((win, l0, l1))
            neg = [i for i, (_, l0, _) in enumerate(recs) if not l0]
            drop = set(rng.choice(neg, size=int(len(neg) * (1 - NEG_KEEP)),
                                  replace=False).tolist()) if neg else []
            for wi, (win, l0, l1) in enumerate(recs):
                if wi in drop:
                    continue
                x0, y0, x1, y1 = win
                crop = img[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                name = f"{r.image_id}__w{wi}.jpg"
                cv2.imwrite(str(OUT / "images" / sp / name), crop)
                (OUT / "labels" / sp / f"{r.image_id}__w{wi}.txt").write_text(
                    "\n".join(l0 + l1))
                meta.append({"split": sp, "image_id": r.image_id,
                             "domain": r.domain, "crop": name,
                             "n_calc": len(l0), "n_other": len(l1),
                             "source": "proposal"})
            if (ri + 1) % 300 == 0:
                print(f"{sp} {ri + 1}/{len(rows)}", flush=True)
    pd.DataFrame(meta).to_csv(OUT / "crops_meta.csv", index=False)
    (OUT / "crops.yaml").write_text(
        f"path: {OUT}\ntrain: images/train\nval: images/val\n"
        f"names:\n  0: calculus\n  1: other_lesion\n")
    df = pd.DataFrame(meta)
    print("crops with calculus:")
    print(df.assign(pos=df.n_calc > 0).groupby("split").pos.agg(["sum", "mean"]))
    print("total crops:", len(df))


if __name__ == "__main__":
    main()
