"""V3c: zoomed Stage1 training data for wide images (tiled paradigm).

Wide images (max(W,H)>=500) are represented by zoomed 320px windows
(2 random + 1 GT-centered if calculus positive), upscaled to 640 (2x zoom).
Closeups (<500) stay whole (reuse existing image + label via symlink).
Gingivitis/negative wides: 2 random windows each (FP suppression in tile
context).  3-class labels (calculus px3 / other px1|2|4) per window.

Output: datasets/processed/calculus_v3c_zoom/{images,labels}/{train,val}
        + zoom.yaml (+ train list = directory scan)
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

from src.utils.paths import PROCESSED_DIR

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

DATA = PROCESSED_DIR / "calculus_seg_v3"
OUT = PROCESSED_DIR / "calculus_v3c_zoom"
RAW = Path(f"{_ROOT}/datasets/raw/oral_mamba/lables")
WIN = 320
TARGET = 640
MIN_PX = 5
WIDE_MIN = 500
SEED = 42


def lines_in_window(full, win, vals, cls_id, cw, ch):
    x0, y0, x1, y1 = win
    m = np.isin(full[y0:y1, x0:x1], vals).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sx, sy = TARGET / cw, TARGET / ch
    lines = []
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
    man = pd.read_csv(DATA / "split_manifest_v3.csv")
    n_wide = n_close = 0
    for sp in ("train", "val"):
        rows = man[man.split == sp].reset_index(drop=True)
        for ri, r in rows.iterrows():
            img = cv2.imread(r.image_path)
            H, W = img.shape[:2]
            wide = max(W, H) >= WIDE_MIN
            if not wide:
                n_close += 1
                (OUT / "images" / sp / f"{r.image_id}.jpg").symlink_to(r.image_path)
                (OUT / "labels" / sp / f"{r.image_id}.txt").symlink_to(
                    DATA / "labels" / sp / f"{r.image_id}.txt")
                continue
            n_wide += 1
            full = cv2.imread(str(RAW / f"{r.image_id}.png"), cv2.IMREAD_UNCHANGED)
            if full is None:
                full = np.zeros((H, W), np.uint8)
            if full.ndim == 3:
                full = full[:, :, 0]
            wins = []
            k = 2
            for _ in range(k):
                x0 = int(rng.integers(0, max(W - WIN, 1)))
                y0 = int(rng.integers(0, max(H - WIN, 1)))
                wins.append((x0, y0, min(x0 + WIN, W), min(y0 + WIN, H)))
            if r.has_calculus:
                calc = (full == 3).astype(np.uint8)
                n_c, comp, stats, cents = cv2.connectedComponentsWithStats(calc, 8)
                areas = stats[1:, cv2.CC_STAT_AREA]
                if len(areas):
                    gi = 1 + int(np.argmax(areas))
                    cx, cy = cents[gi]
                    x0 = int(np.clip(cx - WIN / 2, 0, max(W - WIN, 0)))
                    y0 = int(np.clip(cy - WIN / 2, 0, max(H - WIN, 0)))
                    wins.append((x0, y0, min(x0 + WIN, W), min(y0 + WIN, H)))
            for wi, win in enumerate(wins):
                x0, y0, x1, y1 = win
                cw, ch = x1 - x0, y1 - y0
                crop = img[y0:y1, x0:x1]
                if crop.size == 0 or cw < 40 or ch < 40:
                    continue
                l0 = lines_in_window(full, win, [3], 0, cw, ch)
                l1 = lines_in_window(full, win, [1, 2, 4], 1, cw, ch)
                name = f"{r.image_id}__z{wi}"
                cv2.imwrite(str(OUT / "images" / sp / f"{name}.jpg"), crop)
                (OUT / "labels" / sp / f"{name}.txt").write_text(
                    "\n".join(l0 + l1))
            if (ri + 1) % 400 == 0:
                print(f"{sp} {ri + 1}/{len(rows)}", flush=True)
    (OUT / "zoom.yaml").write_text(
        f"path: {OUT}\ntrain: images/train\nval: images/val\n"
        f"names:\n  0: calculus\n  1: other_lesion\n")
    print(f"wide imgs: {n_wide}, closeups: {n_close}")
    for sp in ("train", "val"):
        print(sp, "samples:", len(list((OUT / 'images' / sp).iterdir())))


if __name__ == "__main__":
    main()
