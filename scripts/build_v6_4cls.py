"""V6: 4-class zoom dataset (0=calculus px3, 1=gingivitis px1, 2=plaque px2,
3=caries px4) — same window logic as v3c_zoom, Oral-Mamba only.

Output: datasets/processed/calculus_v6_4cls/{images,labels}/{train,val}
        + zoom_v6.yaml
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

SRC = Path(f"{_ROOT}/datasets/processed/calculus_seg_v3")
OUT = Path(f"{_ROOT}/datasets/processed/calculus_v6_4cls")
RAW = Path(f"{_ROOT}/datasets/raw/oral_mamba/lables")
WIN = 320
MIN_PX = 5
CLS = {3: 0, 1: 1, 2: 2, 4: 3}


def window_for(cx, cy, W, H, side=WIN):
    side = min(side, max(W, H))
    x0 = int(np.clip(cx - side / 2, 0, max(W - side, 0)))
    y0 = int(np.clip(cy - side / 2, 0, max(H - side, 0)))
    return (x0, y0, min(x0 + side, W), min(y0 + side, H))


def win_iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua else 0.0


def lines_in(full, win, W, H):
    x0, y0, x1, y1 = win
    cw, ch = x1 - x0, y1 - y0
    lines = []
    for px, cid in CLS.items():
        m = (full[y0:y1, x0:x1] == px).astype(np.uint8)
        cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sx, sy = 640 / cw, 640 / ch
        for cc in cnt:
            if cv2.contourArea(cc) < MIN_PX:
                continue
            pts = cc.reshape(-1, 2)
            lines.append(f"{cid} " + " ".join(
                f"{min(max(x * sx / 640, 0), 1):.6f} "
                f"{min(max(y * sy / 640, 0), 1):.6f}" for x, y in pts))
    return lines


def main() -> None:
    import shutil
    rng = np.random.default_rng(42)
    if OUT.exists():
        shutil.rmtree(OUT)
    for sp in ("train", "val"):
        (OUT / "images" / sp).mkdir(parents=True)
        (OUT / "labels" / sp).mkdir(parents=True)
    man = pd.read_csv(SRC / "split_manifest_v3.csv")
    for sp in ("train", "val"):
        rows = man[man.split == sp].reset_index(drop=True)
        for ri, r in rows.iterrows():
            img = cv2.imread(r.image_path)
            H, W = img.shape[:2]
            full = cv2.imread(str(RAW / f"{r.image_id}.png"), cv2.IMREAD_UNCHANGED)
            if full is None:
                full = np.zeros((H, W), np.uint8)
            if full.ndim == 3:
                full = full[:, :, 0]
            calc = (full == 3).astype(np.uint8)
            wins = []
            if calc.any():
                n, comp, _, cents = cv2.connectedComponentsWithStats(calc, 8)
                for i in range(1, n):
                    if comp[i].sum() < MIN_PX:
                        continue
                    wins.append(window_for(cents[i][0], cents[i][1], W, H))
            x0 = int(rng.integers(0, max(W - min(WIN, W), 1)))
            y0 = int(rng.integers(0, max(H - min(WIN, H), 1)))
            wins.append((x0, y0, min(x0 + WIN, W), min(y0 + WIN, H)))
            ded = []
            for w in wins:
                if all(win_iou(w, u) < 0.9 for u in ded):
                    ded.append(w)
            for wi, win in enumerate(ded):
                x0, y0, x1, y1 = win
                crop = img[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                lines = lines_in(full, win, W, H)
                name = f"{r.image_id}__w{wi}"
                cv2.imwrite(str(OUT / "images" / sp / f"{name}.jpg"), crop)
                (OUT / "labels" / sp / f"{name}.txt").write_text("\n".join(lines))
            if (ri + 1) % 400 == 0:
                print(f"{sp} {ri+1}/{len(rows)}", flush=True)
    (OUT / "zoom_v6.yaml").write_text(
        f"path: {OUT}\ntrain: images/train\nval: images/val\n"
        f"names:\n  0: calculus\n  1: gingivitis\n  2: plaque\n  3: caries\n")
    for sp in ("train", "val"):
        print(sp, len(list((OUT / "images" / sp).iterdir())))


if __name__ == "__main__":
    main()
