"""V4 zoom dataset = v3c zoom (Oral-Mamba windows/closeups)
+ synthetic small-lesion windows (GT-centered + 1 random)
+ Lesi whole images (deduped, class remap 0->0, 1/2/3->1)
+ tzg18 subset (400, bbox -> 4-pt polygon).

val stays pure Oral-Mamba (v3c zoom val) for honest calibration.
Output: datasets/processed/calculus_v4_zoom/{images,labels}/{train,val}
        + zoom_v4.yaml
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

V3Z = Path(f"{_ROOT}/datasets/processed/calculus_v3c_zoom")
V4Z = Path(f"{_ROOT}/datasets/processed/calculus_v4_zoom")
SYN = Path(f"{_ROOT}/datasets/processed/calculus_v4_synth")
LESI = Path(f"{_ROOT}/datasets/raw/Lesi_Intraoral")
TZG = Path(f"{_ROOT}/datasets/raw/roboflow_calculus_tzg18")
WIN = 320
TARGET = 640
SEED = 42
TZG_N = 400


def rasterize_label(lbl: Path, W: int, H: int):
    """returns {cls: mask} from a yolo-seg label file."""
    masks = {}
    for line in lbl.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        c = int(p[0])
        xy = np.array(p[1:], float).reshape(-1, 2) * [W, H]
        m = masks.setdefault(c, np.zeros((H, W), np.uint8))
        cv2.fillPoly(m, [np.round(xy).astype(np.int32)], 1)
    return masks


def lines_from_window(masks, win, W, H):
    x0, y0, x1, y1 = win
    cw, ch = x1 - x0, y1 - y0
    lines = []
    for c, m in masks.items():
        sub = m[y0:y1, x0:x1]
        cnt, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sx, sy = TARGET / cw, TARGET / ch
        for cc in cnt:
            if cv2.contourArea(cc) < 5:
                continue
            pts = cc.reshape(-1, 2)
            lines.append(f"{c} " + " ".join(
                f"{min(max(x * sx / TARGET, 0), 1):.6f} "
                f"{min(max(y * sy / TARGET, 0), 1):.6f}" for x, y in pts))
    return lines


def main() -> None:
    rng = np.random.default_rng(SEED)
    if V4Z.exists():
        import shutil
        shutil.rmtree(V4Z)
    for sp in ("train", "val"):
        (V4Z / "images" / sp).mkdir(parents=True)
        (V4Z / "labels" / sp).mkdir(parents=True)

    # 1) carry over v3c zoom (symlinks)
    n_carry = 0
    for sp in ("train", "val"):
        for img in (V3Z / "images" / sp).glob("*"):
            (V4Z / "images" / sp / img.name).symlink_to(img.resolve())
            (V4Z / "labels" / sp / (img.stem + ".txt")).symlink_to(
                (V3Z / "labels" / sp / (img.stem + ".txt")).resolve())
            n_carry += 1
    print(f"carried v3c zoom: {n_carry}")

    # 2) synthetic windows (train only)
    n_syn = 0
    for lf in sorted(SYN.glob("labels/*.txt")):
        stem = lf.stem
        img_path = SYN / "images" / f"{stem}.jpg"
        img = cv2.imread(str(img_path))
        H, W = img.shape[:2]
        masks = rasterize_label(lf, W, H)
        if 0 not in masks:
            continue
        calc = masks[0]
        n, comp, _, cents = cv2.connectedComponentsWithStats(calc, 8)
        centers = [(cents[i][0], cents[i][1]) for i in range(1, n)
                   if comp[i].sum() >= 5]
        side = min(WIN, max(W, H))
        wins = []
        for cx, cy in centers:
            x0 = int(np.clip(cx - side / 2, 0, max(W - side, 0)))
            y0 = int(np.clip(cy - side / 2, 0, max(H - side, 0)))
            w = (x0, y0, min(x0 + side, W), min(y0 + side, H))
            if all(abs(w[0] - u[0]) + abs(w[1] - u[1]) > 20 for u in wins):
                wins.append(w)
        x0 = int(rng.integers(0, max(W - side, 1)))
        y0 = int(rng.integers(0, max(H - side, 1)))
        wins.append((x0, y0, min(x0 + side, W), min(y0 + side, H)))
        for wi, win in enumerate(wins):
            x0, y0, x1, y1 = win
            crop = img[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            name = f"{stem}__w{wi}"
            cv2.imwrite(str(V4Z / "images" / "train" / f"{name}.jpg"), crop)
            (V4Z / "labels" / "train" / f"{name}.txt").write_text(
                "\n".join(lines_from_window(masks, win, W, H)))
            n_syn += 1
    print(f"synthetic windows: {n_syn}")

    # 3) Lesi whole (dedup by md5, class remap)
    n_lesi = 0
    seen_md5 = set()
    for sp in ("train", "valid", "test"):
        for img in (LESI / sp / "images").glob("*"):
            md5 = hashlib.md5(img.read_bytes()).hexdigest()
            if md5 in seen_md5:
                continue
            seen_md5.add(md5)
            lbl = LESI / sp / "labels" / (img.stem + ".txt")
            if not lbl.exists():
                continue
            out_lines = []
            for line in lbl.read_text().splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                c = 0 if int(p[0]) == 0 else 1
                out_lines.append(str(c) + " " + " ".join(p[1:]))
            name = f"lesi_{img.stem}"[:80]
            (V4Z / "images" / "train" / f"{name}{img.suffix}").symlink_to(
                img.resolve())
            (V4Z / "labels" / "train" / f"{name}.txt").write_text(
                "\n".join(out_lines))
            n_lesi += 1
    print(f"lesi images: {n_lesi}")

    # 4) tzg18 subset (bbox -> polygon)
    tzg_meta = yaml.safe_load((TZG / "data.yaml").read_text())
    timgs = sorted((TZG / "train" / "images").glob("*"))
    pick = rng.choice(len(timgs), size=min(TZG_N, len(timgs)), replace=False)
    n_tzg = 0
    for k in pick:
        img = timgs[int(k)]
        lbl = TZG / "train" / "labels" / (img.stem + ".txt")
        if not lbl.exists():
            continue
        lines = []
        for line in lbl.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            cx, cy, w, h = map(float, p[1:5])
            x0, y0 = cx - w / 2, cy - h / 2
            x1, y1 = cx + w / 2, cy + h / 2
            lines.append("0 " + " ".join(f"{v:.6f}" for v in
                                         (x0, y0, x1, y0, x1, y1, x0, y1)))
        name = f"tzg_{img.stem}"[:80]
        (V4Z / "images" / "train" / f"{name}{img.suffix}").symlink_to(img.resolve())
        (V4Z / "labels" / "train" / f"{name}.txt").write_text("\n".join(lines))
        n_tzg += 1
    print(f"tzg images: {n_tzg}")

    (V4Z / "zoom_v4.yaml").write_text(
        f"path: {V4Z}\ntrain: images/train\nval: images/val\n"
        f"names:\n  0: calculus\n  1: other_lesion\n")
    for sp in ("train", "val"):
        print(sp, "total:", len(list((V4Z / "images" / sp).iterdir())))


if __name__ == "__main__":
    main()
