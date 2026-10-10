"""Build v5b stage1 FT dataset: v5_zoom (symlinked) + copy-paste synthetics.

Sources: train-split calculus blobs with area ratio >= 0.8% and >=120 px
(189 available in v3 train). Targets: random v5_zoom train images.
Each target gets 1-2 blobs scaled to 2-7% of target area, blended with
seamlessClone, label polygon appended (class 0).

Output: datasets/processed/calculus_v5b_zoom/{images,labels}/{train,val}
        + zoom_v5b.yaml
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

V3 = Path(f"{_ROOT}/datasets/processed/calculus_seg_v3")
V5Z = Path(f"{_ROOT}/datasets/processed/calculus_v5_zoom")
OUT = Path(f"{_ROOT}/datasets/processed/calculus_v5b_zoom")
N_SYNTH = 1200
MIN_SOURCE_PX = 120
MIN_SOURCE_RATIO = 0.008
RATIO_LO, RATIO_HI = 0.02, 0.07
SEED = 42


def collect_sources(man):
    blobs = []
    tr = man[(man.split == "train") & (man.has_calculus == True)]  # noqa: E712
    for _, r in tr.iterrows():
        m = cv2.imread(r.mask_original_path, cv2.IMREAD_UNCHANGED)
        if m is None:
            continue
        if m.ndim == 3:
            m = m[:, :, 0]
        g = (m > 0).astype(np.uint8)
        H, W = g.shape
        n, comp = cv2.connectedComponents(g, 8)
        img = cv2.imread(r.image_path)
        if img is None:
            continue
        for i in range(1, n):
            px = int((comp == i).sum())
            if px < MIN_SOURCE_PX or px / (H * W) < MIN_SOURCE_RATIO:
                continue
            bm = (comp == i).astype(np.uint8)
            x, y, w, h = cv2.boundingRect(bm)
            pad = 2
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
            crop = img[y0:y1, x0:x1]
            mcrop = bm[y0:y1, x0:x1]
            if crop.size == 0 or mcrop.sum() < MIN_SOURCE_PX:
                continue
            blobs.append({"img": crop, "mask": mcrop, "px": int(mcrop.sum())})
    return blobs


def paste_polygon_mask(target_shape, mask_scaled, x0, y0):
    """Return polygon points (orig coords) of the pasted mask."""
    cnts, _ = cv2.findContours(mask_scaled, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for c in cnts:
        if cv2.contourArea(c) < 4:
            continue
        pts.extend(c.reshape(-1, 2) + [x0, y0])
    return np.array(pts)


def main() -> None:
    rng = random.Random(SEED)
    if OUT.exists():
        shutil.rmtree(OUT)
    for sp in ("train", "val"):
        (OUT / "images" / sp).mkdir(parents=True)
        (OUT / "labels" / sp).mkdir(parents=True)

    # symlink existing v5_zoom
    for sp in ("train", "val"):
        for img in sorted((V5Z / "images" / sp).iterdir()):
            lbl = V5Z / "labels" / sp / (img.stem + ".txt")
            if not lbl.exists():
                continue
            (OUT / "images" / sp / img.name).symlink_to(img)
            (OUT / "labels" / sp / (img.stem + ".txt")).symlink_to(lbl)

    man = pd.read_csv(V3 / "split_manifest_v3.csv")
    blobs = collect_sources(man)
    print(f"source blobs: {len(blobs)}", flush=True)

    tr_imgs = sorted((OUT / "images" / "train").glob("*.jpg"))
    targets = rng.sample(tr_imgs, min(N_SYNTH, len(tr_imgs)))
    n_added = 0
    for k, timg_path in enumerate(targets):
        tlbl_path = OUT / "labels" / "train" / (timg_path.stem + ".txt")
        img = cv2.imread(str(timg_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        label_lines = tlbl_path.read_text().splitlines() if tlbl_path.exists() else []
        # existing GT centroids (from label polygons) for plausible placement
        cents = []
        for ln in label_lines:
            p = ln.split()
            if len(p) < 5 or p[0] != "0":
                continue
            xy = np.array(p[1:], float).reshape(-1, 2) * [W, H]
            cents.append(xy.mean(0))
        work = img.copy()
        new_lines = []
        for _ in range(rng.choice([1, 1, 2])):
            src = rng.choice(blobs)
            sh, sw = src["mask"].shape[:2]
            ratio = rng.uniform(RATIO_LO, RATIO_HI)
            scale = float(np.sqrt(min(ratio * H * W / max(src["px"], 1), 1.0)))
            if scale > 3.0 or scale < 0.15:
                continue
            nw, nh = max(4, int(sw * scale)), max(4, int(sh * scale))
            crop = cv2.resize(src["img"], (nw, nh), interpolation=cv2.INTER_AREA)
            msk = cv2.resize(src["mask"], (nw, nh), interpolation=cv2.INTER_NEAREST)
            if cents and rng.random() < 0.65:
                c = rng.choice(cents)
                cx = int(np.clip(c[0] + rng.gauss(0, 0.08 * W), nw // 2, W - nw // 2))
                cy = int(np.clip(c[1] + rng.gauss(0, 0.08 * H), nh // 2, H - nh // 2))
            else:
                cx = rng.randint(nw // 2, max(nw // 2, W - nw // 2))
                cy = rng.randint(nh // 2, max(nh // 2, H - nh // 2))
            x0, y0 = cx - nw // 2, cy - nh // 2
            x1, y1 = min(W, x0 + nw), min(H, y0 + nh)
            x0, y0 = max(0, x0), max(0, y0)
            if x1 - x0 < 6 or y1 - y0 < 6:
                continue
            crop = crop[:y1 - y0, :x1 - x0]
            msk = msk[:y1 - y0, :x1 - x0]
            if msk.sum() < 20:
                continue
            center = ((x0 + (x1 - x0) // 2), (y0 + (y1 - y0) // 2))
            mask3 = np.repeat(msk[:, :, None], 3, 2).astype(np.uint8) * 255
            try:
                work = cv2.seamlessClone(crop, work, mask3, center, cv2.NORMAL_CLONE)
            except Exception:
                continue
            poly = paste_polygon_mask((H, W), msk, x0, y0)
            if len(poly) < 6:
                continue
            xy = poly / [W, H]
            xy = np.clip(xy, 0, 1)
            new_lines.append("0 " + " ".join(f"{v:.6f}" for v in xy.ravel()))
        if not new_lines:
            continue
        name = f"synth_{k:05d}"
        cv2.imwrite(str(OUT / "images" / "train" / f"{name}.jpg"), work)
        (OUT / "labels" / "train" / f"{name}.txt").write_text(
            "\n".join(list(label_lines) + new_lines) + "\n")
        n_added += 1
        if (k + 1) % 200 == 0:
            print(f"  synth {k+1}/{len(targets)}", flush=True)

    (OUT / "zoom_v5b.yaml").write_text(
        f"path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n"
        f"  0: calculus\n  1: other_lesion\n")
    print("synth images written:", n_added)
    print("train total:", len(list((OUT / 'images' / 'train').iterdir())),
          "val:", len(list((OUT / 'images' / 'val').iterdir())))


if __name__ == "__main__":
    main()
