"""V4 synthetic small-lesion generation (copy-paste at reduced scale).

Attacks the small-lesion FN bottleneck directly: paste real GT calculus
components (from Oral-Mamba train binary masks) onto NEGATIVE train images
at 0.3-0.7x scale, with exact polygon labels.  800 synthetic images,
1-3 lesions each.

Output: datasets/processed/calculus_v4_synth/{images,labels}
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
SEED = 42
N_SYNTH = 800
SCALE = (0.3, 0.7)
MIN_PX = 30

rng = np.random.default_rng(SEED)


def load_components(mask_path: Path) -> list[np.ndarray]:
    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if m is None:
        return []
    if m.ndim == 3:
        m = m[:, :, 0]
    if not m.any():
        return []
    n, comp = cv2.connectedComponents((m > 0).astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        c = (comp == i).astype(np.uint8)
        if c.sum() >= MIN_PX:
            out.append(c)
    return out


def main() -> None:
    man = pd.read_csv(SRC / "split_manifest_v3.csv")
    tr = man[man.split == "train"].reset_index(drop=True)
    neg = tr[~tr.has_calculus].reset_index(drop=True)
    pos = tr[tr.has_calculus].reset_index(drop=True)
    print(f"neg {len(neg)}, pos {len(pos)}")

    # bank of real calculus components
    bank = []
    for r in pos.itertuples():
        bank.extend(load_components(Path(r.mask_original_path)))
    print(f"component bank: {len(bank)}")

    out = Path(f"{_ROOT}/datasets/processed/calculus_v4_synth")
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    pick = rng.choice(len(neg), size=min(N_SYNTH, len(neg)), replace=False)
    n_written = 0
    for k in pick:
        r = neg.iloc[int(k)]
        img = cv2.imread(r.image_path)
        H, W = img.shape[:2]
        # full multiclass mask for other-lesion labels
        full = cv2.imread(f"{_ROOT}/datasets/raw/oral_mamba/lables/{r.image_id}.png",
                          cv2.IMREAD_UNCHANGED)
        if full is None:
            full = np.zeros((H, W), np.uint8)
        if full.ndim == 3:
            full = full[:, :, 0]
        other0 = np.isin(full, [1, 2, 4]).astype(np.uint8)

        paste_mask = np.zeros((H, W), np.uint8)
        n_inst = int(rng.integers(1, 4))
        placed = []
        for _ in range(n_inst):
            comp = bank[int(rng.integers(0, len(bank)))]
            ch, cw = comp.shape
            scale = float(rng.uniform(*SCALE))
            nw, nh = max(2, int(cw * scale)), max(2, int(ch * scale))
            nw, nh = min(nw, W), min(nh, H)
            x0 = int(rng.integers(0, max(W - nw, 1)))
            y0 = int(rng.integers(0, max(H - nh, 1)))
            c = cv2.resize(comp, (nw, nh), interpolation=cv2.INTER_NEAREST)
            box = paste_mask[y0:y0 + nh, x0:x0 + nw]
            if (box & c).sum() > 0.2 * c.sum():
                continue
            ys, xs = np.where(c > 0)
            img[y0 + ys, x0 + xs] = img[np.clip(y0 + ys + int(rng.integers(-20, 21)), 0, H - 1),
                                        np.clip(x0 + xs + int(rng.integers(-20, 21)), 0, W - 1)]
            paste_mask[y0:y0 + nh, x0:x0 + nw] |= c
            # comp array trimmed when window exceeds bounds
            if nh != int(ch * scale) or nw != int(cw * scale):
                comp = cv2.resize(comp, (nw, nh), interpolation=cv2.INTER_NEAREST)
                ch, cw = nh, nw
            placed.append((c, y0, x0))

        name = f"synth_{n_written:04d}"
        cv2.imwrite(str(out / "images" / f"{name}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        lines = []
        for c, oy, ox in placed:
            cnt, _ = cv2.findContours(c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cc in cnt:
                if cv2.contourArea(cc) < 5:
                    continue
                pts = cc.reshape(-1, 2) + np.array([ox, oy])
                lines.append("0 " + " ".join(
                    f"{min(max(x / W, 0), 1):.6f} {min(max(y / H, 0), 1):.6f}"
                    for x, y in pts))
        lines += contours_to(other0, 1, W, H)
        (out / "labels" / f"{name}.txt").write_text("\n".join(lines))
        n_written += 1
        if n_written % 200 == 0:
            print(n_written, flush=True)
    print("synthetic images written:", n_written)


def contours_to(mask, cls_id, W, H):
    lines = []
    cnt, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cc in cnt:
        if cv2.contourArea(cc) < 8:
            continue
        pts = cc.reshape(-1, 2)
        lines.append(f"{cls_id} " + " ".join(
            f"{min(max(x / W, 0), 1):.6f} {min(max(y / H, 0), 1):.6f}"
            for x, y in pts))
    return lines


if __name__ == "__main__":
    main()
