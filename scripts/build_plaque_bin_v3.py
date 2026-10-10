"""Build plaque_bin_v3 = v2 + mined hard negatives + SANO blur/warm offline augs
+ val negatives expanded with 20 SANO images (moved out of train).

Ingredients:
  A. move 20 SANO train images (deterministic pick) from train/normal to
     val/normal  -> val negatives 96 -> ~380, tightens threshold freezing
  B. mined negatives (visually confirmed) from plaque_mined_negs/ -> train/normal
  C. offline robustness variants of every SANO train crop (1 per crop, type by
     md5): gaussian-blur OR warm-dim (Canon-like cast) -> train/normal
     (targets the blur/warm-color shortcut directly)

Output: datasets/processed/plaque_bin_v3 (+ plaque_bin_v3.yaml)
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

import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

ROOT = Path(f"{_ROOT}")
V2 = ROOT / "datasets/processed/plaque_bin_v2"
V3 = ROOT / "datasets/processed/plaque_bin_v3"
SANO_CSV = ROOT / "datasets/processed/plaque_sano/sano_crops.csv"
SANO_DIR = ROOT / "datasets/processed/plaque_sano"
MINED = ROOT / "datasets/processed/plaque_mined_negs"
N_VAL_IMAGES = 20


def robust_variant(img: np.ndarray, seed_hex: str) -> np.ndarray:
    """One deterministic robustness variant: gaussian-blur OR warm-dim."""
    r = int(seed_hex[:8], 16)
    if r % 2 == 0:
        k = 9 if r % 4 < 2 else 13
        return cv2.GaussianBlur(img, (k, k), sigmaX=2.0 + (r % 100) / 50.0)
    out = img.astype(np.float32)
    out[..., 2] *= 1.06   # R up
    out[..., 0] *= 0.92   # B down
    out *= 0.86           # dim
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> None:
    if V3.exists():
        shutil.rmtree(V3)
    for sp in ("train", "val", "test"):
        for cls in ("normal", "plaque"):
            (V3 / sp / cls).mkdir(parents=True)

    # ---- base: symlink everything from v2 ----
    n = 0
    for sp in ("train", "val", "test"):
        for cls in ("normal", "plaque"):
            for f in (V2 / sp / cls).iterdir():
                (V3 / sp / cls / f.name).symlink_to(f.resolve())
                n += 1
    print("symlinked v2 files:", n)

    # ---- A. SANO val expansion ----
    df = pd.read_csv(SANO_CSV)
    tr_imgs = sorted(df[(df.subset == "train") & (df.status == "kept")].image.unique())
    pick = sorted(tr_imgs, key=lambda s: hashlib.md5(s.encode()).hexdigest())[:N_VAL_IMAGES]
    moved = 0
    for im in pick:
        for f in (V3 / "train/normal").glob(f"sano_{Path(im).stem}__t*.jpg"):
            dst = V3 / "val/normal" / f.name
            f.rename(dst)          # symlink move: stays a link to plaque_sano/train/*.jpg
            moved += 1
    print(f"A. moved {moved} SANO crops of {len(pick)} images train->val")
    (V3 / "sano_val_images.json").write_text(json.dumps(pick, indent=1))

    # ---- B. mined negatives ----
    mined = pd.read_csv(MINED / "mined_confirmed.csv") if (MINED / "mined_confirmed.csv").exists() else None
    n_mined = 0
    if mined is not None and len(mined):
        for r in mined.itertuples():
            src = MINED / "crops" / r.crop
            (V3 / "train/normal" / f"mined_{r.crop}").symlink_to(src.resolve())
            n_mined += 1
    print(f"B. mined negatives added: {n_mined}")

    # ---- C. SANO robustness variants ----
    n_var = 0
    for f in sorted((SANO_DIR / "train").glob("*.jpg")):
        stem = Path(f.name).stem
        # skip the 20 images that moved to val
        if any(stem == Path(im).stem.replace(" ", "_") for im in pick):
            continue
        img = cv2.imread(str(f))
        if img is None:
            continue
        var = robust_variant(img, hashlib.md5(f.name.encode()).hexdigest())
        out = V3 / "train/normal" / f"sano_{stem}__rob.jpg"
        if not out.exists():
            cv2.imwrite(str(out), var, [cv2.IMWRITE_JPEG_QUALITY, 95])
        n_var += 1
    print(f"C. SANO robustness variants: {n_var}")

    (V3 / "plaque_bin_v3.yaml").write_text(
        f"path: {_ROOT}/datasets/processed/plaque_bin_v3\n"
        "train: train\nval: val\n\nnc: 2\nnames:\n  0: normal\n  1: plaque\n")

    import collections
    cnt = collections.Counter()
    for sp in ("train", "val", "test"):
        for cls in ("normal", "plaque"):
            cnt[f"{sp}/{cls}"] = len(list((V3 / sp / cls).iterdir()))
    for k, v in sorted(cnt.items()):
        print(f"{k:16s} {v}")


if __name__ == "__main__":
    main()
