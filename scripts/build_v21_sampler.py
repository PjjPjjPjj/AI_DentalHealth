"""Stage2 V2.1 training sampler (STEP2-4): gentle re-balance, no physical copies.

Epoch composition (1600 samples):
  65% normal positive (1040)   bucket weights small:medium:large = 1.3:1.1:1
                               (per-crop repeat cap 3; deficits filled from
                                the next-larger-sampled bucket within normal)
  10% hard positive  (160)     HP_conf:HP_loc = 2:1 (107:53), cap 3 / crop
  10% hard negative  (160)     moderate:hard:very-hard = 6:3:1 (96:48:16), cap 3
   5% easy negative  ( 80)     (confidence calibration)
  10% GT-centered    (160)     bucket weights 1.3:1.1:1 (cap 3)
Symlinks only. Config (augmentation etc.) written alongside."""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

CROPS = Path("datasets/processed/calculus_v2/crops")
V2 = Path("datasets/processed/calculus_v2")
TR = V2 / "v21_train"
N_TOTAL = 1600
CAP = 3


def primary(b: str) -> str:
    if pd.notna(b) and "small" in b:
        return "small"
    if pd.notna(b) and "medium" in b:
        return "medium"
    return "large"


def scale(sub: pd.DataFrame, weights: dict, quota: int) -> list:
    """Sample quota from sub by bucket weight w/ per-crop cap=3, no blanks."""
    out = []
    leftovers = {b: sub[sub.primary_bucket == b] for b in weights}
    order = sorted(weights, key=lambda b: -weights[b])
    for b in order:
        n = int(round(quota * weights[b] / sum(weights.values())))
        p = leftovers[b]
        take = min(n, len(p) * CAP)
        ids = p.crop_id.tolist()
        while take > 0:
            for cid in ids:
                if take <= 0:
                    break
                out.append(cid)
                take -= 1
    # global quota still missing -> refill from all buckets (cap 3)
    base = pd.concat([sub] * 3).groupby("crop_id").head(3)
    remain = quota - len(out)
    if remain > 0:
        pool = base[~base.crop_id.isin(set(out))].crop_id.tolist()
        while remain > 0 and pool:
            for cid in pool:
                if remain <= 0:
                    break
                out.append(cid)
                remain -= 1
    return out[:quota]


def main() -> None:
    m = pd.read_csv(CROPS / "crops_meta.csv")
    m["primary_bucket"] = m.gt_buckets.apply(primary)
    norm = m[m.type == "normal_positive"]
    hp = m[m.type == "hard_positive"]
    hn = m[m.type == "hard_negative"]
    easy = m[m.type == "easy_negative"]
    gtc = m[m.type == "gt_centered_positive"]

    w = {"small": 1.3, "medium": 1.1, "large": 1.0}
    norm_sel = scale(norm, w, 1040)
    hp_conf = scale(hp[hp.hp_split == "HP_conf"], {"x": 1}, 107)
    hp_loc = scale(hp[hp.hp_split == "HP_loc"], {"x": 1}, 53)
    hnm = scale(hn[(hn.max_s2_conf >= 0.20) & (hn.max_s2_conf < 0.50)], {"x": 1}, 96)
    hnh = scale(hn[(hn.max_s2_conf >= 0.50) & (hn.max_s2_conf < 0.70)], {"x": 1}, 48)
    hnv = scale(hn[hn.max_s2_conf >= 0.70], {"x": 1}, 16)
    easy_sel = easy.crop_id.sample(n=min(80, len(easy)), random_state=11).tolist()
    gtc_sel = scale(gtc, w, 160)

    final = norm_sel + hp_conf + hp_loc + hnm + hnh + hnv + easy_sel + gtc_sel
    assert len(final) == N_TOTAL, len(final)
    if TR.exists():
        shutil.rmtree(TR)
    (TR / "images").mkdir(parents=True)
    (TR / "labels").mkdir(parents=True)
    seen = {}
    for cid in final:
        k = seen.get(cid, 1)
        seen[cid] = k + 1
        name = cid if k == 1 else f"{cid}_r{k}"
        src_i = CROPS / "images" / f"{cid}.png"
        src_l = CROPS / "labels" / f"{cid}.txt"
        (TR / "images" / f"{name}.png").symlink_to(src_i.resolve())
        (TR / "labels" / f"{name}.txt").symlink_to(src_l.resolve())
    (TR / "composition.txt").write_text(
        f"normal={len(norm_sel)} hp_conf={len(hp_conf)} hp_loc={len(hp_loc)} "
        f"hn_mod={len(hnm)} hn_hard={len(hnh)} hn_vhard={len(hnv)} "
        f"easy={len(easy_sel)} gtc={len(gtc_sel)} total={len(final)}\n"
        f"bucket(primary) counts normal: "
        + str({b: norm[norm.primary_bucket == b].shape[0] for b in w}) + "\n"
        f"composition per bucket in final: "
        + str(pd.Series(final).apply(
            lambda cid: m[m.crop_id == cid].primary_bucket.iloc[0] if len(m[m.crop_id == cid]) else "x"
        ).value_counts().to_dict()))
    (V2 / "v21_config.yaml").write_text(
        "stage2_v21_config:\n"
        "  model: yolo11m-seg (V1 best init)\n"
        "  imgsz: 640\n  crop: 160 (bbox-center)\n"
        "  epochs: 15\n  lr0: 0.0004  (0.4 * V2 lr0=0.001)\n"
        "  loss: baseline (unchanged)\n"
        "  augment:\n"
        "    mosaic: 0.0\n    mixup: 0.0\n    copy_paste: 0.0\n"
        "    fliplr: 0.5\n    degrees: 8.0\n    translate: 0.05\n"
        "    scale: 0.1\n    shear: 0.0\n    hsv_h: 0.01\n"
        "    hsv_s: 0.2\n    hsv_v: 0.2\n    erasing: 0.0\n"
        "    blur/noise: none (ultralytics has none)\n"
        "  sampler:\n    65/10/10/5/10 normal/hp/hn/easy/gtc\n"
        "    bucket weights primary 1.3:1.1:1 (small:medium:large)\n"
        "    hard caps: per-crop <=3 per epoch\n")
    print("v21 sampler written total =", len(final))
    print((TR / "composition.txt").read_text())


if __name__ == "__main__":
    main()
