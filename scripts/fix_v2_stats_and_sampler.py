"""V2 dataset: corrected statistics + sampler construction.

1) Corrected tally:
   Stage1 proposal positive = normal + hard positive
   Stage1 proposal negative = hard_negative + easy_negative
   GT-centered positive = EXTRA data (not part of Stage1 proposals).
2) hard_positive split:
   HP_loc : best instance IoU < 0.5
   HP_conf: best IoU >= 0.5 but max conf < 0.40
3) primary_bucket per positive crop:
   contains small instance -> small ; elif medium -> medium ; else large.
4) Sampler (fixed per-epoch composition, symlinks only):
   epoch 2000 samples: 800 normal + 400 hard + 600 hard-neg + 200 gtc;
   normal/gtc sampled ~2:1:1 by primary_bucket;
   hard: each crop at most 4 copies (`epoch repeat cap`), small 2x weight;
   if hard pool runs dry, deficit filled from normal_positive.
Outputs: v2_stats.md, v2_train/{images,labels}, v2_data.yaml
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

import csv
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

CROPS = Path("datasets/processed/calculus_v2/crops")
V2 = Path("datasets/processed/calculus_v2")
TR = V2 / "v2_train"
BUCKETS = ("small", "medium", "large")


def primary_bucket(blob: str) -> str:
    if pd.notna(blob) and "small" in blob:
        return "small"
    if pd.notna(blob) and "medium" in blob:
        return "medium"
    return "large"


def main() -> None:
    m = pd.read_csv(CROPS / "crops_meta.csv")
    m["hp_split"] = np.where(
        m.type == "hard_positive",
        np.where(m.best_inst_iou < 0.5, "HP_loc", "HP_conf"), "")
    m["primary_bucket"] = m.gt_buckets.apply(primary_bucket)
    m.to_csv(CROPS / "crops_meta.csv", index=False)

    prop_pos = m[m.type.isin(["normal_positive", "hard_positive"])]
    prop_neg = m[m.type.isin(["hard_negative", "easy_negative"])]
    gtc = m[m.type == "gt_centered_positive"]
    lines = ["# Stage2 V2 dataset: corrected final statistics\n",
             f"**Stage1 proposal total: {len(prop_pos) + len(prop_neg)}"
             f" (positive {len(prop_pos)}, negative {len(prop_neg)})**\n",
             f"GT-centered positives (extra): {len(gtc)}\n"]
    for t in ["normal_positive", "hard_positive", "hard_negative",
              "easy_negative", "gt_centered_positive"]:
        sub = m[m.type == t]
        lines.append(f"- {t}: {len(sub)}")
    lines.append("")
    lines.append("## hard_positive split")
    hp = m[m.type == "hard_positive"]
    lines.append(f"- HP_loc (best IoU<0.5): {(hp.hp_split == 'HP_loc').sum()}")
    lines.append(f"- HP_conf (IoU>=0.5 but conf<0.40): {(hp.hp_split == 'HP_conf').sum()}")
    lines.append("- HP_loc bucket dist: "
                 + str(dict(hp[hp.hp_split == 'HP_loc'].primary_bucket.value_counts())))
    lines.append("- HP_conf bucket dist: "
                 + str(dict(hp[hp.hp_split == 'HP_conf'].primary_bucket.value_counts())))
    lines.append("")
    lines.append("## positive primary_bucket distribution (contains-small-first rule)")
    for t in ["normal_positive", "hard_positive", "gt_centered_positive"]:
        sub = m[m.type == t]
        lines.append(f"- {t}: " + str(dict(sub.primary_bucket.value_counts())))
    (V2 / "v2_stats.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---------------- sampler ----------------
    if TR.exists():
        shutil.rmtree(TR)
    (TR / "images").mkdir(parents=True)
    (TR / "labels").mkdir(parents=True)

    def take(sub: pd.DataFrame, n: int, rng=42):
        rng = np.random.default_rng(rng)
        if len(sub) <= n:
            return sub, 0
        sel = sub.sample(n=n, random_state=rng.integers(1 << 30))
        return sel, 0

    def take_ratio(sub: pd.DataFrame, n_small: int) -> pd.DataFrame:
        """2:1:1 by primary_bucket."""
        pieces = []
        for b, cnt in zip(BUCKETS, [n_small, n_small / 2, n_small / 2]):
            cnt = min(int(round(cnt)), len(sub[sub.primary_bucket == b]))
            pieces.append(sub[sub.primary_bucket == b].sample(n=cnt, random_state=6))
        return pd.concat(pieces) if pieces else sub.iloc[0:0]

    norm = m[m.type == "normal_positive"]
    hards = m[m.type == "hard_positive"]
    negs = m[m.type == "hard_negative"]
    ge = m[m.type == "easy_negative"]
    gtc = m[m.type == "gt_centered_positive"]

    # hard: cap each crop at 4 copies, small 2x weight
    har = hards.copy()
    har["w"] = np.where(har.primary_bucket == "small", 2, 1)
    pool = []
    for _, row in har.iterrows():
        for _ in range(min(4, int(row["w"] * 2))):
            pool.append(row.crop_id)
    rng = np.random.default_rng(42)
    rng.shuffle(pool)
    hard_sel = pool[:400]
    gap = 400 - len(hard_sel)
    # deficit -> normal
    norm_gap = norm[~norm.crop_id.isin(hard_sel)].sample(
        n=gap, random_state=1).crop_id.tolist() if gap else []

    def fill(sub: pd.DataFrame, base_ids: set, target: int, seed: int):
        need = target - len(base_ids)
        rem = sub[~sub.crop_id.isin(base_ids)]
        extra = rem.sample(n=min(need, len(rem)), random_state=seed)
        return list(base_ids) + list(extra.crop_id) + (
            list(rem.sample(n=need - len(extra), replace=True,
                            random_state=seed).crop_id) if need > len(extra) else [])

    nb = take_ratio(norm, 400)
    normal_sel = fill(norm, set(nb.crop_id) | set(norm_gap), 800, 7)
    gtb = take_ratio(gtc, 100)
    gtc_sel = fill(gtc, set(gtb.crop_id), 200, 8)
    neg_sel = negs.sample(n=min(600, len(negs)), random_state=2).crop_id.tolist()
    if len(neg_sel) < 600:
        need = 600 - len(neg_sel)
        extra_e = ge.sample(n=min(need, len(ge)), random_state=3).crop_id.tolist()
        ready = neg_sel + extra_e
        if len(ready) < 600:
            ready += negs.sample(n=600 - len(ready), replace=True,
                                 random_state=4).crop_id.tolist()
    else:
        ready = neg_sel

    final = (list(normal_sel) + hard_sel + ready + list(gtc_sel))
    with open(TR / "sampler_log.txt", "w") as f:
        f.write(f"normal={len(normal_sel)} hard={len(hard_sel)} "
                f"hard_gap_filled_from_normal={gap} neg={len(ready)} "
                f"gtc={len(gtc_sel)}\n")
    print(f"sampler: normal={len(normal_sel)} hard={len(hard_sel)}(gap {gap}) "
          f"neg={len(ready)} gtc={len(gtc_sel)} total={len(final)}")

    seen: dict = {}
    for cid in final:
        k = seen.get(cid, 1)
        seen[cid] = k + 1
        name = cid if k == 1 else f"{cid}_r{k}"
        src_i = CROPS / "images" / f"{cid}.png"
        src_l = CROPS / "labels" / f"{cid}.txt"
        os.symlink(os.path.abspath(src_i), TR / "images" / f"{name}.png")
        os.symlink(os.path.abspath(src_l), TR / "labels" / f"{name}.txt")
    (V2 / "v2_data.yaml").write_text(
        f"path: {V2.resolve()}\ntrain: v2_train/images\n"
        "val: v2_train/images\nnames:\n  0: calculus\n")
    print("sampler dir + data yaml ready")


if __name__ == "__main__":
    main()
