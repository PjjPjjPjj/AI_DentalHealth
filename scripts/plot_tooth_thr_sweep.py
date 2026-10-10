"""Materialize the tooth-level threshold sweep behind the frozen 0.39.

Rule (frozen in eval_plaque_toothlevel_v2 step3): on calibrated scores,
  constraint  Sens(val positive teeth) >= 0.95
  objective   max Spec(SANO-refit 517 healthy teeth)
Outputs outputs/plaque_toothlevel_v2/threshold_sweep.{csv,png}
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, f"{_ROOT}")
from scripts.eval_plaque_toothlevel_v2 import platt, fit_platt  # noqa: E402

ROOT = Path(f"{_ROOT}")
OUT = ROOT / "outputs/plaque_toothlevel_v2"
A, B = 0.6059, 0.8841

big = pd.read_csv(OUT / "tooth_records.csv")
pv = big[(big.subset.isna()) & (big.split == "val") & (big.matched == 1)]
rf = big[big.subset == "train"]
cv = platt(pv.p_full.values, (A, B))
cs = platt(rf.p_full.values, (A, B))
grid = np.arange(0.01, 1.0, 0.005)
rows = []
for t in grid:
    rows.append({"thr": round(float(t), 3),
                 "val_sens": float((cv >= t).mean()),
                 "sano_refit_spec": float((cs < t).mean()),
                 "val_fp": int((cv < t).sum()), "sano_fp": int((cs >= t).sum())})
df = pd.DataFrame(rows)
df.to_csv(OUT / "threshold_sweep.csv", index=False)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(df.thr, df.val_sens, label="Sens  (val, 448 positive teeth)", lw=2)
ax.plot(df.thr, df.sano_refit_spec, label="Spec  (SANO-refit, 517 healthy)", lw=2)
ax.axvline(0.39, color="tab:red", ls="--", lw=1.2)
ax.annotate("0.39  frozen\n(sens>=0.95 -> max spec)", (0.39, 0.5), color="tab:red",
            fontsize=9, ha="left", va="center", xytext=(0.45, 0.55),
            textcoords="data")
ax.axvline(0.746, color="gray", ls=":", lw=1.2)
ax.annotate("0.746 deploy (spec-first)", (0.746, 0.06), color="gray", fontsize=9)
ax.set_xlabel("threshold (tooth-level calibrated scale)")
ax.set_ylabel("rate")
ax.set_ylim(-0.03, 1.05)
ax.grid(alpha=0.3)
ax.legend(loc="center right")
ax.set_title("tooth-level threshold sweep (grid 0.005)")
fig.tight_layout()
fig.savefig(OUT / "threshold_sweep.png", dpi=130)
sel = df[(df.val_sens >= 0.95)]
best = sel.loc[sel.sano_refit_spec.idxmax()]
print(f"frozen thr={best.thr}  val_sens={best.val_sens:.4f}  spec={best.sano_refit_spec:.4f}")
print(df[(df.thr >= 0.30) & (df.thr <= 0.48)].iloc[::4].to_string(index=False))
