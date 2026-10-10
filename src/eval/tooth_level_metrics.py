"""Tooth-level aggregation metrics (plan 33): sensitivity / specificity per tooth.

Given per-tooth lesion presence truth (from annotated data) and per-tooth
prediction (associated lesions), compute tooth-level sensitivity/specificity.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import pandas as pd


def tooth_level_sens_spec(
    truth: list[int],
    pred: list[int],
) -> dict:
    """truth/pred per tooth: 1 = lesion suspected, 0 = none."""
    tp = int(sum(1 for t, p in zip(truth, pred) if t == 1 and p == 1))
    fp = int(sum(1 for t, p in zip(truth, pred) if t == 0 and p == 1))
    fn = int(sum(1 for t, p in zip(truth, pred) if t == 1 and p == 0))
    tn = int(sum(1 for t, p in zip(truth, pred) if t == 0 and p == 0))
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "tooth_sensitivity": round(sens, 4),
        "tooth_specificity": round(spec, 4),
    }


def tooth_level_from_lesion_table(lesion_truth: pd.DataFrame, lesion_pred: pd.DataFrame,
                                  tooth_col: str = "tooth_id") -> pd.DataFrame:
    """Join lesion-level truth/pred per tooth and aggregate."""
    g = lesion_truth.groupby(tooth_col)["present"].max().reset_index()
    g2 = lesion_pred.groupby(tooth_col)["present"].max().reset_index()
    merged = g.merge(g2, on=tooth_col, suffixes=("_truth", "_pred"))
    return merged
