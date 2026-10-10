"""Patient-level split utilities.

Never split images directly when subject ids exist. Rule (plan 8.3):
    train_subjects ∩ val_subjects = ∅
    val_subjects ∩ test_subjects = ∅

Official patient-level splits are preferred and provided by the caller
(e.g. AIRC-LABDEN Data_splits.csv).
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import random
from collections import Counter

import pandas as pd


def patient_split(
    sample_ids: list[str],
    patient_ids: list[str],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> tuple[set[str], set[str], set[str]]:
    """Assign each patient to train/val/test, then derive sample splits.

    sample_ids[i] belongs to patient_ids[i].
    """
    assert abs(sum(ratios) - 1.0) < 1e-6
    patients = sorted(set(patient_ids))
    rng = random.Random(seed)
    rng.shuffle(patients)
    n = len(patients)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train_p = set(patients[:n_train])
    val_p = set(patients[n_train:n_train + n_val])
    test_p = set(patients[n_train + n_val:])

    def samples_for(pset: set[str]) -> set[str]:
        return {sid for sid, pid in zip(sample_ids, patient_ids) if pid in pset}

    return samples_for(train_p), samples_for(val_p), samples_for(test_p)


def assert_no_patient_leak(chosen: list[set[str]], patient_ids: list[str], sample_ids: list[str]) -> None:
    """Raise if any sample appears in more than one partition."""
    sample_to_part = Counter()
    for idx, part in enumerate(chosen):
        for sid in part:
            sample_to_part[(sid, idx)] += 1
    dup = [k for k in Counter(sample_ids).items() if k[1] > 1]
    if dup:
        raise ValueError(f"Duplicate samples across partitions: {dup}")


def write_split_manifest(rows: list[dict], out_path: str) -> None:
    pd.DataFrame(rows).to_csv(out_path, index=False)
