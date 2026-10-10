"""Train YOLO11s-cls on plaque_bin_v1 (box-level binary: normal / plaque).

Conservative augmentation (dental color info matters): low hsv jitter, mild
geometric. Checkpoint selection on rebalanced val (dup negatives to ~1:1).
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from ultralytics import YOLO

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

DATA = f"{_ROOT}/datasets/processed/plaque_bin_v1"

m = YOLO("yolo11s-cls.pt")
m.train(
    data=DATA,
    imgsz=224,
    epochs=24,
    patience=6,
    batch=384,
    device=0,
    seed=42,
    amp=True,
    workers=4,
    cache=False,
    project="runs/classify",
    name="outputs/plaque_bin_v1",
    exist_ok=True,
    hsv_h=0.005, hsv_s=0.3, hsv_v=0.2,
    degrees=0, translate=0.05, scale=0.1,
    fliplr=0.5, flipud=0.0,
    erasing=0.1,
    auto_augment=None,
)
print("TRAIN_DONE")
