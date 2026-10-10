"""Train plaque_bin_v3: v2 + mined braces hard negatives + SANO blur/warm augs
+ expanded val negatives (503 SANO crops).

Init from v2_sano best; lr halved (v2 diverged after epoch2 at lr0=0.002).
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

DATA = f"{_ROOT}/datasets/processed/plaque_bin_v3"
INIT = f"{_ROOT}/outputs/plaque_bin_v2_sano/weights/best.pt"

m = YOLO(INIT)
m.train(
    data=DATA,
    imgsz=224,
    epochs=8,
    patience=3,
    batch=384,
    device=0,
    seed=42,
    amp=True,
    workers=4,
    cache=False,
    project="runs/classify",
    name="outputs/plaque_bin_v3_mined",
    exist_ok=True,
    lr0=0.001,
    lrf=0.05,
    cos_lr=True,
    warmup_epochs=1,
    hsv_h=0.005, hsv_s=0.3, hsv_v=0.25,
    degrees=0, translate=0.05, scale=0.1,
    fliplr=0.5, flipud=0.0,
    erasing=0.1,
    auto_augment=None,
)
print("TRAIN_DONE")
