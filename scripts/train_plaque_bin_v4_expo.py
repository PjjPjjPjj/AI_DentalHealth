"""v4_expo: exposure-robust fine-tune of v2_sano.

Inference-side normalization failed (sweep: train/infer mismatch, AUROC 0.90->0.66).
Correct fix: make the MODEL exposure-invariant via strong value-domain aug:
hsv_v 0.2 -> 0.5 (brightness extremes incl. over-exposure), hsv_s 0.3 -> 0.4.
Everything else identical to v2 recipe (data=plaque_bin_v2, lr halved again).
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

DATA = f"{_ROOT}/datasets/processed/plaque_bin_v2"
INIT = f"{_ROOT}/outputs/plaque_bin_v2_sano/weights/best.pt"

m = YOLO(INIT)
m.train(
    data=DATA,
    imgsz=224,
    epochs=6,
    patience=2,
    batch=384,
    device=0,
    seed=42,
    amp=True,
    workers=4,
    cache=False,
    project="runs/classify",
    name="outputs/plaque_bin_v4_expo",
    exist_ok=True,
    lr0=0.0005,
    lrf=0.05,
    cos_lr=True,
    warmup_epochs=1,
    hsv_h=0.005, hsv_s=0.4, hsv_v=0.5,
    degrees=0, translate=0.05, scale=0.1,
    fliplr=0.5, flipud=0.0,
    erasing=0.1,
    auto_augment=None,
)
print("TRAIN_DONE")
