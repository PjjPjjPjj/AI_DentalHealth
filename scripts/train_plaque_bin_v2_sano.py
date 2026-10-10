"""Fine-tune plaque_bin_v1 classifier with SANO healthy-tooth negatives (v2).

plaque_bin_v2 = plaque_bin_v1 (train/val/test unchanged, symlinked)
              + 3,522 SANO healthy tooth crops in train/normal (x1).

Rationale: v1 flags warm-toned / slightly blurred healthy teeth as plaque
(median P=0.986 on SANO eval crops) -> SANO negatives are the hard negatives
that fix the blur/color false-positive shortcut and lift specificity.

Start from v1 best.pt (low lr fine-tune) so the strong plaque prior is kept;
val unchanged -> checkpoint selection comparable with v1.
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
from ultralytics import YOLO

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

DATA = f"{_ROOT}/datasets/processed/plaque_bin_v2"
INIT = f"{_ROOT}/outputs/plaque_bin_v1/weights/best.pt"

m = YOLO(INIT)
m.train(
    data=DATA,
    imgsz=224,
    epochs=10,
    patience=4,
    batch=384,
    device=0,
    seed=42,
    amp=True,
    workers=4,
    cache=False,
    project="runs/classify",
    name="outputs/plaque_bin_v2_sano",
    exist_ok=True,
    lr0=0.002,
    lrf=0.05,
    cos_lr=True,
    warmup_epochs=1,
    hsv_h=0.005, hsv_s=0.3, hsv_v=0.2,
    degrees=0, translate=0.05, scale=0.1,
    fliplr=0.5, flipud=0.0,
    erasing=0.1,
    auto_augment=None,
)
print("TRAIN_DONE")
