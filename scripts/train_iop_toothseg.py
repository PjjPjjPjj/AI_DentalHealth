"""IOP-Compass tooth instance segmenter (single class, imgsz 896)."""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from ultralytics import YOLO

model = YOLO("yolo11m-seg.pt")
model.train(
    data="datasets/processed/iop_toothseg/toothseg.yaml",
    project="outputs/iop_toothseg",
    name="train",
    exist_ok=True,
    task="segment",
    imgsz=896,
    epochs=40,
    patience=8,
    batch=12,
    device=0,
    workers=8,
    amp=True,
    seed=42,
    deterministic=True,
    optimizer="adamw",
    lr0=0.002,
    cos_lr=True,
    hsv_h=0.01,
    hsv_s=0.2,
    hsv_v=0.25,
    degrees=5.0,
    translate=0.05,
    scale=0.2,
    fliplr=0.5,
    mosaic=0.5,
    close_mosaic=8,
    erasing=0.0,
    plots=True,
    verbose=True,
)
m = model.val(data="datasets/processed/iop_toothseg/toothseg.yaml",
              split="test", imgsz=896, device=0, verbose=False)
print("TEST:", m.results_dict)
