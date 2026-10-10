"""Train YOLO11m tooth detector on Eka work copies (single class)."""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from ultralytics import YOLO

model = YOLO("yolo11m.pt")
model.train(
    data="datasets/processed/eka_toothdet/toothdet.yaml",
    project="outputs/eka_toothdet",
    name="train",
    exist_ok=True,
    task="detect",
    imgsz=640,
    epochs=100,
    patience=20,
    batch=16,
    device=0,
    workers=8,
    amp=True,
    seed=42,
    deterministic=True,
    optimizer="auto",
    hsv_h=0.01,
    hsv_s=0.2,
    hsv_v=0.25,
    degrees=5.0,
    translate=0.05,
    scale=0.2,
    fliplr=0.5,
    erasing=0.0,
    plots=True,
    verbose=True,
)
metrics = model.val(data="datasets/processed/eka_toothdet/toothdet.yaml",
                    split="test", imgsz=640, device=0, verbose=True)
print("TEST:", metrics.results_dict)
