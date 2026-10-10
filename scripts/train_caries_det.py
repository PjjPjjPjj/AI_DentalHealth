"""Caries detection (caries6313 benchmarking subset, patient-disjoint v2 split)."""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from ultralytics import YOLO

model = YOLO("yolo11s.pt")
model.train(
    data="datasets/processed/caries_det_v2/data.yaml",
    project="outputs/caries_det_v2",
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
    lr0=0.01,
    hsv_h=0.01,
    hsv_s=0.25,
    hsv_v=0.25,
    degrees=5.0,
    translate=0.05,
    scale=0.2,
    fliplr=0.5,
    mosaic=1.0,
    close_mosaic=10,
    erasing=0.0,
    plots=True,
    verbose=True,
)
m = model.val(data="datasets/processed/caries_det_v2/data.yaml",
              split="test", imgsz=640, device=0, verbose=False)
print("TEST:", m.results_dict)
