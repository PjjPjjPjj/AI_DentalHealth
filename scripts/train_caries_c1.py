"""Caries C1: YOLO11m @960, cos_lr, 120ep — resolution + capacity upgrade."""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from ultralytics import YOLO

model = YOLO("yolo11m.pt")
model.train(
    data="datasets/processed/caries_det_v2/data.yaml",
    project="outputs/caries_c1",
    name="train",
    exist_ok=True,
    task="detect",
    imgsz=960,
    epochs=120,
    patience=25,
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
    mosaic=1.0,
    close_mosaic=10,
    erasing=0.0,
    plots=True,
    verbose=True,
)
