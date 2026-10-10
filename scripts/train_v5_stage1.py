"""V5 Stage1: fine-tune v4 on large-lesion-oversampled zoom data."""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from ultralytics import YOLO

model = YOLO("outputs/calculus_v4_stage1/weights/best.pt")
model.train(
    data="datasets/processed/calculus_v5_zoom/zoom_v5.yaml",
    project="outputs/calculus_v5_stage1",
    name="train",
    exist_ok=True,
    task="segment",
    imgsz=640,
    epochs=30,
    patience=10,
    batch=16,
    device=0,
    workers=8,
    amp=True,
    seed=42,
    deterministic=True,
    optimizer="adamw",
    lr0=0.0005,
    cos_lr=True,
    hsv_h=0.01,
    hsv_s=0.25,
    hsv_v=0.25,
    degrees=5.0,
    translate=0.05,
    scale=0.2,
    shear=2.0,
    fliplr=0.0,
    mosaic=0.5,
    close_mosaic=8,
    erasing=0.0,
    plots=True,
    verbose=True,
)
