"""V5b Stage1: fine-tune v5 stage1 with copy-paste large-blob synthetics."""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from ultralytics import YOLO

model = YOLO("outputs/calculus_v5_stage1/weights/best.pt")
model.train(
    data="datasets/processed/calculus_v5b_zoom/zoom_v5b.yaml",
    project="outputs/calculus_v5b_stage1",
    name="train",
    exist_ok=True,
    task="segment",
    imgsz=640,
    epochs=15,
    patience=6,
    batch=16,
    device=0,
    workers=8,
    amp=True,
    seed=42,
    deterministic=True,
    optimizer="adamw",
    lr0=0.0003,
    cos_lr=True,
    hsv_h=0.01,
    hsv_s=0.25,
    hsv_v=0.25,
    degrees=5.0,
    translate=0.05,
    scale=0.3,
    shear=2.0,
    fliplr=0.0,
    mosaic=0.5,
    close_mosaic=5,
    erasing=0.0,
    plots=True,
    verbose=True,
)
print("TRAIN_DONE")
