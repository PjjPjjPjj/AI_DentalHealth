"""Fine-tune the P2-augmented v3b Stage1 (2-class)."""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from ultralytics import YOLO

model = YOLO("outputs/stage1_v3b_p2_clone.pt")
model.train(
    data="datasets/processed/calculus_seg_v3/calculus_v3b.yaml",
    project="outputs/calculus_v3b_p2",
    name="train",
    exist_ok=True,
    task="segment",
    imgsz=640,
    epochs=70,
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
    hsv_s=0.25,
    hsv_v=0.25,
    degrees=5.0,
    translate=0.05,
    scale=0.2,
    shear=2.0,
    fliplr=0.0,
    flipud=0.0,
    mosaic=0.5,
    mixup=0.0,
    copy_paste=0.0,
    close_mosaic=12,
    erasing=0.0,
    plots=True,
    verbose=True,
)
