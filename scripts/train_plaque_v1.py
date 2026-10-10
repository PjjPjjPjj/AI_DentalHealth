"""Plaque V1: YOLO11m-seg 2-class (plaque vs other_lesion), honest v3 split."""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from ultralytics import YOLO

model = YOLO("yolo11m-seg.pt")
model.train(
    data="datasets/processed/plaque_v1/plaque_v1.yaml",
    project="outputs/plaque_v1",
    name="train",
    exist_ok=True,
    task="segment",
    imgsz=640,
    epochs=80,
    patience=15,
    batch=16,
    device=0,
    workers=8,
    amp=True,
    seed=42,
    deterministic=True,
    optimizer="adamw",
    lr0=0.001,
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
    mixup=0.0,
    copy_paste=0.0,
    close_mosaic=10,
    erasing=0.0,
    plots=True,
    verbose=True,
)
