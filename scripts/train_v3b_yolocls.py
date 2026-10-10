# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from ultralytics import YOLO
model = YOLO("yolo11m-cls.pt")
model.train(data="datasets/processed/calculus_v3b_cls",
    project="outputs/calculus_v3b_yolocls", name="train", exist_ok=True,
    imgsz=448, epochs=30, patience=8, batch=32, device=0, workers=8, amp=True,
    seed=42, optimizer="auto", hsv_h=0.01, hsv_s=0.25, hsv_v=0.25,
    degrees=5.0, translate=0.05, scale=0.2, fliplr=0.5, erasing=0.0, verbose=True)
