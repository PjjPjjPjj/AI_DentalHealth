"""Export approved queue masks into YOLO-seg label format.

Reads outputs/annotation_queue/<image>/*.json after a human reviews the
geometry and assigns the medical class. Writes label txt (class from decision,
minimum class set from a pass-in class map).
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import json
from pathlib import Path

import cv2

from src.utils.paths import ANNOTATION_QUEUE_DIR


def export_queue_to_yolo(queue_dir: Path | None = None,
                         out_labels: str = "datasets/processed/manual_annotations/labels",
                         class_map: dict[str, int] | None = None,
                         image_dir: str = "datasets/raw/manual") -> list[Path]:
    """Export approved entries. Approved = status 'approved' and bbox present.

    Returns list of written label files.
    """
    class_map = class_map or {"tooth": 0, "calculus": 1, "caries": 2}
    queue_dir = queue_dir or ANNOTATION_QUEUE_DIR
    written = []
    for qf in queue_dir.glob("*/queue.json"):
        image_id = qf.parent.name
        img_path = Path(image_dir) / f"{image_id}.jpg"
        if not img_path.exists():
            img_path = Path(image_dir) / f"{image_id}.png"
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        entries = json.loads(qf.read_text())
        lines = []
        for e in entries:
            if e.get("status") != "approved":
                continue
            cls = class_map.get((e.get("approved_class") or e.get("class_suggestion") or ""), 0)
            x1, y1, x2, y2 = e["bbox"]
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = abs(x2 - x1) / w, abs(y2 - y1) / h
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        out_dir = Path(out_labels)
        out_dir.mkdir(parents=True, exist_ok=True)
        lp = out_dir / f"{image_id}.txt"
        lp.write_text("\n".join(lines))
        written.append(lp)
    return written
