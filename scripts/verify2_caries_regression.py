"""Verify 2: caries regression through the pipeline model path.

Loads outputs/caries_c2/weights via src.inference.models.CariesModel
(conf=0.01 sweep -> frozen 0.55) and must reproduce eval_c2 test metrics:
P 0.8722 / R 0.9028 / F1 0.8872 (IoU 0.5 matching, imgsz 960).
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, f"{_ROOT}")
from src.inference.models import CariesModel  # noqa: E402

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

MAN = f"{_ROOT}/datasets/processed/caries_det_v2/split_manifest.csv"
FROZEN_CONF = 0.55


def main() -> None:
    model = CariesModel(weights=f"{_ROOT}/outputs/caries_c2/weights/best.pt",
                        conf=0.01, iou=0.6, imgsz=960)
    man = pd.read_csv(MAN)
    rows = man[man.split == "test"]
    tp_c, fp_c, n_gt = [], [], 0
    for r in rows.itertuples():
        img_h_w = None
        gts = []
        for line in Path(r.label).read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            cx, cy, w, h = map(float, p[1:5])
            gts.append((cx, cy, w, h))  # normalized
        n_gt += len(gts)
        res = model.predict(r.img)
        det = [(d["conf"], d["bbox"]) for d in res]
        det.sort(key=lambda t: -t[0])
        # need pixel dims for matching: reload via cv2-free path -> use res? wrapper
        # returns px coords; convert GT to px with image size from wrapper predict
        import cv2
        img = cv2.imread(r.img)
        H, W = img.shape[:2]
        gpx = [(cx * W, cy * H, w * W, h * H) for cx, cy, w, h in gts]
        used = set()
        for c, b in det:
            bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            bw, bh = b[2] - b[0], b[3] - b[1]
            best, bi = 0.0, None
            for gi, g in enumerate(gpx):
                if gi in used:
                    continue
                ix0 = max(bx - bw / 2, g[0] - g[2] / 2)
                iy0 = max(by - bh / 2, g[1] - g[3] / 2)
                ix1 = min(bx + bw / 2, g[0] + g[2] / 2)
                iy1 = min(by + bh / 2, g[1] + g[3] / 2)
                inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
                ua = bw * bh + g[2] * g[3] - inter
                io = inter / ua if ua else 0
                if io > best:
                    best, bi = io, gi
            if best >= 0.5 and bi is not None:
                used.add(bi)
                tp_c.append(float(c))
            else:
                fp_c.append(float(c))
    tp_c = np.array(tp_c); fp_c = np.array(fp_c)
    tp = int((tp_c >= FROZEN_CONF).sum())
    fp = int((fp_c >= FROZEN_CONF).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(n_gt, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    out = {"frozen_conf": FROZEN_CONF, "P": round(prec, 4), "R": round(rec, 4),
           "F1": round(f1, 4),
           "reference": {"P": 0.8722, "R": 0.9028, "F1": 0.8872},
           "pass": bool(abs(f1 - 0.8872) < 0.01)}
    Path(f"{_ROOT}/outputs/pipeline_v2_validation").mkdir(exist_ok=True, parents=True)
    Path(f"{_ROOT}/outputs/pipeline_v2_validation/caries_regression.json").write_text(
        json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
