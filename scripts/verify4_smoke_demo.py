"""Verify 4: end-to-end smoke demo of the v2 pipeline (tooth+caries+plaque).

Runs the real pipeline on demo images, draws tooth contours + ids + plaque
prob, caries boxes, validates the JSON schema, saves outputs.
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

import cv2
import numpy as np

sys.path.insert(0, f"{_ROOT}")
from src.inference.pipeline import OralAIPipeline  # noqa: E402
from src.inference.schemas import validate_result  # noqa: E402

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

ROOT = Path(f"{_ROOT}")
OUT = ROOT / "outputs/pipeline_v2_validation"


def draw(img, result):
    vis = img.copy()
    rng = np.random.default_rng(42)
    colors = rng.integers(60, 255, (max(len(result["teeth"]), 1), 3))
    for t in result["teeth"]:
        i = t["tooth_id"] - 1
        c = tuple(int(v) for v in colors[i])
        x0, y0, x1, y1 = [int(v) for v in t["bbox"]]
        cv2.rectangle(vis, (x0, y0), (x1, y1), c, 1)
        pl = t["plaque"]
        tag = f"{i + 1} p{pl['score']:.2f}" + ("!" if pl.get("present") else "")
        cv2.putText(vis, tag, (x0, max(12, y0 - 3)), 0, 0.45, c, 1)
    for les in result.get("caries_lesions", []):
        x0, y0, x1, y1 = [int(v) for v in les["bbox"]]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(vis, f"caries {les['confidence']:.2f}", (x0, max(12, y1 + 14)),
                    0, 0.5, (0, 0, 255), 1)
    for les in result.get("unassigned_lesions", []):
        if les["type"] == "caries":
            x0, y0, x1, y1 = [int(v) for v in les["bbox"]]
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 140, 255), 1)
    return vis


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pipe = OralAIPipeline()  # real weights; calculus auto-disabled
    summary = {}
    real = [ROOT / "demo_images/intraoral_163.jpg", ROOT / "demo_images/intraoral_real.jpg"]
    for p in real:
        result = pipe.run(p)
        errs = validate_result(result)
        teeth = result["teeth"]
        summary[p.name] = {
            "schema_violations": errs,
            "n_teeth": len(teeth),
            "n_caries": len(result.get("caries_lesions", [])),
            "n_unassigned": len(result.get("unassigned_lesions", [])),
            "plaque_present": sum(1 for t in teeth if t["plaque"].get("present")),
            "quality_usable": result["quality"]["usable"],
        }
        (OUT / f"demo_{p.stem}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
        img = cv2.imread(str(p))
        cv2.imwrite(str(OUT / f"demo_{p.stem}.jpg"), draw(img, result))
        print(p.name, json.dumps(summary[p.name]), flush=True)
    (OUT / "smoke_summary.json").write_text(json.dumps(summary, indent=2))
    ok = all(not v["schema_violations"] and v["n_teeth"] > 0 for v in summary.values())
    print("SMOKE", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
