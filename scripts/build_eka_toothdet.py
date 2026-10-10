"""Build YOLO tooth-DETECTION dataset from Eka work copies + tooth BBs.

Single class 'tooth'.  Fresh group-level split 70/15/15 (seed 42) — the
official Eka split leaks (4 md5 dup pairs cross it).
Output: datasets/processed/eka_toothdet/{images,labels}/{train,val,test}
        + toothdet.yaml + split_manifest.csv
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

import numpy as np
import pandas as pd

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

WORK = Path(f"{_ROOT}/datasets/processed/eka_work")
OUT = Path(f"{_ROOT}/datasets/processed/eka_toothdet")
SEED = 42


def main() -> None:
    rng = np.random.default_rng(SEED)
    tt = pd.read_csv(WORK / "tooth_table.csv")
    if OUT.exists():
        import shutil
        shutil.rmtree(OUT)
    for sp in ("train", "val", "test"):
        (OUT / "images" / sp).mkdir(parents=True)
        (OUT / "labels" / sp).mkdir(parents=True)
    # group-level split (groups==unique md5 here, but keep group logic for safety)
    groups = tt.md5.unique()
    rng.shuffle(groups)
    n = len(groups)
    n_tr, n_va = int(n * 0.7), int(n * 0.15)
    split_of = {g: ("train" if i < n_tr else "val" if i < n_tr + n_va else "test")
                for i, g in enumerate(groups)}
    rows = []
    n_boxes = 0
    for r in tt.itertuples():
        sp = split_of[r.md5]
        try:
            teeth = json.loads(r.teeth_json) if isinstance(r.teeth_json, str) else []
        except Exception:
            teeth = []
        if not teeth:
            continue
        img_name = Path(r.work_path).name
        (OUT / "images" / sp / img_name).symlink_to(r.work_path)
        lines = []
        for t in teeth:
            bx = t.get("Bounding_Box_Coordinates")
            if not bx or len(bx) != 4:
                continue
            x, y, w, h = bx
            # clip to [0,1]
            x0, y0 = max(0.0, x), max(0.0, y)
            x1 = min(1.0, x + w); y1 = min(1.0, y + h)
            if x1 - x0 < 0.01 or y1 - y0 < 0.01:
                continue
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            lines.append(f"0 {cx:.6f} {cy:.6f} {x1 - x0:.6f} {y1 - y0:.6f}")
            n_boxes += 1
        (OUT / "labels" / sp / (Path(img_name).stem + ".txt")).write_text(
            "\n".join(lines))
        rows.append({"md5": r.md5, "split": sp, "n_teeth": len(lines),
                     "work_path": r.work_path})
    pd.DataFrame(rows).to_csv(OUT / "split_manifest.csv", index=False)
    (OUT / "toothdet.yaml").write_text(
        f"path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n  0: tooth\n")
    df = pd.DataFrame(rows)
    print(json.dumps({"images": int(len(df)),
                      "boxes": n_boxes,
                      "split": df.split.value_counts().to_dict(),
                      "teeth_per_img_median": float(df.n_teeth.median())},
                     indent=2))


if __name__ == "__main__":
    main()
