"""Build IOP-Compass tooth SEGMENTATION dataset (contours -> yolo-seg).

Patient-disjoint split by patient (seed 42): 900/50/50 patients.
Output: datasets/processed/iop_toothseg/{images,labels}/{train,val,test}
        + toothseg.yaml
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
import struct

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

ROOT = Path(f"{_ROOT}/datasets/raw/IOP_Compass/IOP-Compass")
OUT = Path(f"{_ROOT}/datasets/processed/iop_toothseg")
SEED = 42


def main() -> None:
    rng = np.random.default_rng(SEED)
    if OUT.exists():
        import shutil
        shutil.rmtree(OUT)
    for sp in ("train", "val", "test"):
        (OUT / "images" / sp).mkdir(parents=True)
        (OUT / "labels" / sp).mkdir(parents=True)
    patients = sorted({p.parent.name for p in ROOT.rglob("*.png")})
    rng.shuffle(patients)
    n = len(patients)
    n_tr, n_va = int(n * 0.9), int(n * 0.05)
    split_of = {p: ("train" if i < n_tr else "val" if i < n_tr + n_va else "test")
                for i, p in enumerate(patients)}
    rows = []
    skipped = 0
    for jf in sorted(ROOT.rglob("*.json")):
        patient = jf.parent.name
        sp = split_of[patient]
        img_name = jf.stem + ".png"
        img_path = jf.parent / img_name
        if not img_path.exists():
            skipped += 1
            continue
        o = json.loads(jf.read_text())
        # read true image size from PNG header
        with open(img_path, "rb") as f:
            head = f.read(24)
        W, H = struct.unpack(">II", head[16:24])
        lines = []
        for t in o.get("teeth", []):
            cont = t.get("contour")
            if not cont or len(cont) < 3:
                continue
            fdi = str(t.get("FDI_NUM"))
            pts = np.array(cont, float)
            pts[:, 0] = np.clip(pts[:, 0] / W, 0, 1)
            pts[:, 1] = np.clip(pts[:, 1] / H, 0, 1)
            # keep FDI as part of a per-image index file (single class for training)
            lines.append("0 " + " ".join(f"{x:.6f} {y:.6f}" for x, y in pts))
        if not lines:
            continue
        (OUT / "images" / sp / img_name).symlink_to(img_path)
        (OUT / "labels" / sp / (jf.stem + ".txt")).write_text("\n".join(lines))
        # sidecar: FDI per instance order for later attribution
        fdis = [str(t.get("FDI_NUM")) for t in o.get("teeth", [])
                if t.get("contour") and len(t["contour"]) >= 3]
        (OUT / "labels" / sp / (jf.stem + ".fdi.json")).write_text(
            json.dumps(fdis))
        rows.append({"patient": patient, "split": sp, "view": jf.stem,
                     "n_teeth": len(lines)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "split_manifest.csv", index=False)
    (OUT / "toothseg.yaml").write_text(
        f"path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n  0: tooth\n")
    print(json.dumps({"patients": n, "images": len(df),
                      "split": df.split.value_counts().to_dict(),
                      "teeth_total": int(df.n_teeth.sum()),
                      "skipped": skipped}, indent=2))


if __name__ == "__main__":
    main()
