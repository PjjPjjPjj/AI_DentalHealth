"""Build caries6313 detection dataset (benchmarking subset, labeled only).

- Patient-disjoint re-split 70/15/15 (official split leaks: valid/test
  patients all appear in train).
- Merge cryptic classes 'd'(0)/'D'(1) -> single class 0 'caries'
  (semantics unverified even by the dataset's own prompt; both are caries
  annotations).
- Unlabeled 4039 images EXCLUDED (mix of negatives and unannotated positives
  from shared patients - unsafe as background).
Output: datasets/processed/caries_det_v2/{images,labels}/{train,val,test}
        + data.yaml + split_manifest.csv
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

ROOT = Path(f"{_ROOT}/datasets/raw/caries6313/Benchmarking Dataset")
OUT = Path(f"{_ROOT}/datasets/processed/caries_det_v2")
PAT = re.compile(r"anonymous_(\d+-\d+-\d+-\d+)_")
SEED = 42


def main() -> None:
    rng = np.random.default_rng(SEED)
    if OUT.exists():
        import shutil
        shutil.rmtree(OUT)
    for sp in ("train", "val", "test"):
        (OUT / "images" / sp).mkdir(parents=True)
        (OUT / "labels" / sp).mkdir(parents=True)

    # collect all labeled (img, label) with patient ids
    rows = []
    for sp in ("train", "valid", "test"):
        for lf in (ROOT / sp / "yolo").glob("*.txt"):
            img = None
            for cand in (ROOT / sp / "images").glob(lf.stem + ".*"):
                img = cand
                break
            if img is None:
                continue
            m = PAT.search(lf.stem)
            patient = m.group(1) if m else "solo_" + img.stem
            rows.append({"patient": patient, "split_src": sp,
                         "img": str(img), "label": str(lf)})
    df = pd.DataFrame(rows)
    patients = sorted(df.patient.unique())
    rng.shuffle(patients)
    n = len(patients)
    n_tr, n_va = int(n * 0.7), int(n * 0.15)
    split_of = {p: ("train" if i < n_tr else "val" if i < n_tr + n_va else "test")
                for i, p in enumerate(patients)}
    df["split"] = df.patient.map(split_of)

    n_boxes = 0
    for r in df.itertuples():
        img = Path(r.img)
        (OUT / "images" / r.split / img.name).symlink_to(img.resolve())
        out_lines = []
        for line in Path(r.label).read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            cx, cy, w, h = map(float, p[1:5])
            cx, cy = min(max(cx, 0), 1), min(max(cy, 0), 1)
            w, h = min(w, 1), min(h, 1)
            out_lines.append("0 " + " ".join(f"{v:.6f}" for v in (cx, cy, w, h)))
            n_boxes += 1
        (OUT / "labels" / r.split / (img.stem + ".txt")).write_text(
            "\n".join(out_lines))
    df.to_csv(OUT / "split_manifest.csv", index=False)
    (OUT / "data.yaml").write_text(
        f"path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n  0: caries\n")
    print(json.dumps({
        "images": len(df), "boxes": n_boxes,
        "split": df.split.value_counts().to_dict(),
        "patients": {"train": df[df.split=='train'].patient.nunique(),
                     "val": df[df.split=='val'].patient.nunique(),
                     "test": df[df.split=='test'].patient.nunique()}}, indent=2))


if __name__ == "__main__":
    main()
