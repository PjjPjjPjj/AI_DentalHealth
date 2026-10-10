"""Extract images from Eka gingivitis parquet shards + audit.

Reads datasets/raw/eka_gingivitis_hires/parquet/*.parquet, extracts the Image
column (bytes) to images/<split>/<md5>.jpg, verifies md5 vs md5_image column,
and audits resolution / file size / split counts.

Outputs:
  datasets/raw/eka_gingivitis_hires/images/{train,validation,test}/
  datasets/raw/eka_gingivitis_hires/captions.csv
  outputs/reports/eka_gingivitis_audit.json
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

ROOT = Path(f"{_ROOT}/datasets/raw/eka_gingivitis_hires")
OUT = ROOT / "images"
COLS = ["Image", "md5_image", "Labels", "Dentist_1_caption",
        "Dentist_2_caption", "Dentist_3_caption"]


def main() -> None:
    shards = sorted(ROOT.glob("parquet/*.parquet"))
    assert shards, "no parquet shards found"
    meta = []
    widths, heights = [], []
    bad_md5 = 0
    for sh in shards:
        split = sh.stem.split("-")[0]
        (OUT / split).mkdir(parents=True, exist_ok=True)
        t = pq.read_table(sh, columns=COLS)
        df = t.to_pandas()
        for _, r in df.iterrows():
            data = r["Image"]
            if isinstance(data, dict):
                data = data.get("bytes")
            md5 = hashlib.md5(data).hexdigest()
            name = f"{split}_{md5}.jpg"
            p = OUT / split / name
            if not p.exists():
                p.write_bytes(data)
            if str(r["md5_image"]).strip().lower() not in ("", "none", "nan"):
                ok = str(r["md5_image"]).strip().lower() == md5
                bad_md5 += (not ok)
            meta.append({"split": split, "file": name, "md5": md5,
                         "labels": r["Labels"],
                         "cap1": r["Dentist_1_caption"],
                         "cap2": r["Dentist_2_caption"],
                         "cap3": r["Dentist_3_caption"]})
        print(f"{sh.name}: {len(df)} rows", flush=True)
    pd.DataFrame(meta).to_csv(ROOT / "captions.csv", index=False)

    # resolution audit (all files)
    recs = []
    for split in ("train", "validation", "test"):
        for p in sorted((OUT / split).glob("*.jpg")):
            img = cv2.imread(str(p))
            if img is None:
                recs.append({"split": split, "file": p.name, "w": 0, "h": 0,
                             "kb": p.stat().st_size // 1024, "ok": False})
                continue
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)
            recs.append({"split": split, "file": p.name, "w": w, "h": h,
                         "kb": p.stat().st_size // 1024, "ok": True})
    adf = pd.DataFrame(recs)
    adf.to_csv(ROOT / "resolution_audit.csv", index=False)

    long_side = np.maximum(adf.w, adf.h)
    report = {
        "n_total": int(len(adf)),
        "n_unreadable": int((~adf.ok).sum()),
        "md5_mismatch": int(bad_md5),
        "split_counts": adf.split.value_counts().to_dict(),
        "long_side_px": {"min": int(long_side.min()),
                         "p25": int(np.percentile(long_side, 25)),
                         "median": int(np.median(long_side)),
                         "p75": int(np.percentile(long_side, 75)),
                         "max": int(long_side.max())},
        "aspect_gt_1p5": int(((adf.w / adf.h.clip(lower=1)) > 1.5).sum()),
        "unique_md5": int(len(set(adf.file))),
        "total_gb": round(sum(f.stat().st_size for f in ROOT.rglob("*.jpg"))
                          / 1e9, 2),
    }
    (Path(f"{_ROOT}/outputs/reports/eka_gingivitis_audit.json")
     .write_text(json.dumps(report, indent=2)))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
