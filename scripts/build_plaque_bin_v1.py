"""Build box-level binary plaque classification dataset (plaque_bin_v1).

Source: Mendeley "A new multi-modal dataset for Dental Plaque Diagno"
  raw: datasets/raw/Plaque/mendeley-dataset-materials_Part_{1,2}
  label line: <plaque 0/1> <cx> <cy> <w> <h> <tooth_id(1-32)>

Design (audit-driven):
  - official patient-level split from data_splits.csv (P1 col "status", P2 col "Recommend")
  - BOX-LEVEL instances: each annotated box = one sample with its own 0/1 flag.
    (Tooth-level aggregation is impossible: val/test contain no all-negative teeth.)
  - train keeps official augmented variants; val/test use base (clean) images only
  - train negatives oversampled x8 (to ~1:5 pos:neg)
  - training-time val (for ultralytics checkpoint selection) rebalanced to ~1:1 by
    duplicating the 96 val negative crops; final metrics are computed offline on the
    UNrebalanced official val/test via plaque_crops.csv (is_dup=0 rows only)
  - min crop side 8 px; undecodable images skipped

Output:
  datasets/processed/plaque_bin_v1/{train,val,test}/{normal,plaque}/*.jpg
  datasets/processed/plaque_bin_v1/plaque_crops.csv  (is_dup marks oversampled copies)
  datasets/processed/plaque_bin_v1/plaque_bin_v1.yaml
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

import csv
import json
import math
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

RAW = Path(f"{_ROOT}/datasets/raw/Plaque")
PARTS = [RAW / "mendeley-dataset-materials_Part_1", RAW / "mendeley-dataset-materials_Part_2"]
OUT = Path(f"{_ROOT}/datasets/processed/plaque_bin_v1")
MIN_SIDE_PX = 8
JPEG_Q = 95
NEG_DUP_TRAIN = 8        # x copies of each train negative (total 9x)
VAL_BAL_RATIO = 1.0      # val negatives duplicated to pos:neg = 1:1 for checkpoint selection

AUG_RE = re.compile(r"_(brightness-down|brightness-up|rotate-left-15|rotate-right-15|"
                    r"flip_horizontal|flip-horizontal|blur|light|dark)$")


def load_splits():
    split_of = {}
    for pi, part in enumerate(PARTS, 1):
        with open(part / "data_splits.csv", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                stem = Path(row["Image-Filename"]).stem.lower()
                status = (row.get("status") or row.get("Recommend") or "").strip().lower()
                split_of[(pi, stem)] = status
    return split_of


def load_fdi_map():
    js = json.loads((PARTS[0] / "metadata" / "teeth_position_mapping.json").read_text())
    return {c["id"]: c["name"] for c in js["categories"]}


def build_label_index(part: Path):
    idx = {}
    for d in (part / "data" / "labels").iterdir():
        if d.is_dir():
            idx[d.name.lower()] = {f.stem.lower(): f for f in d.iterdir() if f.suffix.lower() == ".txt"}
    return idx


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    for sp in ("train", "val", "test"):
        for cls in ("normal", "plaque"):
            (OUT / sp / cls).mkdir(parents=True)

    split_of = load_splits()
    fdi_map = load_fdi_map()
    stats = Counter()
    rows = []
    skipped_small = Counter()
    skipped_undec = []

    for pi, part in enumerate(PARTS, 1):
        lbl_idx = build_label_index(part)
        img_root = part / "data" / "images"
        for pdir in sorted(img_root.iterdir()):
            if not pdir.is_dir():
                continue
            files = lbl_idx.get(pdir.name.lower(), {})
            for img in sorted(pdir.iterdir()):
                if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                stem = img.stem.lower()
                sp = split_of.get((pi, stem))
                if sp is None:
                    stats["img_unassigned"] += 1
                    continue
                is_aug = bool(AUG_RE.search(stem))
                lf = files.get(img.stem.lower())
                if lf is None:
                    stats["no_label"] += 1
                    continue
                lines = [l.split() for l in lf.read_text().splitlines() if l.strip()]
                if not lines:
                    stats["empty_label"] += 1
                    continue
                # val/test: skip augmented variants unless they carry negative boxes
                # (negatives are scarce; extended negatives are flagged is_aug=1)
                has_neg = any(len(f) == 6 and int(f[0]) == 0 for f in lines)
                if sp in ("validation", "test") and is_aug and not has_neg:
                    stats[f"{sp}_aug_skipped"] += 1
                    continue
                im = cv2.imread(str(img))
                if im is None:
                    skipped_undec.append(f"part{pi}/{pdir.name}/{img.name}")
                    stats["undecodable"] += 1
                    continue
                H, W = im.shape[:2]
                dst_split = {"train": "train", "validation": "val", "test": "test"}[sp]
                for f in lines:
                    if len(f) != 6:
                        stats["malformed"] += 1
                        continue
                    p = int(f[0])
                    # val/test augmented variants contribute negatives only
                    if sp in ("validation", "test") and is_aug and p == 1:
                        stats[f"{sp}_aug_pos_skipped"] += 1
                        continue
                    cx, cy, w, h = map(float, f[1:5])
                    tid = int(f[5])
                    x1 = int(np.clip((cx - w / 2) * W, 0, W))
                    x2 = int(np.clip((cx + w / 2) * W, 0, W))
                    y1 = int(np.clip((cy - h / 2) * H, 0, H))
                    y2 = int(np.clip((cy + h / 2) * H, 0, H))
                    if x2 - x1 < MIN_SIDE_PX or y2 - y1 < MIN_SIDE_PX:
                        skipped_small[f"{dst_split}_{'plaque' if p else 'normal'}"] += 1
                        continue
                    crop = im[y1:y2, x1:x2]
                    cls_name = "plaque" if p == 1 else "normal"
                    name = f"{img.stem}__t{tid:02d}.jpg"
                    if (OUT / dst_split / cls_name / name).exists():  # same tooth twice in one image
                        name = f"{img.stem}__t{tid:02d}__b{len(rows)}.jpg"
                    cv2.imwrite(str(OUT / dst_split / cls_name / name), crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
                    rows.append({
                        "crop": f"{dst_split}/{cls_name}/{name}",
                        "split": dst_split, "patient": pdir.name.lower(),
                        "image_id": img.stem, "part": pi, "is_aug": int(is_aug),
                        "tooth_id": tid, "fdi": fdi_map.get(tid, str(tid)),
                        "label": p, "is_dup": 0,
                        "box_w": x2 - x1, "box_h": y2 - y1,
                    })
                    stats[f"{dst_split}_{cls_name}"] += 1

    # ---- oversampling ----
    def dup_files(cls_dir: Path, factor_extra: int, rows, stat_key: str):
        if factor_extra <= 0:
            return
        added = []
        for p in sorted(cls_dir.glob("*.jpg")):
            for i in range(1, factor_extra + 1):
                np_ = p.with_name(p.stem + f"__dup{i}.jpg")
                shutil.copy2(p, np_)
                base = rows_by_crop[p.relative_to(OUT).as_posix()]
                d = dict(base)
                d["crop"] = np_.relative_to(OUT).as_posix()
                d["is_dup"] = 1
                added.append(d)
        rows.extend(added)
        stats[stat_key] = len(added)

    rows_by_crop = {r["crop"]: r for r in rows}

    tr_pos, tr_neg = stats["train_plaque"], stats["train_normal"]
    va_pos, va_neg = stats["val_plaque"], stats["val_normal"]
    dup_files(OUT / "train" / "normal", NEG_DUP_TRAIN, rows, "train_neg_dup_files")
    if va_neg > 0:
        k = min(300, max(0, math.ceil(va_pos * VAL_BAL_RATIO / va_neg) - 1))
        dup_files(OUT / "val" / "normal", k, rows, "val_neg_dup_files")

    with open(OUT / "plaque_crops.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    (OUT / "plaque_bin_v1.yaml").write_text(
        f"path: {OUT}\ntrain: train\nval: val\ntest: test\nnc: 2\nnames:\n  0: normal\n  1: plaque\n")

    summary = {
        "raw_counts_per_split": {k: v for k, v in stats.items() if not k.endswith("_files")},
        "skipped_small": dict(skipped_small),
        "undecodable_images": skipped_undec,
        "train_final": {"plaque": tr_pos, "normal": tr_neg * (NEG_DUP_TRAIN + 1)},
        "val_final_selection": {"plaque": va_pos, "normal": va_neg + stats.get("val_neg_dup_files", 0)},
        "test_final": {"plaque": stats["test_plaque"], "normal": stats["test_normal"]},
        "note": "final eval uses csv rows with is_dup=0 only; val/test negatives include "
                "augmented-variant boxes flagged is_aug=1 (extended set)",
    }
    (OUT / "build_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
