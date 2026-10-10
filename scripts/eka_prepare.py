"""Eka hi-res dataset preparation (steps 1-2 of the pseudo-label plan).

1) Work copies: unique-md5 images decoded from raw (7286px) and resized to
   long side 1536 (closest to S1 v3c training scale: teeth ~180-320px).
2) Dedup: md5 + dHash(hamming<=8) union-find groups; one representative per
   group (the raw set contains 14 md5 dup pairs, 4 across official splits).
3) Tooth structure table: parse labels JSON (per-tooth BB + gingivitis
   Class_ID + Maxilla/Mandible BB) and best-effort FDI positions from captions.

Outputs:
  datasets/processed/eka_work/images/{split}_{md5[:12]}.jpg
  datasets/processed/eka_work/manifest.csv
  datasets/processed/eka_work/tooth_table.csv
  outputs/reports/eka_prepare_report.json
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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

RAW = Path(f"{_ROOT}/datasets/raw/eka_gingivitis_hires")
WORK = Path(f"{_ROOT}/datasets/processed/eka_work")
LONG = 1536
DHASH_BITS = 256
DHASH_THR = 8


def dhash(p: Path) -> np.ndarray | None:
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (17, 16))
    return (img[:, 1:] > img[:, :-1]).astype(np.uint8).ravel()


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def make_work(args):
    src, dst = args
    if dst.exists():
        return (str(dst), *cv2.imread(str(dst), cv2.IMREAD_REDUCED_COLOR_4).shape[:2])
    img = cv2.imread(str(src))
    if img is None:
        return (str(dst), 0, 0)
    h, w = img.shape[:2]
    s = LONG / max(h, w)
    img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                     interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return (str(dst), img.shape[0], img.shape[1])


def parse_fdi_from_captions(cap: str) -> dict:
    """Best-effort: {'healthy': [13,12], 'gin1': [11], ...} per jaw section."""
    out = {}
    if not isinstance(cap, str):
        return out
    for level in re.finditer(
            r"(non-inflamed|gingivitis\s*(\d+))[^;:]*?position\s*([\d,\s]+)", cap, re.I):
        key = "healthy" if level.group(1).lower().startswith("non") \
            else f"gin{level.group(2)}"
        teeth = [int(t) for t in re.findall(r"\d{1,2}", level.group(3)) if 11 <= int(t) <= 48]
        out.setdefault(key, []).extend(teeth)
    return out


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    cap = pd.read_csv(RAW / "captions.csv")
    cap = cap.drop_duplicates(subset=["md5"]).reset_index(drop=True)
    print(f"unique md5 rows: {len(cap)}")

    # file map: images/<split>_<md5>.jpg
    fmap = {}
    for p in RAW.glob("images/*/*.jpg"):
        md5 = p.stem.split("_", 1)[1]
        fmap.setdefault(md5, p)
    cap = cap[cap.md5.isin(fmap)].reset_index(drop=True)
    print(f"rows with file: {len(cap)}")

    # dHash near-dup groups
    hashes, ok = {}, {}
    for i, r in cap.iterrows():
        h = dhash(fmap[r.md5])
        ok[i] = h is not None
        if h is not None:
            hashes[i] = h
    uf = UF(len(cap))
    POP = np.array([bin(b).count("1") for b in range(256)], np.uint8)
    idxs = [i for i in cap.index if ok[i]]
    packed = {i: np.packbits(hashes[i]) for i in idxs}
    ids = list(idxs)
    near = 0
    for a in range(0, len(ids), 128):
        chunk = ids[a:a + 128]
        A = np.stack([packed[i] for i in chunk])
        B = np.stack([packed[i] for i in ids])
        d = POP[np.bitwise_xor(A[:, None, :], B[None, :, :])].sum(axis=2)
        for bi, bj in zip(*np.where((d <= DHASH_THR) & (d > 0))):
            uf.union(chunk[bi], int(ids[bj]))
            near += 1
    cap["group_id"] = [uf.find(i) for i in cap.index]
    print(f"near pairs merged: {near}, groups: {cap.group_id.nunique()}")

    # keep one representative per group
    cap = cap.sort_values("group_id").drop_duplicates("group_id").reset_index(drop=True)
    print(f"representatives: {len(cap)}")

    # work copies (parallel)
    tasks = [(fmap[r.md5], WORK / "images" / f"{r.split}_{r.md5[:12]}.jpg")
             for r in cap.itertuples()]
    (WORK / "images").mkdir(exist_ok=True)
    with ProcessPoolExecutor(8) as ex:
        sizes = list(ex.map(make_work, tasks))
    size_map = {Path(p).name: (h, w) for p, h, w in sizes}
    cap["work_path"] = [str(WORK / "images" / f"{r.split}_{r.md5[:12]}.jpg")
                        for r in cap.itertuples()]

    # tooth structure table
    rows = []
    for r in cap.itertuples():
        try:
            lab = json.loads(r.labels)
        except Exception:
            lab = {}
        teeth = lab.get("Teeth", [])
        rows.append({
            "md5": r.md5, "work_path": r.work_path, "orig_split": r.split,
            "n_teeth_bb": len(teeth),
            "gin_class_ids": ",".join(str(t.get("Class_ID")) for t in teeth),
            "maxilla_bb": json.dumps(lab.get("Maxilla_BB_Coords")),
            "mandible_bb": json.dumps(lab.get("Mandible_BB_Coords")),
            "teeth_json": json.dumps(teeth),
            "fdi_parse": json.dumps(parse_fdi_from_captions(r.cap1)),
            "work_h": size_map.get(Path(r.work_path).name, (0, 0))[0],
            "work_w": size_map.get(Path(r.work_path).name, (0, 0))[1],
        })
    tt = pd.DataFrame(rows)
    tt.to_csv(WORK / "tooth_table.csv", index=False)
    cap_out = cap[["md5", "group_id", "split", "work_path"]].copy()
    cap_out.to_csv(WORK / "manifest.csv", index=False)

    rep = {
        "unique_md5": int(len(cap)), "groups": int(cap.group_id.nunique()),
        "near_pairs_merged": near,
        "work_long_side": LONG,
        "work_size_median": [int(tt.work_h.median()), int(tt.work_w.median())],
        "n_teeth_bb_total": int(tt.n_teeth_bb.sum()),
        "n_teeth_bb_median": float(tt.n_teeth_bb.median()),
        "fdi_parse_success": int((tt.fdi_parse != "{}").sum()),
        "unreadable_work": int(sum(1 for _, h, w in sizes if h == 0)),
    }
    (Path(f"{_ROOT}/outputs/reports/eka_prepare_report.json")
     .write_text(json.dumps(rep, indent=2)))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
