"""Lesi_Intraoral (Roboflow YOLOv11-seg export) full audit.

4 classes: Calculus / Caries / Gingivitis / Ulcer, polygon labels.
Checks: label stats per class/split, resolution, internal dups (md5+dHash),
CROSS-DATASET LEAKAGE vs Oral-Mamba v3 (dHash, re-encode invariant),
thumbnails.
Outputs: outputs/reports/lesi_intraoral_audit.json
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
import struct
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

ROOT = Path(f"{_ROOT}/datasets/raw/Lesi_Intraoral")
OM = Path(f"{_ROOT}/datasets/processed/calculus_seg_v3")
REPORT = Path(f"{_ROOT}/outputs/reports/lesi_intraoral_audit.json")
NAMES = ["Calculus", "Caries", "Gingivitis", "Ulcer"]


def md5_of(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash_path(args):
    p, red = args
    img = cv2.imread(str(p), red)
    if img is None:
        return p.name, b"", (0, 0)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (17, 16))
    h = np.packbits((g[:, 1:] > g[:, :-1]).astype(np.uint8).ravel()).tobytes()
    return p.name, h, img.shape[:2]


def main() -> None:
    # ---- label stats ----
    stats = {}
    for sp in ("train", "valid", "test"):
        imgs = sorted((ROOT / sp / "images").glob("*"))
        lbls = sorted((ROOT / sp / "labels").glob("*.txt"))
        cls_c, inst_per_img = Counter(), []
        calc_imgs = 0
        empty = 0
        for lf in lbls:
            cc = Counter()
            for line in lf.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cc[int(parts[0])] += 1
            if cc:
                inst_per_img.append(sum(cc.values()))
                cls_c.update(cc)
                if cc.get(0, 0):
                    calc_imgs += 1
            else:
                empty += 1
        stats[sp] = {"images": len(imgs), "labels": len(lbls), "empty_labels": empty,
                     "class_instances": {NAMES[k]: v for k, v in sorted(cls_c.items())},
                     "calculus_images": calc_imgs,
                     "inst_per_img_median": float(np.median(inst_per_img)) if inst_per_img else 0}
    print(json.dumps(stats, indent=1), flush=True)

    # ---- resolution + md5 (parallel) ----
    all_imgs = {sp: sorted((ROOT / sp / "images").glob("*"))
                for sp in ("train", "valid", "test")}
    flat = [(p, cv2.IMREAD_REDUCED_COLOR_2) for sp in all_imgs for p in all_imgs[sp]]
    with ProcessPoolExecutor(10) as ex:
        results = list(ex.map(dhash_path, flat, chunksize=10))
    sizes = {r[0]: r[2] for r in results}
    ws = np.array([sizes[p.name][1] for sp in all_imgs for p in all_imgs[sp]])
    hs = np.array([sizes[p.name][0] for sp in all_imgs for p in all_imgs[sp]])
    long_side = np.maximum(ws, hs)

    # ---- internal dups ----
    md5s = [md5_of(p) for sp in all_imgs for p in all_imgs[sp]]
    mc = Counter(md5s)
    md5_dups = {k: v for k, v in mc.items() if v > 1}
    from collections import defaultdict
    m2f = defaultdict(list)
    for sp in all_imgs:
        for p in all_imgs[sp]:
            m2f[md5_of(p)].append(f"{sp}/{p.name}")
    packed = np.stack([np.frombuffer(r[1], np.uint8) for r in results])
    POP = np.array([bin(b).count("1") for b in range(256)], np.uint8)
    names_all = [r[0] for r in results]
    uf_pairs = []
    for a in range(0, len(packed), 256):
        ch = packed[a:a + 256]
        d = POP[np.bitwise_xor(ch[:, None, :], packed[None, :, :])].sum(axis=2)
        for bi, bj in zip(*np.where((d <= 5) & (d > 0))):
            uf_pairs.append((names_all[a + bi], names_all[int(a + bj)]))

    # ---- CROSS-DATASET LEAKAGE vs Oral-Mamba ----
    om_man = pd_read_csv(OM / "split_manifest_v3.csv")
    om_files = [(r.image_path, r.split, r.image_id) for r in om_man.itertuples()]
    with ProcessPoolExecutor(12) as ex:
        om_h = list(ex.map(dhash_path, [(Path(p), cv2.IMREAD_REDUCED_COLOR_8)
                                        for p, _, _ in om_files], chunksize=20))
    om_names = [o[0] for o in om_h]
    om_packed = np.stack([np.frombuffer(o[1], np.uint8) for o in om_h])
    lesi_packed = np.stack([np.frombuffer(r[1], np.uint8) for r in results])
    cross = []
    POPC = np.array([bin(b).count("1") for b in range(256)], np.uint8)
    for a in range(0, len(lesi_packed), 64):
        ch = lesi_packed[a:a + 64]
        d = POPC[np.bitwise_xor(ch[:, None, :], om_packed[None, :, :])].sum(axis=2)
        for bi, bj in zip(*np.where(d <= 5)):
            cross.append({"lesi": names_all[a + bi],
                          "om": om_names[int(bj)],
                          "om_split": om_files[int(bj)][1],
                          "hamming": int(d[bi, bj])})
    print(f"cross-dataset near-dups (hamming<=5/256): {len(cross)}", flush=True)
    for c in cross[:20]:
        print("  ", c)

    rep = {
        "splits": stats,
        "long_side_px": {"min": int(long_side.min()),
                         "p5": int(np.percentile(long_side, 5)),
                         "median": int(np.median(long_side)),
                         "p95": int(np.percentile(long_side, 95)),
                         "max": int(long_side.max())},
        "under_1280": int((long_side < 1280).sum()),
        "internal_md5_dups": len(md5_dups),
        "internal_near_dup_pairs": len(uf_pairs),
        "internal_near_dup_examples": uf_pairs[:10],
        "cross_dataset_dups_vs_oral_mamba": cross,
    }
    REPORT.write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items()
                      if k != "cross_dataset_dups_vs_oral_mamba"}, indent=2))

    # thumbnails
    tdir = Path(f"{_ROOT}/outputs/lesi_thumbnails")
    tdir.mkdir(exist_ok=True)
    for sp in ("train", "valid", "test"):
        for p in all_imgs[sp][:4]:
            img = cv2.imread(str(p), cv2.IMREAD_REDUCED_COLOR_4)
            if img is not None:
                cv2.imwrite(str(tdir / f"{sp}_{p.name[:30]}.jpg"), img)


def pd_read_csv(p):
    import pandas as pd
    return pd.read_csv(p)


if __name__ == "__main__":
    main()
