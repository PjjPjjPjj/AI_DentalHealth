"""Audit the Kaggle dental-disease archive (classification folders + YOLO subset).

Checks:
  - per-folder counts, decodability, image sizes
  - md5 dups within dataset
  - md5 + dHash cross-check against our training data (oral_mamba, caries_det_v2)
    -> leakage would invalidate external-test usage
  - YOLO subset: label class distribution, boxes/img, image-label matching
Outputs: outputs/reports/archive_kaggle_audit.{json,md}
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
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

ROOT = Path(f"{_ROOT}")
A = ROOT / "datasets/raw/archive/extracted"
OUT = ROOT / "outputs/reports"
CLASSES = ["Caries", "Ulcer", "Tooth Discoloration", "Gingivitis"]

YIMG = A / "Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/Data/images"
YLBL = A / "Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/Data/labels"


def md5(p, chunk=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def dhash(img, size=8):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (size + 1, size))
    return np.packbits(g[:, 1:] > g[:, :-1]).tobytes()


def main() -> None:
    rep = {"folders": {}, "yolo": {}, "leakage": {}}

    # ---- classification folders ----
    folder_files = {}
    for d in sorted(A.iterdir()):
        if not d.is_dir() or d.name.startswith("Caries_Gingivitus"):
            continue
        files = [p for p in d.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        bad, dims = [], []
        for p in files:
            im = cv2.imread(str(p))
            if im is None:
                bad.append(p.name)
            else:
                dims.append((im.shape[1], im.shape[0]))
        dims_a = np.array(dims) if dims else np.zeros((0, 2))
        rep["folders"][d.name] = {
            "n_images": len(files),
            "undecodable": len(bad),
            "w_median": int(np.median(dims_a[:, 0])) if len(dims) else 0,
            "h_median": int(np.median(dims_a[:, 1])) if len(dims) else 0,
            "w_min": int(dims_a[:, 0].min()) if len(dims) else 0,
            "w_max": int(dims_a[:, 0].max()) if len(dims) else 0,
        }
        folder_files[d.name] = files

    # ---- YOLO subset ----
    cls_cnt = Counter()
    boxes_per_img = []
    unmatched_lbl = 0
    unmatched_img = 0
    lbl_sizes = Counter()
    splits = Counter()
    for sp in ("train", "val", "test"):
        ldir = YLBL / sp
        idir = YIMG / sp
        if not ldir.exists():
            continue
        lbls = list(ldir.glob("*.txt"))
        splits[sp] = len(lbls)
        for lf in lbls:
            imgf = None
            for ext in (".jpg", ".jpeg", ".png"):
                c = idir / (lf.stem + ext)
                if c.exists():
                    imgf = c
                    break
            if imgf is None:
                unmatched_lbl += 1
                continue
            lines = [l.split() for l in lf.read_text().splitlines() if l.strip()]
            boxes_per_img.append(len(lines))
            for l in lines:
                if len(l) >= 5:
                    cls_cnt[int(l[0])] += 1
                    lbl_sizes[lf.stem] += 1
        if idir.exists():
            imgs = list(idir.glob("*.jpg")) + list(idir.glob("*.jpeg"))
            stems = {p.stem for p in lbls}
            unmatched_img = sum(1 for p in imgs if p.stem not in stems)
    rep["yolo"] = {
        "splits": dict(splits), "class_counts": {CLASSES[k] if k < 4 else k: v
                                                 for k, v in sorted(cls_cnt.items())},
        "boxes_per_img_mean": round(float(np.mean(boxes_per_img)), 2) if boxes_per_img else 0,
        "images_without_labels": unmatched_img,
        "labels_without_images": unmatched_lbl,
        "caries_images": sum(1 for lf in YLBL.rglob("*.txt")
                             if any(l.split() and l.split()[0] == "0"
                                    for l in lf.read_text().splitlines() if l.strip())),
    }

    # ---- leakage vs our training data ----
    ours = []
    om = ROOT / "datasets/processed/calculus_seg_v3"
    man = om / "split_manifest_v3.csv"
    import pandas as pd
    if man.exists():
        df = pd.read_csv(man)
        ours += [str(p) for p in df.image_path]
    c2 = ROOT / "datasets/processed/caries_det_v2/split_manifest.csv"
    if c2.exists():
        df2 = pd.read_csv(c2)
        ours += [str(p) for p in df2.img]
    ours_md5 = {}
    ours_dh = {}
    for p in ours:
        if Path(p).exists():
            h = md5(p)
            ours_md5[h] = p
            try:
                im = cv2.imread(p)
                if im is not None:
                    ours_dh[dhash(im)] = p
            except Exception:
                pass
    new_files = [p for flist in folder_files.values() for p in flist]
    ys = list(YIMG.rglob("*.jpg")) + list(YIMG.rglob("*.jpeg"))
    all_new = new_files + ys
    md5_hits, dh_hits = [], []
    for p in all_new:
        h = md5(p)
        if h in ours_md5:
            md5_hits.append({"new": str(p), "ours": ours_md5[h]})
            continue
        im = cv2.imread(str(p))
        if im is None:
            continue
        d = dhash(im)
        if d in ours_dh:
            dh_hits.append({"new": str(p), "ours": ours_dh[d]})
    rep["leakage"] = {
        "checked_new": len(all_new), "checked_ours": len(ours_md5),
        "md5_overlap": len(md5_hits), "dhash_overlap": len(dh_hits),
        "md5_examples": md5_hits[:5], "dhash_examples": dh_hits[:5],
    }

    # ---- within-dataset md5 dups ----
    h2 = defaultdict(list)
    for p in all_new:
        h2[md5(p)].append(str(p))
    dups = {h: v for h, v in h2.items() if len(v) > 1}
    rep["within_dups_groups"] = len(dups)
    rep["within_dups_examples"] = list(dups.values())[:5]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "archive_kaggle_audit.json").write_text(json.dumps(rep, indent=2, default=str))
    m = ["# Kaggle 口腔疾病数据集审计（archive.zip）", ""]
    for k, v in rep["folders"].items():
        m.append(f"- **{k}**: {v['n_images']} 图（损坏 {v['undecodable']}），中位尺寸 {v['w_median']}x{v['h_median']}（范围 {v['w_min']}~{v['w_max']}）")
    m.append(f"- YOLO 子集 splits: {rep['yolo']['splits']}，类别框数: {rep['yolo']['class_counts']}")
    m.append(f"- YOLO 含龋齿(class 0)框的图片数: {rep['yolo']['caries_images']}；框/图均值 {rep['yolo']['boxes_per_img_mean']}")
    m.append(f"- 图片无标签: {rep['yolo']['images_without_labels']}；标签无图: {rep['yolo']['labels_without_images']}")
    m.append(f"- **与我们训练数据泄漏**: md5 {rep['leakage']['md5_overlap']}，dHash {rep['leakage']['dhash_overlap']}（检查 {rep['leakage']['checked_new']} 新图 vs {rep['leakage']['checked_ours']} 旧图）")
    m.append(f"- 数据集内部 md5 重复组: {rep['within_dups_groups']}")
    (OUT / "archive_kaggle_audit.md").write_text("\n".join(m) + "\n")
    print("\n".join(m))


if __name__ == "__main__":
    main()
