"""Audit the Mendeley dental plaque dataset (Part 1 + Part 2).

Checks: image/label matching, split purity (patient-level), label validity,
class balance, image integrity/dims, md5+dHash duplicates across splits.

Outputs:
  outputs/reports/plaque_mendeley_audit.json
  outputs/reports/plaque_mendeley_audit.md
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

RAW = Path(f"{_ROOT}/datasets/raw/Plaque")
PARTS = [RAW / "mendeley-dataset-materials_Part_1", RAW / "mendeley-dataset-materials_Part_2"]
OUT = Path(f"{_ROOT}/outputs/reports")
MIN_TOOTH_ID, MAX_TOOTH_ID = 1, 32


def dhash(img, size=8):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (size + 1, size))
    return np.packbits(g[:, 1:] > g[:, :-1]).tobytes()


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> None:
    rep = {"parts": {}, "splits": {}, "leakage": {}, "labels": {}, "integrity": {}, "duplicates": {}}
    issues = []

    # ---------- load official splits ----------
    split_of = {}          # (part, filename_lower) -> split
    patient_split = defaultdict(set)   # patient -> {splits}
    csv_rows = 0
    for pi, part in enumerate(PARTS, 1):
        csvp = part / "data_splits.csv"
        with open(csvp, newline="", encoding="utf-8-sig") as f:
            rd = csv.DictReader(f)
            cols = rd.fieldnames
            for row in rd:
                fn = row["Image-Filename"].strip()
                status = (row.get("status") or row.get("Recommend") or "").strip().lower()
                stem = Path(fn).stem.lower()
                if stem in {k[1] for k in split_of if k[0] == pi}:
                    issues.append(f"part{pi}: duplicate rows for {fn}")
                split_of[(pi, stem)] = status
                patient_split[row["patient"].strip().lower()].add(status)
                csv_rows += 1
        rep["parts"][f"part{pi}"] = {"csv_cols": cols}
    rep["splits"]["csv_rows"] = csv_rows

    split_conflicts = {p: s for p, s in patient_split.items() if len(s) > 1}
    rep["leakage"]["patients_with_multiple_splits"] = dict(split_conflicts)
    rep["leakage"]["patient_split_counts"] = dict(Counter(next(iter(s)) for s in patient_split.values()))

    # ---------- enumerate images ----------
    images = {}   # (part, stem_lower) -> path
    for pi, part in enumerate(PARTS, 1):
        root = part / "data" / "images"
        n = 0
        for pdir in sorted(root.iterdir()):
            if not pdir.is_dir():
                continue
            for img in pdir.iterdir():
                if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                key = (pi, img.stem.lower())
                if key in images:
                    issues.append(f"part{pi}: duplicate image stem {img.name}")
                images[key] = img
                n += 1
        rep["parts"][f"part{pi}"]["images_found"] = n

    # split coverage
    in_csv = {k for k in split_of}
    in_dir = set(images)
    missing_in_csv = sorted(f"part{k[0]}/{images[k].parent.name}/{images[k].name}" for k in in_dir - in_csv)
    missing_in_dir = [f"part{k[0]}/{k[1]}" for k in in_csv - in_dir]
    rep["splits"]["missing_in_csv"] = missing_in_csv
    rep["splits"]["missing_in_dir"] = missing_in_dir[:50]
    rep["splits"]["missing_in_dir_n"] = len(missing_in_dir)

    # augmented variants per base
    aug_re = re.compile(r"_(brightness-down|brightness-up|rotate-left-15|rotate-right-15|"
                        r"flip_horizontal|flip-horizontal|blur|light|dark)$")
    base_count = sum(1 for k in in_dir if not aug_re.search(k[1]))
    rep["splits"]["base_images"] = base_count
    rep["splits"]["augmented_images"] = len(in_dir) - base_count
    rep["splits"]["split_counts_all"] = dict(Counter(split_of.get(k, "UNASSIGNED") for k in in_dir))
    rep["splits"]["split_counts_base"] = dict(
        Counter(split_of.get(k, "UNASSIGNED") for k in in_dir if not aug_re.search(k[1])))

    # ---------- labels + integrity ----------
    # case-insensitive label resolution: lower(dirname) -> dir, then files
    lbl_dirs = {}
    for part in PARTS:
        m = {}
        lr = part / "data" / "labels"
        if lr.exists():
            for d in lr.iterdir():
                if d.is_dir():
                    m[d.name.lower()] = d
        lbl_dirs[part] = m
    lbl_files = {}  # (part, lower_dirname) -> {lower_stem: Path}
    for part, m in lbl_dirs.items():
        for low, d in m.items():
            lbl_files[(part, low)] = {f.stem.lower(): f for f in d.iterdir() if f.suffix.lower() == ".txt"}

    def resolve_lbl(part, img_path):
        d = lbl_dirs[part].get(img_path.parent.name.lower())
        if d is None:
            return None
        return lbl_files[(part, img_path.parent.name.lower())].get(img_path.stem.lower())

    lbl_stats = Counter()
    tooth_ids = Counter()
    split_box = Counter()
    split_tooth_pos = Counter()
    bad_img = []
    dims = []
    no_lbl = []
    box_split_label = Counter()  # (split, plaque) -> n

    for key, img_path in sorted(images.items()):
        pi, stem = key
        part = PARTS[pi - 1]
        lbl = resolve_lbl(part, img_path)
        if lbl is None:
            no_lbl.append(f"part{pi}/{img_path.name}")
            continue
        lines = [l for l in lbl.read_text().splitlines() if l.strip()]
        lbl_stats["images_with_labels"] += 1
        for ln in lines:
            f = ln.split()
            if len(f) != 6:
                lbl_stats["malformed_lines"] += 1
                continue
            p, cx, cy, w, h, tid = int(f[0]), *map(float, f[1:5]), int(f[5])
            if p not in (0, 1):
                lbl_stats["bad_plague_flag"] += 1
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                lbl_stats["bbox_out_of_range"] += 1
            if not (MIN_TOOTH_ID <= tid <= MAX_TOOTH_ID):
                lbl_stats["bad_tooth_id"] += 1
            tooth_ids[tid] += 1
            sp = split_of.get(key, "UNASSIGNED")
            split_box[sp] += 1
            box_split_label[(sp, p)] += 1
        # image decode (skip known CRC-broken file quickly by decode catch)
        try:
            im = cv2.imread(str(img_path))
            if im is None:
                raise IOError("decode failed")
            dims.append((im.shape[1], im.shape[0]))
            lbl_stats["images_decoded"] += 1
        except Exception:
            bad_img.append(f"part{pi}/{img_path.parent.name}/{img_path.name}")
            lbl_stats["images_undecodable"] += 1

    rep["labels"] = {
        "no_label_images": len(no_lbl),
        "no_label_examples": no_lbl[:20],
        "undecodable": bad_img,
        "total_boxes": sum(split_box.values()),
        "boxes_per_split": dict(split_box),
        "plaque_ratio_per_split": {
            sp: round(box_split_label[(sp, 1)] / max(box_split_label[(sp, 1)] + box_split_label[(sp, 0)], 1), 4)
            for sp in ("train", "validation", "test")},
        "boxes_per_image_mean": round(sum(split_box.values()) / max(lbl_stats["images_with_labels"], 1), 2),
        "tooth_id_top10": tooth_ids.most_common(10),
        "n_tooth_ids_used": len(tooth_ids),
    }
    rep["labels"].update(lbl_stats)
    if dims:
        d = np.array(dims)
        rep["integrity"] = {"n_dims": len(d), "w_min": int(d[:, 0].min()), "w_max": int(d[:, 0].max()),
                            "h_min": int(d[:, 1].min()), "h_max": int(d[:, 1].max()),
                            "w_median": float(np.median(d[:, 0])), "h_median": float(np.median(d[:, 1]))}
    if issues:
        rep["integrity"]["issues"] = issues[:30]

    # ---------- tooth-level dedup label check ----------
    # (part, stem, tooth) -> {0/1 flags}
    tooth_conflict = 0
    tooth_map = defaultdict(list)
    for key in sorted(images):
        part = PARTS[key[0] - 1]
        lbl = resolve_lbl(part, images[key])
        if lbl is None:
            continue
        for ln in lbl.read_text().splitlines():
            f = ln.split()
            if len(f) != 6:
                continue
            tooth_map[(key[0], key[1], int(f[5]))].append(int(f[0]))
    for v in tooth_map.values():
        if len(set(v)) > 1:
            tooth_conflict += 1
    rep["labels"]["tooth_group_label_conflicts"] = tooth_conflict
    rep["labels"]["tooth_instances_total"] = len(tooth_map)

    # ---------- duplicates across splits (md5 on base images only) ----------
    hash2split = defaultdict(set)
    h2name = {}
    for key, img_path in sorted(images.items()):
        if aug_re.search(key[1]):
            continue
        sp = split_of.get(key, "UNASSIGNED")
        h = md5(img_path)
        hash2split[h].add(sp)
        h2name.setdefault(h, []).append(f"part{key[0]}/{img_path.parent.name}/{img_path.name}")
    cross = {h: s for h, s in hash2split.items() if len(s - {"UNASSIGNED"}) > 1}
    rep["duplicates"]["md5_cross_split_groups"] = len(cross)
    rep["duplicates"]["md5_cross_split_examples"] = [
        {"splits": sorted(s), "files": h2name[h][:4]} for h, s in list(cross.items())[:10]]
    rep["duplicates"]["md5_unique_hashes"] = len(hash2split)

    # dHash cross-split (near-duplicates)
    dh2split = defaultdict(set)
    dh2name = {}
    for key, img_path in sorted(images.items()):
        if aug_re.search(key[1]):
            continue
        sp = split_of.get(key, "UNASSIGNED")
        try:
            im = cv2.imread(str(img_path))
            if im is None:
                continue
            d = dhash(im)
        except Exception:
            continue
        dh2split[d].add(sp)
        dh2name.setdefault(d, []).append(f"part{key[0]}/{img_path.parent.name}/{img_path.name}")
    cross_dh = {d: s for d, s in dh2split.items() if len(s - {"UNASSIGNED"}) > 1}
    rep["duplicates"]["dhash_cross_split_groups"] = len(cross_dh)
    rep["duplicates"]["dhash_cross_split_examples"] = [
        {"splits": sorted(s), "files": dh2name[d][:4]} for d, s in list(cross_dh.items())[:10]]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plaque_mendeley_audit.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))

    # ---------- markdown ----------
    m = ["# Mendeley 牙菌斑数据集审计报告", ""]
    m.append(f"- 官方 split 行数: {csv_rows}；图片总数: {len(images)}（base {base_count} / 增强变体 {len(in_dir)-base_count}）")
    m.append(f"- 图片-split 覆盖: 缺 csv {len(missing_in_csv)}，缺文件 {len(missing_in_dir)}")
    m.append(f"- 患者跨 split 泄漏: {len(split_conflicts)} 名")
    m.append(f"- split 分布(全部): {rep['splits']['split_counts_all']}")
    m.append(f"- split 分布(仅 base): {rep['splits']['split_counts_base']}")
    m.append(f"- 无法解码图片: {bad_img}")
    m.append(f"- 无标签图片: {len(no_lbl)}")
    m.append(f"- 总框数: {sum(split_box.values())}；均值/图: {rep['labels']['boxes_per_image_mean']}")
    m.append(f"- plaque=1 框占比 per split: {rep['labels']['plaque_ratio_per_split']}")
    m.append(f"- (image,tooth) 实例: {len(tooth_map)}，组内标签冲突: {tooth_conflict}")
    m.append(f"- md5 跨 split 重复组: {len(cross)}；dHash 跨 split 近重复组: {len(cross_dh)}")
    m.append(f"- 牙位 ID 使用数: {len(tooth_ids)}（top: {tooth_ids.most_common(6)}）")
    m.append(f"- 类别失衡说明: 数据集按框标注，plaque=1 框占比约 98%，负样本极少；"
             f"训练需对 normal 类过采样，评估以 Sens/Spec/F1/AUROC 为主，不使用 Accuracy。")
    (OUT / "plaque_mendeley_audit.md").write_text("\n".join(m) + "\n")
    print("\n".join(m))
    return 0


if __name__ == "__main__":
    sys.exit(main())
