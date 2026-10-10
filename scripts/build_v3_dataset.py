"""V3 data prep: dedup groups + group-level stratified split + 3-class labels.

Fixes two structural problems of the v1/v2 dataset:
  1) train/test leakage: 346 exact-duplicate groups + massive near-duplicates
     spanned random splits.  Now md5 + dHash(hamming<=8) union-find groups are
     kept intact within one split.
  2) contradictory supervision: Oral-Mamba masks annotate gingivitis/plaque/
     caries (px 1/2/4) but v1 trained binary calculus (px 3) treating every
     other lesion pixel as background.  V3 labels: class0=calculus(px3),
     class1=other_lesion(px1|px2|4) so confusable lesions are explicit.

Outputs (raw read-only, everything derived):
  datasets/processed/calculus_seg_v3/
      images/{train,val,test}          symlinks -> calculus_seg images
      labels/{train,val,test}          NEW 3-class polygon txt
      masks_original/{train,val,test}  symlinks -> calculus_seg masks
      split_manifest_v3.csv
      calculus_v3.yaml
  outputs/reports/v3_split_report.json
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

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.utils.paths import PROCESSED_DIR, REPORTS_DIR

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

SRC = PROCESSED_DIR / "calculus_seg"
OUT = PROCESSED_DIR / "calculus_seg_v3"
DHASH_BITS = 256
DHASH_THRESHOLD = 8
MIN_AREA_RATIO = 4e-5
MIN_AREA_PX_FLOOR = 8
SEED = 42


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(p: Path) -> np.ndarray | None:
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (17, 16), interpolation=cv2.INTER_AREA)
    return (img[:, 1:] > img[:, :-1]).astype(np.uint8).ravel()


class UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def contours_to_lines(mask: np.ndarray, cls_id: int, min_area_px: int) -> list[str]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    lines = []
    for c in contours:
        if cv2.contourArea(c) < min_area_px:
            continue
        pts = c.reshape(-1, 2)
        lines.append(f"{cls_id} " + " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in pts))
    return lines


def main() -> None:
    rng = np.random.default_rng(SEED)
    man = pd.read_csv(SRC / "split_manifest.csv")
    man = man.reset_index(drop=True)
    n = len(man)
    print(f"images: {n}")

    # ---- 1. fingerprints ----
    man["md5"] = [md5_file(Path(p)) for p in man.image_path]
    hashes = np.zeros((n, DHASH_BITS), np.uint8)
    ok = np.ones(n, bool)
    for i, p in enumerate(man.image_path):
        h = dhash(Path(p))
        if h is None:
            ok[i] = False
        else:
            hashes[i] = h
    print(f"dhash ok: {ok.sum()} / {n}")

    uf = UF(n)
    for md5, idx in man.groupby("md5").groups.items():
        idx = list(idx)
        for j in idx[1:]:
            uf.union(idx[0], j)
    POPCNT = np.array([bin(i).count("1") for i in range(256)], np.uint8)
    packed = np.packbits(hashes, axis=1)  # (n, 32)
    CH = 256
    near_pairs = 0
    for s in range(0, n, CH):
        e = min(s + CH, n)
        xor = np.bitwise_xor(packed[s:e, None, :], packed[None, :, :])
        d = POPCNT[xor].sum(axis=2)
        for bi, bj in zip(*np.where((d <= DHASH_THRESHOLD) & (d > 0))):
            uf.union(s + bi, int(bj))
            near_pairs += 1
    man["group_id"] = [uf.find(i) for i in range(n)]
    n_groups = man.group_id.nunique()
    print(f"groups: {n_groups} (exact-md5 merges + {near_pairs} near pairs)")

    # ---- 2. group-level stratified split ----
    dom = man.image_id.str.extract(r"^(caries_0|gingivitis)")[0].fillna("other")
    man["domain"] = dom
    g = man.groupby("group_id").agg(
        domain=("domain", "first"), pos=("has_calculus", "max"), size=("image_id", "size"))
    g = g.reset_index()
    g["stratum"] = g.domain + "|" + g.pos.map({True: "pos", False: "neg"})
    g["rnd"] = rng.random(len(g))
    g = g.sort_values(["stratum", "rnd"]).reset_index(drop=True)
    split_of_group = {}
    for st, sub in g.groupby("stratum"):
        k = len(sub)
        n_tr, n_va = int(round(k * 0.6)), int(round(k * 0.2))
        splits = (["train"] * n_tr + ["val"] * n_va + ["test"] * (k - n_tr - n_va))
        for gid, sp in zip(sub.group_id, splits):
            split_of_group[gid] = sp
    man["split"] = [split_of_group[g] for g in man.group_id]

    # ---- 3. dataset build ----
    if OUT.exists():
        import shutil
        shutil.rmtree(OUT)
    for sp in ("train", "val", "test"):
        (OUT / "images" / sp).mkdir(parents=True)
        (OUT / "labels" / sp).mkdir(parents=True)
        (OUT / "masks_original" / sp).mkdir(parents=True)

    rows = []
    conv_check = []
    check_idx = set(rng.choice(n, size=120, replace=False).tolist())
    n_cls0 = n_cls1 = 0
    for i, r in man.iterrows():
        sp = r.split
        img_p = Path(r.image_path)
        msk_p = Path(r.mask_original_path)
        (OUT / "images" / sp / img_p.name).symlink_to(img_p)
        (OUT / "masks_original" / sp / msk_p.name).symlink_to(msk_p)
        m = cv2.imread(str(Path(r.mask_full_path) if "mask_full_path" in man.columns
                           else msk_p), cv2.IMREAD_UNCHANGED)
        # mask_original_path is the BINARY calculus mask; need the FULL multiclass
        # mask from raw labels for 3-class labels
        stem = img_p.stem
        raw_msk = Path(f"{_ROOT}/datasets/raw/oral_mamba/lables") / f"{stem}.png"
        full = cv2.imread(str(raw_msk), cv2.IMREAD_UNCHANGED)
        if full is None:
            rows.append({**r.to_dict(), "label_ok": False})
            continue
        if full.ndim == 3:
            full = full[:, :, 0]
        h, w = full.shape
        min_area = max(MIN_AREA_PX_FLOOR, int(MIN_AREA_RATIO * h * w))
        lines0 = contours_to_lines((full == 3).astype(np.uint8), 0, min_area)
        other = np.isin(full, [1, 2, 4]).astype(np.uint8)
        lines1 = contours_to_lines(other, 1, min_area)
        lines = lines0 + lines1
        n_cls0 += len(lines0)
        n_cls1 += len(lines1)
        (OUT / "labels" / sp / f"{stem}.txt").write_text("\n".join(lines))
        rows.append({**r.to_dict(), "image_path": str(OUT / "images" / sp / img_p.name),
                     "mask_original_path": str(OUT / "masks_original" / sp / msk_p.name),
                     "label_ok": True, "n_calc_inst": len(lines0),
                     "n_other_inst": len(lines1)})
        if i in check_idx:
            from src.data.build_calculus_dataset import rasterize_lines
            rec0 = rasterize_lines([ln for ln in lines if ln.startswith("0 ")], (h, w))
            gt0 = (full == 3).astype(np.uint8)
            inter = int((rec0 & gt0).sum())
            union = int((rec0 | gt0).sum())
            conv_check.append({"stem": stem, "iou_cls0": inter / union if union else 1.0})
    v3 = pd.DataFrame(rows)
    v3.to_csv(OUT / "split_manifest_v3.csv", index=False)

    # split integrity check
    leak = []
    for md5v, sub in v3.groupby("md5"):
        if sub.split.nunique() > 1:
            leak.append(sub[["image_id", "group_id", "split"]].to_dict("records"))
    yaml_txt = (f"path: {OUT}\ntrain: images/train\nval: images/val\n"
                f"test: images/test\nnames:\n  0: calculus\n  1: other_lesion\n")
    (OUT / "calculus_v3.yaml").write_text(yaml_txt)

    rep = {
        "n_images": int(n), "n_groups": int(n_groups),
        "near_pairs_merged": int(near_pairs),
        "group_size_hist": g["size"].value_counts().sort_index().to_dict(),
        "split_counts": v3.split.value_counts().to_dict(),
        "split_pos_rate": v3.groupby("split").has_calculus.mean().round(4).to_dict(),
        "split_domain": v3.groupby(["split", "domain"]).size().unstack(fill_value=0).to_dict(),
        "n_cls0_instances": int(n_cls0), "n_cls1_instances": int(n_cls1),
        "conv_check_n": len(conv_check),
        "conv_iou_min": float(min(c["iou_cls0"] for c in conv_check)) if conv_check else None,
        "conv_iou_mean": float(np.mean([c["iou_cls0"] for c in conv_check])) if conv_check else None,
        "cross_split_md5_groups": leak,
        "images_without_full_mask": int((~v3.label_ok).sum()),
    }
    (REPORTS_DIR / "v3_split_report.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps({k: v for k, v in rep.items() if k != "split_domain"}, indent=2, default=str))
    print("cross-split leak groups:", len(leak))


if __name__ == "__main__":
    main()
