"""Plan B STEP1-2: build Stage1-proposal binary classifier dataset.

Classes: positive = normal_positive + hard_positive (1197),
         negative = hard_negative + easy_negative (574);
ambiguous excluded. Crops stay bbox-centered 160x160 (no GT re-centering).
Split: by ORIGINAL image id (grouped), train_internal 80/20 seed 42.
Main val/test never touch this dataset.
Metadata: image_id, crop_id, label, source_type, primary_bucket,
stage1_conf (recovered by rerunning frozen Stage1 in identical order),
v1_stage2_conf, matched_gt_id, best_iou.

Outputs: datasets/processed/calculus_v2/classifier_dataset/
         {train_classifier,val_classifier}/{positive,negative}
         metadata.csv
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
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.utils.paths import PROCESSED_DIR
from src.inference.twostage import TwoStageSegmenter, TARGET

DATA = PROCESSED_DIR / "calculus_seg"
V2 = PROCESSED_DIR / "calculus_v2"
CROPS = V2 / "crops"
OUT = V2 / "classifier_dataset"
C1 = 0.05
SIDE = 160


def primary(b: str) -> str:
    if pd.notna(b) and "small" in b:
        return "small"
    if pd.notna(b) and "medium" in b:
        return "medium"
    return "large"


def main() -> None:
    m = pd.read_csv(CROPS / "crops_meta.csv")
    pos = m[m.type.isin(["normal_positive", "hard_positive"])].copy()
    neg = m[m.type.isin(["hard_negative", "easy_negative"])].copy()
    pos["label"] = 1
    neg["label"] = 0
    allm = pd.concat([pos, neg])

    # recover stage1 conf by rerunning frozen stage1 in identical order
    seg = TwoStageSegmenter("outputs/calculus_yolo11m_full/weights/best.pt",
                            "outputs/calculus_yolo11m_crop/weights/best.pt",
                            c1=C1, c2=0.05, device=0, nms_iou=0.5, score_mode="s2")
    man = pd.read_csv(DATA / "split_manifest.csv")
    rows = man[man.split == "train"]
    conf_map = {}
    gidx = 0
    for _, r in rows.iterrows():
        img = cv2.imread(r["image_path"])
        H, W = img.shape[:2]
        boxes, confs = seg._stage1_full(r["image_path"])
        for b, sc1 in zip(boxes, confs):
            cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
            x0 = int(round(cx - SIDE / 2)); y0 = int(round(cy - SIDE / 2))
            x0 = max(0, min(x0, W - SIDE)); y0 = max(0, min(y0, H - SIDE))
            conf_map[gidx] = float(sc1)
            gidx += 1
    allm["stage1_conf"] = allm.crop_id.map(
        lambda cid: conf_map.get(int(cid[2:7]), np.nan) if cid.startswith("tr") else np.nan)

    allm["primary_bucket"] = allm.gt_buckets.apply(primary)
    allm["split_var"] = ""

    # grouped split on image id
    srcs = sorted(allm.src_image.unique())
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(srcs))
    cut = int(0.8 * len(srcs))
    train_img = set(srcs[i] for i in perm[:cut])
    allm["image_split"] = allm.src_image.map(lambda s: "train" if s in train_img else "val")

    if OUT.exists():
        shutil.rmtree(OUT)
    for sp in ("train_classifier", "val_classifier"):
        for cls in ("positive", "negative"):
            (OUT / sp / cls).mkdir(parents=True)
    for _, r2 in allm.iterrows():
        sp = "train_classifier" if r2["image_split"] == "train" else "val_classifier"
        cls = "positive" if r2["label"] == 1 else "negative"
        src = CROPS / "images" / f"{r2['crop_id']}.png"
        (OUT / sp / cls / f"{r2['crop_id']}.png").symlink_to(src.resolve())
    meta = allm[["crop_id", "src_image", "label", "type",
                 "primary_bucket", "stage1_conf", "max_s2_conf",
                 "best_inst_iou", "image_split"]].rename(
        columns={"src_image": "image_id", "type": "source_type",
                 "max_s2_conf": "v1_stage2_conf", "best_inst_iou": "best_iou",
                 "image_split": "split"})
    meta.to_csv(OUT / "metadata.csv", index=False)

    print("== counts ==")
    print(meta.groupby(["split", "label"]).size().to_string())
    print("\n== source_type counts ==")
    print(meta.groupby(["split", "source_type"]).size().to_string())
    print("\n== primary bucket (positive) ==")
    print(meta[meta.label == 1].groupby(["split", "primary_bucket"]).size().to_string())
    print("\n== image-level cross-split check ==")
    tr = set(meta[meta.split == "train"].image_id)
    va = set(meta[meta.split == "val"].image_id)
    print("train imgs:", len(tr), "val imgs:", len(va), "overlap:", len(tr & va))
    main_val_test = set(pd.read_csv(DATA / "split_manifest.csv")[
        pd.read_csv(DATA / "split_manifest.csv").split.isin(["val", "test"])].image_id)
    print("main val/test images touching classifier set:", len(
        (tr | va) & main_val_test))


if __name__ == "__main__":
    main()
