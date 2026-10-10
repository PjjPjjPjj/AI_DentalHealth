"""Post-Stage2 verifier dataset build (Plan B, data only - NO training).

Pipeline for positive/negative sampling (train split only):
  image -> Stage1 (c1=0.05) -> bbox-center 160 crop -> Stage2 V1 (imgsz 640,
  independent instances, conf>=0.05) -> intra-crop mask-NMS(0.5) -> final
  instances.  Each final instance = one verifier sample.

Labels (per instance vs crop GT instances, 640-space 1-to-1 greedy):
  positive : best IoU >= 0.5
  negative : best IoU < 0.1
  ambiguous: 0.1 <= best IoU < 0.5  (excluded from first version, tagged)

Saved per sample: RGB 160x160 crop PNG, Stage2 instance mask PNG (160x160),
Stage1 candidate mask PNG (160x160), metadata with stage1_conf,
stage2_conf, stage1_mask_area, stage2_mask_area, stage1_stage2_mask_iou,
matched_gt, best_gt_iou, size_bucket.
Split by ORIGINAL image id (80/20 seed 42); main val/test untouched.
Outputs: datasets/processed/calculus_v2/verifier_dataset/
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

import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.inference.twostage import TARGET, TwoStageSegmenter
from src.utils.paths import PROCESSED_DIR

DATA = PROCESSED_DIR / "calculus_seg"
V2 = PROCESSED_DIR / "calculus_v2"
OUT = V2 / "verifier_dataset"
C1 = 0.05
SIDE = 160
NMS_IOU = 0.5


def bucket_of(r: float) -> str:
    if r < 0.00552:
        return "small"
    if r < 0.01565:
        return "medium"
    return "large"


def main() -> None:
    seg = TwoStageSegmenter("outputs/calculus_yolo11m_full/weights/best.pt",
                            "outputs/calculus_yolo11m_crop/weights/best.pt",
                            c1=C1, c2=0.05, device=0, nms_iou=NMS_IOU,
                            score_mode="s2")
    man = pd.read_csv(DATA / "split_manifest.csv")
    rows = man[man.split == "train"]
    if OUT.exists():
        shutil.rmtree(OUT)
    for sp in ("train", "val"):
        for cls in ("positive", "negative", "ambiguous"):
            (OUT / sp / cls).mkdir(parents=True)
    meta = []
    img_splits = {}
    srcs = sorted(rows.image_id.unique())
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(srcs))
    cut = int(0.8 * len(srcs))
    train_img = set(srcs[i] for i in perm[:cut])
    for ri, (_, r) in enumerate(rows.iterrows()):
        img = cv2.imread(r["image_path"])
        H, W = img.shape[:2]
        gtb = cv2.imread(r["mask_original_path"], cv2.IMREAD_UNCHANGED)
        gtb = (gtb > 0) if gtb is not None else np.zeros((H, W), np.uint8)
        sp = "train" if r["image_id"] in train_img else "val"
        boxes, confs, masks = seg._stage1_masks(r["image_path"])
        for (b, sc1, msk) in zip(boxes, confs, masks):
            msk = msk.astype(np.uint8)
            bx = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
            x0 = int(round((bx[0] + bx[2]) / 2 - SIDE / 2))
            y0 = int(round((bx[1] + bx[3]) / 2 - SIDE / 2))
            x0 = max(0, min(x0, W - SIDE)); y0 = max(0, min(y0, H - SIDE))
            x1, y1 = x0 + SIDE, y0 + SIDE
            ch, cw = y1 - y0, x1 - x0
            crop_img = img[y0:y1, x0:x1]
            crop = cv2.resize(crop_img, (TARGET, TARGET),
                              interpolation=cv2.INTER_AREA)
            # GT instances in crop -> 640 space
            gtc = gtb[y0:y1, x0:x1].astype(np.uint8)
            ng, comp, _, _ = cv2.connectedComponentsWithStats(gtc, 8)
            gt_insts, gt_full_ratio = [], []
            for i in range(1, ng):
                if (comp == i).sum() < 5:
                    continue
                m = (comp == i).astype(np.uint8)
                gm640 = cv2.resize(m, (TARGET, TARGET),
                                   interpolation=cv2.INTER_AREA) > 0.5
                gt_insts.append(gm640.astype(np.uint8))
                # full-GT ratio estimate for bucket (crop-local px scaled)
                gt_full_ratio.append(
                    bucket_of(int(m.sum()) / float(SIDE * SIDE) * 0.5))
            # step: stage2 instances + intra-crop NMS (640-space)
            insts = seg._stage2_instances(crop)
            sel = [{"x0": 0, "y0": 0, "cw": TARGET, "ch": TARGET,
                    "bin": i["bin640"], "s2_conf": i["s2_conf"],
                    "score": i["s2_conf"]} for i in insts]
            from src.inference.fusion import greedy_nms
            keep = greedy_nms(sel, "mask_nms", NMS_IOU, TARGET, TARGET)
            for k in keep:
                P = k["bin"]
                best_iou, best_gi = 0.0, -1
                for gi, G in enumerate(gt_insts):
                    inter = int((P & G).sum())
                    uni = int((P | G).sum())
                    io = inter / uni if uni else 0.0
                    if io > best_iou:
                        best_iou, best_gi = io, gi
                label = ("positive" if best_iou >= 0.5 else
                         ("ambiguous" if best_iou >= 0.1 else "negative"))
                # features
                s2m = cv2.resize(P.astype(np.uint8), (SIDE, SIDE),
                                 interpolation=cv2.INTER_AREA) > 0.5
                s1m = cv2.resize(msk[y0:y1, x0:x1], (SIDE, SIDE),
                                 interpolation=cv2.INTER_AREA) > 0.5
                s1s2_iou = (int((s1m & s2m).sum()) /
                            max(int((s1m | s2m).sum()), 1))
                cid = f"v{ri:05d}_{len(meta):06d}"
                if label != "ambiguous":
                    dst = OUT / sp / label
                    cv2.imwrite(str(dst / f"{cid}.png"), crop_img)
                else:
                    dst = OUT / sp / label
                    cv2.imwrite(str(dst / f"{cid}.png"), crop_img)
                cv2.imwrite(str(OUT / sp / label / f"{cid}_s2mask.png"),
                            (s2m * 255).astype(np.uint8))
                cv2.imwrite(str(OUT / sp / label / f"{cid}_s1mask.png"),
                            (s1m * 255).astype(np.uint8))
                meta.append({"sample_id": cid, "split_var": sp, "label": label,
                             "image_id": r["image_id"],
                             "s1_conf": round(float(sc1), 4),
                             "s2_conf": round(float(k["s2_conf"]), 4),
                             "s1_mask_area": int(s1m.sum()),
                             "s2_mask_area": int(s2m.sum()),
                             "s1_s2_mask_iou": round(s1s2_iou, 4),
                             "matched_gt": best_gi,
                             "best_gt_iou": round(best_iou, 4),
                             "bucket": (gt_full_ratio[best_gi]
                                        if 0 <= best_gi < len(gt_full_ratio)
                                        else "none")})
        if (ri + 1) % 400 == 0:
            print(f"  {ri + 1}/{len(rows)}", flush=True)
    md = pd.DataFrame(meta)
    md = md.rename(columns={"split_var": "split"})
    md.to_csv(OUT / "metadata.csv", index=False)
    print("\n== verifier dataset counts ==")
    print(md.groupby(["split", "label"]).size().to_string())
    print("\n== label x bucket ==")
    print(md.groupby(["label", "bucket"]).size().to_string())
    print("\n== split leakage check ==")
    tr = set(md[md.split == "train"].image_id)
    va = set(md[md.split == "val"].image_id)
    print("image overlap:", len(tr & va))
    main_val_test = set(pd.read_csv(DATA / "split_manifest.csv")[
        pd.read_csv(DATA / "split_manifest.csv").split.isin(["val", "test"])].image_id)
    print("main val/test images touched:", len((tr | va) & main_val_test))


if __name__ == "__main__":
    main()
