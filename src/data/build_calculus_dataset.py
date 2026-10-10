"""Build the calculus YOLO-seg dataset from Oral-Mamba raw masks.

Pipeline (reproducible, seed=42, raw data read-only):
  STEP 5-6  calculus binary masks (pixel==3) + quality audit
  STEP 7    fixed stratified 6:2:2 split (calculus positive/negative)
  STEP 8    mask -> YOLO segmentation polygons (external contours)
  STEP 8.5  polygon conversion quality check (IoU vs original mask, 100 random)

Outputs:
  datasets/processed/calculus_seg/
      images/{train,val,test}
      labels/{train,val,test}          (YOLO polygon txt)
      masks_original/{train,val,test}  (binary calculus masks for audit)
      split_manifest.csv
      calculus.yaml
  outputs/reports/calculus_dataset_statistics.json
  outputs/visualizations/calculus_conversion_check/
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

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from src.utils.logger import get_logger
from src.utils.paths import OUTPUTS_DIR, PROCESSED_DIR, REPORTS_DIR, VISUALIZATIONS_DIR

logger = get_logger("calculus_build")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
CALCULUS_VALUE = 3  # confirmed, see outputs/reports/mask_semantics.md
TRAIN, VAL, TEST = 0.60, 0.20, 0.20


def load_orig_mask(path: Path) -> np.ndarray | None:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        m = m[:, :, 0]
    return m


def contours_to_lines(mask: np.ndarray, min_area_px: int) -> list[str]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    lines = []
    for c in contours:
        if cv2.contourArea(c) < min_area_px:
            continue
        pts = c.reshape(-1, 2)
        lines.append("0 " + " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in pts))
    return lines


def rasterize_lines(lines: list[str], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    m = np.zeros((h, w), np.uint8)
    for line in lines:
        parts = line.split()
        n = (len(parts) - 1) // 2
        xy = [(float(parts[1 + 2 * i]) * w, float(parts[2 + 2 * i]) * h) for i in range(n)]
        if len(xy) < 3:
            continue
        cv2.fillPoly(m, [np.round(np.array(xy)).astype(np.int32)], 1)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="datasets/raw/oral_mamba")
    ap.add_argument("--out", default=str(PROCESSED_DIR / "calculus_seg"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-area-ratio", type=float, default=4e-5,
                    help="min contour area as fraction of image pixels (small-lesion noise filter)")
    ap.add_argument("--min-area-px-floor", type=int, default=8)
    ap.add_argument("--check-n", type=int, default=100,
                    help="number of random samples for polygon IoU check")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    raw = Path(args.raw)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    masks_out = out / "masks_original"
    masks_out.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in (raw / "images").iterdir() if p.suffix.lower() in IMG_EXTS)
    masks = sorted(p for p in (raw / "lables").iterdir() if p.suffix.lower() == ".png")
    img_by_stem = {p.stem: p for p in images}
    msk_by_stem = {p.stem: p for p in masks}
    common = sorted(set(img_by_stem) & set(msk_by_stem))
    logger.info(f"matched pairs: {len(common)}")

    all_split = []  # (stem, has_calculus) for stratified split
    row_list = []
    audit = {"positive_images": 0, "negative_images": 0, "empty_mask_files": [],
             "shape_mismatch": [], "unreadable": [], "anomalies": [],
             "lesion_areas": [], "n_components": [], "lesion_ratio": [],
             "empty_label_count": 0}
    for stem in common:
        im, ms = img_by_stem[stem], msk_by_stem[stem]
        img = cv2.imread(str(im))
        if img is None:
            audit["unreadable"].append(stem)
            continue
        h, w = img.shape[:2]
        m = load_orig_mask(ms)
        if m is None:
            audit["unreadable"].append(stem)
            continue
        if m.shape != (h, w):
            audit["shape_mismatch"].append({
                "image_id": stem, "img": [h, w], "mask": list(m.shape)})
            if m.shape[0] > h:
                m = m[:h, :w]
            else:
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        uniq = set(int(v) for v in np.unique(m))
        if not uniq.isdisjoint({-1, 255}):
            audit["anomalies"].append({"image_id": stem, "values": sorted(uniq)})
        calc = (m == CALCULUS_VALUE).astype(np.uint8)
        n_pos = int(calc.sum())
        has = n_pos > 0
        all_split.append((stem, bool(has)))
        if has:
            audit["positive_images"] += 1
            audit["lesion_areas"].append(int(n_pos))
            audit["lesion_ratio"].append(n_pos / (h * w))
            ncc, _ = cv2.connectedComponents(calc)
            audit["n_components"].append(ncc - 1)
        else:
            audit["negative_images"] += 1
            ncc, _ = cv2.connectedComponents(calc)
            audit["n_components"].append(ncc - 1)

        min_area = max(int(args.min_area_ratio * h * w), args.min_area_px_floor)
        lines = contours_to_lines(calc, min_area)
        if has and not lines:
            audit["empty_mask_files"].append(stem)
            audit["empty_label_count"] += 1
        (masks_out / "train").mkdir(parents=True, exist_ok=True)
        (masks_out / "val").mkdir(parents=True, exist_ok=True)
        (masks_out / "test").mkdir(parents=True, exist_ok=True)
        row_list.append({
            "image_id": stem, "has_calculus": bool(has),
            "lesion_pixels": int(n_pos), "n_components": ncc - 1,
            "area_ratio": n_pos / (h * w), "min_area_px": min_area,
            "n_polygons": len(lines), "img_shape": f"{h}x{w}",
            "polygon_text": "\n".join(lines),
        })
    audit["matched"] = len(common)
    df = pd.DataFrame(row_list)
    df.to_csv(out / "_pre_split_audit.csv", index=False)

    assert len(all_split) == len(df)
    pos = [s for s, h in all_split if h]
    neg = [s for s, h in all_split if not h]
    rng.shuffle(pos); rng.shuffle(neg)

    n_pos_split = {"train": int(len(pos) * TRAIN), "val": int(len(pos) * VAL)}
    n_pos_split["test"] = len(pos) - n_pos_split["train"] - n_pos_split["val"]
    n_neg_split = {"train": int(len(neg) * TRAIN), "val": int(len(neg) * VAL)}
    n_neg_split["test"] = len(neg) - n_neg_split["train"] - n_neg_split["val"]

    splits = {"train": [], "val": [], "test": []}
    for grp, name in ((pos, "pos"), (neg, "neg")):
        i = 0
        for part in ("train", "val", "test"):
            n_take = n_pos_split[part] if name == "pos" else n_neg_split[part]
            splits[part] += grp[i:i + n_take]
            i += n_take
    for part in splits:
        assert len(splits[part]) == n_pos_split[part] + n_neg_split[part]
    logger.info(
        f"split train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    manifest = []
    df_by_stem = df.set_index("image_id")
    for part in ("train", "val", "test"):
        img_dir = out / "images" / part
        lbl_dir = out / "labels" / part
        msk_dir = masks_out / part
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for stem in sorted(splits[part]):
            row = df_by_stem.loc[stem]
            shutil.copy(img_by_stem[stem], img_dir / (stem + ".jpg"))
            (lbl_dir / (stem + ".txt")).write_text(row["polygon_text"])
            m = load_orig_mask(msk_by_stem[stem])
            h, w = m.shape[:2]
            m = (m == CALCULUS_VALUE).astype(np.uint8) * 255
            cv2.imwrite(str(msk_dir / (stem + ".png")), m)
            manifest.append({
                "image_id": stem,
                "image_path": str(img_dir / (stem + ".jpg")),
                "label_path": str(lbl_dir / (stem + ".txt")),
                "mask_original_path": str(msk_dir / (stem + ".png")),
                "has_calculus": bool(row["has_calculus"]),
                "split": part,
            })
    pd.DataFrame(manifest).to_csv(out / "split_manifest.csv", index=False)

    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train", "val": "images/val", "test": "images/test",
        "names": {0: "calculus"},
    }
    (out / "calculus.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    # ---- conversion quality check (STEP 8.5) ----
    check_rng = random.Random(args.seed)
    positives = [m for m in manifest if m["has_calculus"]]
    negatives = [m for m in manifest if not m["has_calculus"]]
    pool = check_rng.sample(positives, min(args.check_n, len(positives)))
    if len(pool) < args.check_n:
        pool += check_rng.sample(negatives, min(args.check_n - len(pool), len(negatives)))
    check_dir = VISUALIZATIONS_DIR / "calculus_conversion_check"
    if check_dir.exists():
        shutil.rmtree(check_dir)
    check_dir.mkdir(parents=True, exist_ok=True)

    ious = []
    per_file = []
    for row in pool:
        msk_png = Path(row["mask_original_path"])
        msg = cv2.imread(str(msk_png), cv2.IMREAD_UNCHANGED)
        lab = Path(row["label_path"]).read_text().strip()
        rebuild = rasterize_lines(lab.split("\n") if lab else [], msg.shape)
        inter = int(((msg > 0) & (rebuild > 0)).sum())
        union = int(((msg > 0) | (rebuild > 0)).sum())
        iou = inter / union if union else 1.0
        ious.append(iou)
        per_file.append({"image_id": row["image_id"], "split": row["split"],
                         "iou": round(iou, 6)})
        img = cv2.imread(row["image_path"])
        vis = np.hstack([
            img,
            cv2.cvtColor((msg > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(rebuild.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR),
        ])
        cv2.imwrite(str(check_dir / f"{row['image_id']}.png"), vis)
    iou_arr = np.array(ious)
    logger.info(f"conversion IoU: mean={iou_arr.mean():.4f} min={iou_arr.min():.4f} "
                f"median={np.median(iou_arr):.4f}")
    (out / "conversion_check.csv").write_text(
        pd.DataFrame(per_file).to_csv(index=False))

    pos_areas = np.array(audit["lesion_areas"])
    stat = {
        "dataset": "oral_mamba",
        "calculus_pixel_value": CALCULUS_VALUE,
        "matched": audit["matched"],
        "positive_images": audit["positive_images"],
        "negative_images": audit["negative_images"],
        "positive_ratio": round(audit["positive_images"] / audit["matched"], 4),
        "negative_ratio": round(audit["negative_images"] / audit["matched"], 4),
        "mean_lesion_area_px": float(pos_areas.mean()) if len(pos_areas) else None,
        "median_lesion_area_px": float(np.median(pos_areas)) if len(pos_areas) else None,
        "mean_lesion_area_ratio": float(np.mean(audit["lesion_ratio"])),
        "median_lesion_area_ratio": float(np.median(audit["lesion_ratio"])),
        "min_lesion_area": int(pos_areas.min()) if len(pos_areas) else None,
        "max_lesion_area": int(pos_areas.max()) if len(pos_areas) else None,
        "tiny_lesions_lt_30px": int((pos_areas < 30).sum()),
        "mean_n_components": float(np.mean(audit["n_components"])),
        "max_n_components": int(np.max(audit["n_components"])),
        "anomaly_values": audit["anomalies"][:10],
        "anomaly_count": len(audit["anomalies"]),
        "shape_mismatch": audit["shape_mismatch"][:10],
        "shape_mismatch_count": len(audit["shape_mismatch"]),
        "unreadable": audit["unreadable"],
        "empty_label_after_polygon": audit["empty_mask_files"][:20],
        "empty_label_count": audit["empty_label_count"],
        "split_counts": {k: len(v) for k, v in splits.items()},
        "conversion_check": {
            "n": len(ious),
            "mean_iou": round(float(iou_arr.mean()), 4),
            "median_iou": round(float(np.median(iou_arr)), 4),
            "min_iou": round(float(iou_arr.min()), 4),
            "below_0.95_count": int((iou_arr < 0.95).sum()),
        },
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "calculus_dataset_statistics.json").write_text(
        json.dumps(stat, indent=2, ensure_ascii=False))
    logger.info(f"done. stats -> {REPORTS_DIR / 'calculus_dataset_statistics.json'}")


if __name__ == "__main__":
    main()
