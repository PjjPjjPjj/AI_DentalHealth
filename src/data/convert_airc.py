"""Convert AIRC-LABDEN -> region crop classification dataset.

AIRC-LABDEN layout (typical, verified at runtime - converter adapts):
  raw/airc_labden/
      images/                 (png/jpg intraoral images)
      labels.csv or region CSV (columns like image_path,region,label,tooth_id...)
      Data_splits.csv        (official patient-level split - PREFERRED)
      xml/json annotations   (region bboxes G/I/M/D)

Region labels: G=Gingival, I=Incisal/Occlusal, M=Mesial, D=Distal.
Classes: plaque=1, normal=0 per region.

Patient-level split: if Data_splits.csv exists with patient_id + split columns,
use OFFICIAL split. Otherwise derive patient-level split via patient_split().

Output (plan 11):
  datasets/processed/plaque_cls/{train,val,test}/{plaque,normal}/*.jpg
  datasets/metadata/plaque_crops.csv
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
import csv
import shutil
from pathlib import Path

import cv2
import pandas as pd

from src.data.split import patient_split
from src.utils.logger import get_logger
from src.utils.paths import METADATA_DIR, PROCESSED_DIR

logger = get_logger("airc_convert")

REGIONS = {"G", "I", "M", "D"}
RATIOS = (0.7, 0.15, 0.15)


def _read_region_rows(raw: Path) -> list[dict]:
    """Try to find region-level table: csv/tsv/json/xml any candidate."""
    rows: list[dict] = []
    for csv_path in list(raw.glob("*.csv")) + list(raw.rglob("*.csv")):
        if "split" in csv_path.name.lower():
            continue
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    rows.append(dict(r))
        except Exception:
            continue
    return rows


def _normalize_row(r: dict) -> dict | None:
    def pick(*names):
        for n in names:
            if n in r and r[n] not in (None, "", "nan"):
                return r[n]
        return None
    img = pick("image_path", "image", "filename", "image_name", "img")
    if not img:
        return None
    g = pick("region", "region_label", "zone")
    label = pick("label", "plaque", "status", "score")
    tooth = pick("tooth_id", "tooth", "tooth_number")
    patient = pick("patient_id", "subject_id", "case_id", "patient")
    box = pick("bbox", "box", "coords", "xyxy")
    if g and g.strip().upper() not in REGIONS:
        g = None
    return {
        "image": img, "region": str(g).strip().upper() if g else None,
        "label": str(label).strip() if label is not None else None,
        "tooth_id": tooth, "patient_id": patient, "bbox_text": box,
    }


def parse_bbox(text: str, w: int, h: int) -> tuple[int, int, int, int] | None:
    import ast, re
    try:
        vals = ast.literal_eval(text)
    except Exception:
        vals = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", text)]
    if not vals or len(vals) < 4:
        return None
    try:
        if len(vals) == 8:  # polygon -> bbox
            xs, ys = vals[::2], vals[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:
            x1, y1, x2, y2 = vals[0], vals[1], vals[2], vals[3]
        return (int(x1), int(y1), int(x2), int(y2))
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="datasets/raw/airc_labden")
    parser.add_argument("--out", default=None)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw = Path(args.raw)
    out = Path(args.out) if args.out else (PROCESSED_DIR / "plaque_cls")
    out.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in raw.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not images:
        logger.info(f"No images in {raw} - conversion skipped (BLOCKED_EXTERNAL).")
        return

    rows = _read_region_rows(raw)
    logger.info(f"Found {len(rows)} region rows, {len(images)} images")

    # Official split csv
    split_assign: dict[str, str] = {}
    split_files = [p for p in raw.rglob("*split*.csv")]
    if split_files:
        df = pd.read_csv(split_files[0])
        for _, r in df.iterrows():
            pid = getattr(r, "patient_id", None) or getattr(r, "subject_id", None) or getattr(r, "Patient", None)
            spl = getattr(r, "split", None) or getattr(r, "Split", None) or getattr(r, "Data_split", None)
            if pid is not None and spl is not None:
                split_assign[str(pid).strip()] = str(spl).strip().lower()
        logger.info(f"Official patient split loaded: {len(split_assign)} patients")

    crops: list[dict] = []
    split_rows_train, split_rows_val, split_rows_test = [], [], []
    pl = [r for r in rows if _normalize_row(r) is not None]
    patients = sorted({(r.get("patient_id") or "").strip() or f"s{i:05d}" for i, r in enumerate(pl)})

    def split_of(patient_id: str, image_stem: str) -> str:
        if split_assign:
            return split_assign.get(patient_id, "val")
        return "train"  # filled in patient_split below (needs global mapping)

    # build image->patient and image->region rows
    img_patients: dict[str, str] = {}
    for r in pl:
        n = _normalize_row(r)
        pid = (n["patient_id"] or "").strip() or "unknown"
        img_patients[n["image"]] = pid

    label_map = {"1": 1, "plaque": 1, "yes": 1, "true": 1, "observed": 1,
                 "0": 0, "normal": 0, "no": 0, "false": 0, "absent": 0}

    for r in pl:
        n = _normalize_row(r)
        if n["region"] is None or n["label"] is None or n["bbox_text"] is None:
            continue
        label = label_map.get(str(n["label"]).lower(), None)
        if label is None:
            label = 0 if str(n["label"]).lower().startswith(("norm", "0", "no")) else 1
        # locate image
        candidates = [im for im in images if n["image"] in (im.name, im.stem, str(im).replace(str(raw), "").strip("/"))]
        if not candidates:
            continue
        im = candidates[0]
        img = cv2.imread(str(im))
        if img is None:
            continue
        h, w = img.shape[:2]
        bbox = parse_bbox(n["bbox_text"], w, h)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        crop_name = f"{im.stem}_{n['region']}_{n['tooth_id'] or 'T'}{len(crops):04d}.jpg"
        label_dir = "plaque" if label == 1 else "normal"
        split = split_of(n["patient_id"] or "unknown", im.stem)
        crops.append({
            "crop_path": crop_name, "patient_id": n["patient_id"] or "unknown",
            "image_id": im.stem, "tooth_id": n["tooth_id"] or "NA", "region": n["region"],
            "label": label, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "split": split,
            "image_rel": str(im),
        })

    if not split_assign and crops:
        # derive patient-level split
        sids = [c["image_id"] + f"_{i}" for i, c in enumerate(crops)]
        pids = [c["patient_id"] for c in crops]
        tr, va, te = patient_split(sids, pids, ratios=RATIOS, seed=args.seed)
        for i, c in enumerate(crops):
            if sids[i] in tr:
                c["split"] = "train"
            elif sids[i] in va:
                c["split"] = "val"
            else:
                c["split"] = "test"

    n_cropped = 0
    for c in crops:
        split, label = c["split"], c["label"]
        dest = out / split / ("plaque" if label == 1 else "normal") / c["crop_path"]
        src = Path(c["image_rel"])
        if not src.exists():
            continue
        img = cv2.imread(str(src))
        if img is None:
            continue
        crop = img[c["y1"]:c["y2"], c["x1"]:c["x2"]]
        if crop.size == 0:
            continue
        dst_dir = dest.parent
        dst_dir.mkdir(parents=True, exist_ok=True)
        # resize to classification square for stable cls training
        crop = cv2.resize(crop, (args.imgsz, args.imgsz), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dest), crop)
        c["crop_path"] = str(dest)
        n_cropped += 1

    meta = pd.DataFrame(crops)
    meta.to_csv(METADATA_DIR / "plaque_crops.csv", index=False)
    logger.info(f"Wrote {n_cropped} crops -> {out}; manifest -> datasets/metadata/plaque_crops.csv")
    logger.info(f"Label distribution: {meta['label'].value_counts().to_dict() if len(meta) else {}}")


if __name__ == "__main__":
    main()
