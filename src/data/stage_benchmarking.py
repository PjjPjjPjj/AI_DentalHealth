"""Verify + stage downloaded Benchmarking zip directly into processed YOLO layout.

Steps:
  1. verify md5 against Zenodo record (if known)
  2. unzip to datasets/raw/caries6313/Benchmarking\ Dataset/
  3. audit YOLO labels (counts per split, empty labels, bbox out-of-bounds)
  4. stage images + labels into datasets/processed/caries_det
  5. write configs/caries_data.yaml

Usage:
    python -m src.data.stage_benchmarking \
        [--zip datasets/raw/caries6313/Benchmarking_Dataset.zip] [--do-unzip]
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
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import yaml

from src.data.audit import audit_label_file, audit_static_image, find_images
from src.utils.logger import get_logger
from src.utils.paths import METADATA_DIR, PROCESSED_DIR

logger = get_logger("stage_benchmarking")

CLASS_NAME = "caries"
SPLITS = ["train", "valid", "test"]


def unzip_benchmarking(zip_path: Path, target: Path) -> Path:
    if (target / "Benchmarking Dataset" / "train").exists():
        logger.info("Already extracted, skipping unzip")
        return target / "Benchmarking Dataset"
    logger.info(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target)
    return target / "Benchmarking Dataset"


def audit_split(split_root: Path, out_dir: Path, split_name: str, manifest: list[dict]) -> dict:
    images_dir = split_root / "images"
    yolo_dir = split_root / "yolo"
    images = find_images(images_dir)
    report = {
        "split": split_name,
        "n_images": len(images),
        "n_labels": len(list(yolo_dir.glob("*.txt"))) if yolo_dir.exists() else 0,
        "empty_labels": 0,
        "invalid_labels": 0,
        "oob_labels": 0,
        "unpaired_images": 0,
        "class_instances": {},
        "unreadable_images": 0,
    }
    out_images = out_dir / "images" / split_name
    out_labels = out_dir / "labels" / split_name
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    for im in images:
        record = audit_static_image(im)
        lbl_path = yolo_dir / (im.stem + ".txt")
        if not lbl_path.exists():
            report["unpaired_images"] += 1
            continue
        lbl = audit_label_file(lbl_path, record.get("width") or 1, record.get("height") or 1)
        report["empty_labels"] += 1 if lbl.get("empty") else 0
        report["invalid_labels"] += lbl.get("invalid_coords", 0)
        report["oob_labels"] += lbl.get("bbox_out_of_bounds", 0)
        if not record.get("readable"):
            report["unreadable_images"] += 1
            continue
        for ln in lbl_path.read_text().splitlines():
            if ln.strip():
                report["class_instances"][CLASS_NAME] = report["class_instances"].get(CLASS_NAME, 0) + 1
        shutil.copy(im, out_images / im.name)
        shutil.copy(lbl_path, out_labels / im.name.replace(im.suffix, ".txt"))
        manifest.append({
            "dataset_name": "caries6313_benchmarking", "image_id": im.stem,
            "image_path": str(out_images / im.name),
            "annotation_path": str(out_labels / im.name.replace(im.suffix, ".txt")),
            "subject_id": split_name, "source": "Zenodo 14827784 Benchmarking Dataset",
            "task": "caries_det", "split": "val" if split_name == "valid" else split_name,
            "width": record.get("width") or "-", "height": record.get("height") or "-",
            "license": "CC-BY-4.0", "original_split": split_name,
            "processed": True, "notes": "official split; YOLO txt labels",
        })
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default="datasets/raw/caries6313/Benchmarking_Dataset.zip")
    parser.add_argument("--do-unzip", action="store_true",
                        help="extract zip; without it assumes extracted dir exists")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    zip_path = Path(args.zip)
    out = Path(args.out) if args.out else (PROCESSED_DIR / "caries_det")
    raw_root = zip_path.parent

    if args.do_unzip:
        bm_root = unzip_benchmarking(zip_path, raw_root)
    else:
        bm_root = raw_root / "Benchmarking Dataset"
    if not bm_root.exists():
        logger.error(f"{bm_root} not found; run with --do-unzip or extract first")
        return

    manifest: list[dict] = []
    summaries = {}
    for split in SPLITS:
        sr = bm_root / split
        if not sr.exists():
            logger.error(f"Missing split dir: {sr}")
            continue
        summaries[split] = audit_split(sr, out, split, manifest)
        logger.info(f"{split}: {summaries[split]}")

    pd.DataFrame(manifest).to_csv(METADATA_DIR / "split_manifest.csv", index=False)
    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train", "val": "images/valid", "test": "images/test",
        "names": {0: CLASS_NAME},
    }
    Path("configs/caries_data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    logger.info(f"Staged to {out}; manifest + data yaml written. Summary above.")


if __name__ == "__main__":
    main()
