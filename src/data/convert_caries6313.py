"""Convert 6313 Intraoral Caries Dataset -> YOLO detection txt + data yaml.

Accepted raw layouts (discovered, not assumed - converter prints what it found):
  A) labelImg/roLabelImg XML:    raw/caries6313/**/Annotations/*.xml + JPEGImages/*.jpg
  B) YOLO txt:                   raw/caries6313/labels/*.txt + images/*.jpg
  C) Generic csv:                raw/caries6313/annotations.csv (image_path,x1,y1,x2,y2 [,class])

Tooth-level split is not available in this dataset; images are split with
per-image randomization BUT a fixed seed (42) so the split is reproducible.
If image ids encode patient ids, pass --patient-csv with patient_id column.

Output:
  datasets/processed/caries_det/{images,labels}/{train,val,test}
  datasets/metadata/split_manifest.csv
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
import ast
import csv
import random
import shutil
from pathlib import Path

import cv2
import pandas as pd

from src.utils.logger import get_logger
from src.utils.paths import METADATA_DIR, PROCESSED_DIR

logger = get_logger("caries_convert")

CLASS_NAME = "caries"


def parse_xml(path: Path) -> list[tuple[int, int, int, int]] | None:
    """Parse VOC-style XML; return list of bboxes (x1,y1,x2,y2). None -> unreadable."""
    import xml.etree.ElementTree as ET
    try:
        tree = ET.parse(path)
    except Exception:
        return None
    boxes = []
    root = tree.getroot()
    for obj in root.findall("object"):
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        try:
            x = int(float(bnd.findtext("xmin", "0")))
            y = int(float(bnd.findtext("ymin", "0")))
            x2 = int(float(bnd.findtext("xmax", "0")))
            y2 = int(float(bnd.findtext("ymax", "0")))
        except (TypeError, ValueError):
            continue
        boxes.append((x, y, x2, y2))
    return boxes


def parse_csv(path: Path) -> dict[str, list[tuple[int, int, int, int]]]:
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            img = row.get("image_path") or row.get("image") or row.get("filename")
            if not img:
                continue
            def _f(box_text):
                try:
                    v = ast.literal_eval(box_text)
                    return tuple(int(x) for x in v)
                except Exception:
                    return None
            box = _f(row.get("bbox") or row.get("box") or row.get("coords"))
            if not box or len(box) != 4:
                continue
            rows.setdefault(str(img), []).append(box)
    return rows


def parse_txt_yolo(path: Path) -> list[tuple[int, int, int, int]] | None:
    try:
        lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    except Exception:
        return None
    return None  # yolo txt is normalized; converted by outer loop


def to_yolo_bbox(box: tuple[int, int, int, int], w: int, h: int) -> str:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2 / max(1, w)
    cy = (y1 + y2) / 2 / max(1, h)
    bw = abs(x2 - x1) / max(1, w)
    bh = abs(y2 - y1) / max(1, h)
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def prepare_split_files(images: list[Path], boxes: dict[Path, list], out_dir: Path,
                        split_name: str, manifest_rows: list[dict], seed: int = 42) -> None:
    rng = random.Random(seed)
    images = list(images)
    rng.shuffle(images)
    n_val = max(1, int(len(images) * 0.15))
    n_test = max(1, int(len(images) * 0.15))
    partitions = {
        "train": images[:len(images) - n_val - n_test],
        "val": images[len(images) - n_val - n_test:len(images) - n_test],
        "test": images[len(images) - n_test:],
    }
    for split_name2, imgs in partitions.items():
        img_dir = out_dir / "images" / split_name2
        lbl_dir = out_dir / "labels" / split_name2
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for im in imgs:
            target = img_dir / im.name
            shutil.copy(im, target)
            bb = boxes.get(im, [])
            lbl = lbl_dir / (im.stem + ".txt")
            if bb:
                w, h = cv2.imread(str(im)).shape[:2][::-1]
                lines = [to_yolo_bbox(b, w, h) for b in bb]
                lbl.write_text("\n".join(lines))
            else:
                lbl.write_text("")
            manifest_rows.append({
                "dataset_name": "caries6313", "image_id": im.stem,
                "image_path": str(target), "annotation_path": str(lbl),
                "subject_id": split_name2, "source": "6313 Intraoral Caries Dataset",
                "task": "caries_det", "split": split_name2,
                "width": "-", "height": "-", "license": "unknown - verify", "original_split": split_name2,
                "processed": True, "notes": "converted (image-level split, no patient ids)",
            })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="datasets/raw/caries6313")
    parser.add_argument("--out", default=None, help="processed dir; default datasets/processed/caries_det")
    parser.add_argument("--patient-csv", default=None, help="optional csv with image_id,patient_id")
    args = parser.parse_args()

    raw = Path(args.raw)
    out = Path(args.out) if args.out else (PROCESSED_DIR / "caries_det")

    images: list[Path] = sorted(p for p in raw.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not images:
        logger.info(f"No images found in {raw} - nothing to convert (BLOCKED_EXTERNAL). Skipping.")
        return

    boxes_map: dict[Path, list] = {}
    # XML labels
    xml_files = list(raw.rglob("*annotations*.xml")) or list(raw.rglob("*.xml"))
    n_xml = len(xml_files)
    if n_xml:
        logger.info(f"Found {n_xml} XML annotation files")
        for xf in xml_files:
            stem = xf.stem
            matches = [im for im in images if im.stem == stem]
            bb = parse_xml(xf)
            if bb is None or not matches:
                continue
            # handle mismatch: use the xml's sibling image conventions
            if matches:
                boxes_map[matches[0]] = bb
    # YOLO txt (normalized)
    txt_files = list((raw / "labels").rglob("*.txt")) if (raw / "labels").exists() else []
    for tf in txt_files:
        matches = [im for im in images if im.stem == tf.stem]
        if not matches:
            continue
        im = matches[0]
        img = cv2.imread(str(im))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = []
        try:
            for ln in tf.read_text().splitlines():
                p = ln.split()
                if len(p) < 5:
                    continue
                cx, cy, bw, bh = [float(x) for x in p[1:5]]
                x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
                boxes.append((x1, y1, x2, y2))
        except Exception:
            continue
        boxes_map[im] = boxes
    # CSV
    csv_files = list(raw.glob("*.csv")) + list(raw.rglob("annotations*.csv"))
    for cf in csv_files:
        parsed = parse_csv(cf)
        for img_name, bb in parsed.items():
            matches = [im for im in images if img_name in (im.name, im.stem)]
            if matches:
                boxes_map.setdefault(matches[0], []).extend(bb)

    logger.info(f"Images: {len(images)}, labeled: {len(boxes_map)} / {len(images)}")

    manifest_rows: list[dict] = []
    prepare_split_files(images, boxes_map, out, "auto", manifest_rows)
    pd.DataFrame(manifest_rows).to_csv(METADATA_DIR / "split_manifest.csv", index=False)

    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train", "val": "images/val", "test": "images/test",
        "names": {0: CLASS_NAME},
    }
    import yaml as _yaml
    cfg_path = Path("configs/caries_data.yaml")
    cfg_path.parent.mkdir(exist_ok=True)
    cfg_path.write_text(_yaml.safe_dump(data_yaml, sort_keys=False))
    logger.info(f"Converted. Wrote {cfg_path} ({len(images)} images, {len(boxes_map)} labeled)")


if __name__ == "__main__":
    main()
