"""Clean Oral-Mamba raw dataset: deduplicate + select best mask per duplicate group.

Verified problems in datasets/raw/oral_mamba (audit 2026-08-28):
  - 355 duplicate-content groups (1028 files, 673 redundant)
  - 662/663 duplicate label pairs differ (same photo, different masks)
  - 10 images without labels: 146(1).jpg .. 155(1).jpg
  - calculus (mask value 3) annotations exist only in `plain`-named files

Policy (calculus-task driven, no annotation merging/fabrication):
  For each content-MD5 group keep ONE image:
    1. if any copy's mask has value-3 (calculus) pixels -> keep the copy with
       the largest value-3 area (most complete calculus annotation);
    2. else keep the copy with the largest total foreground area,
       tie-break: plain > gingivitis_ > caries_0_ name order.
  Label-less images are excluded (logged).
  Raw is never modified; output goes to datasets/processed/oral_mamba_clean/.

Output:
  datasets/processed/oral_mamba_clean/images/*.jpg
  datasets/processed/oral_mamba_clean/labels/*.png
  datasets/metadata/oral_mamba_clean_manifest.csv
  outputs/reports/oral_mamba_clean_report.json
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

import collections
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.utils.paths import METADATA_DIR, OUTPUTS_DIR, PROCESSED_DIR

logger = get_logger("oralmamba_clean")

RAW = Path("datasets/raw/oral_mamba")
IMG_DIR = RAW / "images"
LAB_DIR = RAW / "lables"
OUT = PROCESSED_DIR / "oral_mamba_clean"

NAME_PRIORITY = {"plain": 0, "gingivitis": 1, "caries": 2}


def name_cat(stem: str) -> str:
    if stem.startswith("caries_0_"):
        return "caries"
    if stem.startswith("gingivitis_"):
        return "gingivitis"
    return "plain"


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def mask_stats(p: Path) -> dict | None:
    m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        m = m[:, :, 0]
    return {int(v): int((m == v).sum()) for v in np.unique(m)}


def main() -> None:
    images = sorted(IMG_DIR.glob("*.jpg"))
    logger.info(f"raw images: {len(images)}")

    h2f: dict[str, list[Path]] = collections.defaultdict(list)
    for im in images:
        h2f[md5(im)].append(im)

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "reports").mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    kept = dropped_dup = dropped_unlabeled = 0
    mask_src_calc = mask_src_bestfg = 0
    calc_groups = 0

    for h, copies in sorted(h2f.items()):
        copies = sorted(copies)
        stats = {}
        for im in copies:
            lp = LAB_DIR / (im.stem + ".png")
            stats[im] = {"mask": lp if lp.exists() else None,
                         "ms": mask_stats(lp) if lp.exists() else None,
                         "cat": name_cat(im.stem)}

        if all(s["ms"] is None for s in stats.values()):
            dropped_unlabeled += len(copies)
            for im in copies:
                manifest.append(dict(content_md5=h, kept_file="", action="excluded_unlabeled",
                                     image=im.name, mask="", calculus_area=-1,
                                     foreground_area=-1, reason="no mask for any copy"))
            continue

        calc = [(im, s["ms"].get(3, 0)) for im, s in stats.items() if s["ms"]]
        calc = [(im, a) for im, a in calc if a is not None]
        max_calc = max((a for _, a in calc), default=0)

        if max_calc > 0:
            # keep copy with largest calculus area; tie-break by name priority
            best = sorted(calc, key=lambda t: (-t[1], NAME_PRIORITY[stats[t[0]]["cat"]]))[0][0]
            mask_rule = "max_calculus_area"
            mask_src_calc += 1
            calc_groups += 1
        else:
            cand = [(im, sum(v for k, v in s["ms"].items() if k != 0)) for im, s in stats.items() if s["ms"]]
            best = sorted(cand, key=lambda t: (-t[1], NAME_PRIORITY[stats[t[0]]["cat"]]))[0][0]
            mask_rule = "max_foreground_area"
            mask_src_bestfg += 1

        s = stats[best]
        shutil.copy(best, OUT / "images" / best.name)
        shutil.copy(s["mask"], OUT / "labels" / (best.stem + ".png"))
        kept += 1
        manifest.append(dict(content_md5=h, kept_file=best.name,
                             action="kept", image=best.name,
                             mask=s["mask"].name, calculus_area=s["ms"].get(3, 0),
                             foreground_area=sum(v for k, v in s["ms"].items() if k != 0),
                             reason=mask_rule))
        for im in copies:
            if im == best:
                continue
            act = "dropped_duplicate" if stats[im]["ms"] else "dropped_duplicate_unlabeled"
            dropped_dup += 1
            manifest.append(dict(content_md5=h, kept_file=best.name, action=act,
                                 image=im.name,
                                 mask=(stats[im]["mask"].name if stats[im]["mask"] else ""),
                                 calculus_area=(stats[im]["ms"].get(3, 0) if stats[im]["ms"] else -1),
                                 foreground_area=(sum(v for k, v in stats[im]["ms"].items() if k != 0) if stats[im]["ms"] else -1),
                                 reason=f"group kept {best.name} ({mask_rule})"))

    df = pd.DataFrame(manifest)
    df.to_csv(METADATA_DIR / "oral_mamba_clean_manifest.csv", index=False)

    # verification pass
    out_imgs = sorted((OUT / "images").glob("*.jpg"))
    out_md5 = [md5(p) for p in out_imgs]
    dup_left = len(out_md5) - len(set(out_md5))
    missing_mask = [p.name for p in out_imgs if not (OUT / "labels" / (p.stem + ".png")).exists()]

    dims_bad = 0
    empty_calc = 0
    for im in out_imgs:
        img = cv2.imread(str(im))
        m = cv2.imread(str(OUT / "labels" / (im.stem + ".png")), cv2.IMREAD_UNCHANGED)
        if img is None or m is None or m.shape[:2] != img.shape[:2]:
            dims_bad += 1
        if int((m == 3).sum()) == 0:
            empty_calc += 1

    report = {
        "raw_images": len(images),
        "unique_content_groups": len(h2f),
        "kept": kept,
        "dropped_duplicates": dropped_dup,
        "excluded_unlabeled": dropped_unlabeled,
        "mask_rule_max_calculus_groups": mask_src_calc,
        "mask_rule_max_foreground_groups": mask_src_bestfg,
        "groups_with_calculus": calc_groups,
        "verify": {"duplicate_left": dup_left, "missing_mask": len(missing_mask),
                   "dim_mismatch": dims_bad, "images_without_calculus_pixels": empty_calc,
                   "missing_mask_files": missing_mask[:15]},
    }
    json.dump(report, open(OUTPUTS_DIR / "reports" / "oral_mamba_clean_report.json", "w"),
              ensure_ascii=False, indent=1)
    logger.info(f"clean done: {json.dumps(report, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
