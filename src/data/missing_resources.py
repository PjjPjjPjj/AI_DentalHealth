"""Generate outputs/reports/missing_resources.md - the honest inventory of
what V1 requires but is not present in datasets/raw/ yet.

BLOCKED_EXTERNAL items do NOT stop the project: code, configs and tests are
written as if data exists; only actual training is paused until data lands.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.paths import RAW_DIR, REPORTS_DIR

logger = get_logger("resources")

REQUIRED = [
    {
        "dataset": "AIRC-LABDEN",
        "task": "plaque_cls",
        "needed": ["images", "region bbox labels", "patient_id in Data_splits.csv"],
        "supports": "Plaque classification training + patient-level split",
        "status": "missing",
    },
    {
        "dataset": "6313 Intraoral Caries",
        "task": "caries_det",
        "needed": ["images", "caries bbox labels (roLabelImg/VOC/CSV)"],
        "supports": "Caries detection baseline",
        "status": "missing",
    },
    {
        "dataset": "Oral-Mamba",
        "task": "calculus_seg",
        "needed": ["RGB images", "calculus masks"],
        "supports": "Calculus segmentation baseline",
        "status": "missing",
    },
    {
        "dataset": "IO150K / TeethSEG",
        "task": "tooth_seg",
        "needed": ["images", "tooth instance masks", "domain note (real RGB vs rendered)"],
        "supports": "Tooth instance segmentation baseline",
        "status": "missing",
    },
]


def main() -> None:
    missing = []
    for item in REQUIRED:
        ds_dir = RAW_DIR / item["dataset"].split("/")[0].split(" ")[0] or RAW_DIR / item["dataset"]
        present = any(ds_dir.iterdir()) if ds_dir.exists() else False
        item["present"] = present
        if not present:
            missing.append(item)

    lines = [
        "# Missing Resources (BLOCKED_EXTERNAL)", "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}", "",
        "The following items block only their own training phase. All code, "
        "configs, converters, eval and tests are implemented so that once data "
        "is placed into `datasets/raw/<dataset>/`, the pipeline can run end to end.", "",
    ]
    for item in missing:
        lines.append(f"## {item['dataset']} - {item['status']}")
        lines.append(f"- Task: `{item['task']}`")
        lines.append(f"- Needed: {', '.join(item['needed'])}")
        lines.append(f"- Supports: {item['supports']}")
        lines.append(f"- Expected location: `datasets/raw/{item['dataset'].split('/')[0]}/`")
        lines.append("")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "missing_resources.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Missing resources report -> {out} ({len(missing)} items)")


if __name__ == "__main__":
    main()
