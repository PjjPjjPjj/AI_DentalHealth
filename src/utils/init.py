"""Phase 0 init: write environment report and scaffold configs.

Usage:
    python -m src.utils.init
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import datetime
import json

from .logger import get_logger
from .paths import CONFIGS_DIR, METADATA_DIR, REPORTS_DIR
from .reproducibility import save_environment_report


def run() -> None:
    logger = get_logger("init")
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report_path = save_environment_report()
    logger.info(f"Environment report written to {report_path}")

    env = json.loads(report_path.read_text())
    lines = [
        "Python version: " + env.get("python_version", "NA"),
        "PyTorch version: " + env.get("torch_version", "NA"),
        "CUDA version: " + str(env.get("torch_version_cuda", "NA")),
        "GPU names: " + " | ".join(env.get("gpu_names", []) or ["none"]),
        "GPU VRAM (MB): " + str(env.get("gpu_vram_mb", [])),
        "CUDA available: " + str(env.get("cuda_available", False)),
        "cuDNN version: " + str(env.get("cudnn_version", "NA")),
        "Timestamp: " + str(env.get("timestamp", "")),
    ]
    for line in lines:
        logger.info(line)

    (REPORTS_DIR / "phase_0_report.md").write_text(
        f"# Phase 0 Report\n\n"
        f"Completed: {datetime.datetime.now().isoformat(timespec='seconds')}\n\n"
        f"- Project scaffold created (configs, src, scripts, tests, outputs, demo_images)\n"
        f"- requirements.txt created\n"
        f"- .gitignore created\n"
        f"- Logging system: src/utils/logger.py (console + outputs/logs/oralai.log)\n"
        f"- Config loading: src/utils/config.py\n"
        f"- Reproducibility: src/utils/reproducibility.py (seed=42 default)\n"
        f"- Environment report: outputs/reports/environment.json\n\n"
        f"Environment:\n\n```text\n" + "\n".join(lines) + "\n```\n",
        encoding="utf-8",
    )
    logger.info("Phase 0 report written.")


if __name__ == "__main__":
    run()
