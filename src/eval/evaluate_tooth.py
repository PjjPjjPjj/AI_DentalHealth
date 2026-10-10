"""Evaluate tooth instance segmentation + real-RGB subset reporting.

Usage:
    python -m src.eval.evaluate_tooth --weights outputs/checkpoints/tooth/best.pt
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import argparse

from src.eval.common import run_yolo_val, write_error_analysis
from src.utils.config import resolve_config
from src.utils.logger import get_logger

logger = get_logger("eval_tooth")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tooth.yaml")
    parser.add_argument("--weights", default="outputs/checkpoints/tooth/best.pt")
    parser.add_argument("--data", default="configs/tooth_data.yaml")
    parser.add_argument("--subset", default="all", choices=["all", "real_rgb", "rendered"])
    args = parser.parse_args()

    cfg = resolve_config(args.config)
    ecfg = cfg.get("eval", {})
    metrics, _ = run_yolo_val(
        args.weights, args.data, "tooth", "segment",
        conf=ecfg.get("conf_threshold", 0.25), iou=ecfg.get("iou_threshold", 0.5),
    )
    write_error_analysis("tooth", metrics, [
        "Domain mix (real RGB vs rendered/scan) must be separated in reporting; "
        "converter marks domain=unknown when not derivable from filenames.",
        "Only class 'tooth'; FDI numbering is out of V1 scope.",
    ])
    if args.subset != "all":
        logger.info(f"Subset report {args.subset}: rerun with a subset data yaml for precise metrics.")
    logger.info("Done.")


if __name__ == "__main__":
    main()
