"""Train plaque region classification baseline (YOLO11s-cls).

Usage:
    python -m src.train.train_plaque --config configs/plaque.yaml [--yes]
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import argparse

from src.train.common import append_experiment, run_yolo_train
from src.utils.config import resolve_config
from src.utils.logger import get_logger

logger = get_logger("train_plaque")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/plaque.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    cfg = resolve_config(args.config)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    data_root = cfg.get("data", {}).get("root", "datasets/processed/plaque_cls")

    if not args.yes:
        ans = input(f"Start plaque training on {data_root}? [y/N]: ")
        if ans.strip().lower() != "y":
            print("Aborted.")
            return

    record = run_yolo_train(cfg, data_yaml=None)
    append_experiment(record)
    logger.info(f"Training finished. Checkpoint: {record['checkpoint_path']}")


if __name__ == "__main__":
    main()
