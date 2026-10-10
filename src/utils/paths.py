"""Centralized path management for OralAI-V1.

All modules resolve absolute paths through this module so that paths are
never hardcoded scattered in scripts. Paths are resolved relative to the
project root (the directory containing this file's parent).
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIGS_DIR = PROJECT_ROOT / "configs"
DATASETS_DIR = PROJECT_ROOT / "datasets"
RAW_DIR = DATASETS_DIR / "raw"
PROCESSED_DIR = DATASETS_DIR / "processed"
METADATA_DIR = DATASETS_DIR / "metadata"
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
VISUALIZATIONS_DIR = OUTPUTS_DIR / "visualizations"
LOGS_DIR = OUTPUTS_DIR / "logs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
DEMO_DIR = OUTPUTS_DIR / "demo"
DEMO_IMAGES_DIR = PROJECT_ROOT / "demo_images"
ANNOTATION_QUEUE_DIR = OUTPUTS_DIR / "annotation_queue"
EXPERIMENTS_CSV = OUTPUTS_DIR / "experiments.csv"


def ensure_output_dirs() -> None:
    """Ensure all write-target directories exist."""
    for d in [
        CHECKPOINTS_DIR, METRICS_DIR, PREDICTIONS_DIR, VISUALIZATIONS_DIR,
        LOGS_DIR, REPORTS_DIR, DEMO_DIR, ANNOTATION_QUEUE_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
    for task in ["caries", "calculus", "plaque", "tooth"]:
        (CHECKPOINTS_DIR / task).mkdir(parents=True, exist_ok=True)
        (METRICS_DIR / task).mkdir(parents=True, exist_ok=True)
        (VISUALIZATIONS_DIR / task).mkdir(parents=True, exist_ok=True)
    for task in ["caries", "calculus", "plaque", "tooth"]:
        (PREDICTIONS_DIR / task).mkdir(parents=True, exist_ok=True)


def task_checkpoint_dir(task: str) -> Path:
    d = CHECKPOINTS_DIR / task
    d.mkdir(parents=True, exist_ok=True)
    return d


def task_metrics_dir(task: str) -> Path:
    d = METRICS_DIR / task
    d.mkdir(parents=True, exist_ok=True)
    return d


def task_predictions_dir(task: str) -> Path:
    d = PREDICTIONS_DIR / task
    d.mkdir(parents=True, exist_ok=True)
    return d


def task_visualizations_dir(task: str) -> Path:
    d = VISUALIZATIONS_DIR / task
    d.mkdir(parents=True, exist_ok=True)
    return d


def processed_dataset_path(name: str) -> Path:
    d = PROCESSED_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d
