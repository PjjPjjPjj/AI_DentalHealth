"""Shared YOLO training/inference helpers.

All four task training scripts use run_yolo_train(). Handles:
  - seed fixing
  - experiment logging (append outputs/experiments.csv)
  - checkpoint saving into outputs/checkpoints/<task>/
  - FP16/AMP (never BF16 by default - V100)
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
import shutil
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.paths import EXPERIMENTS_CSV, OUTPUTS_DIR, task_checkpoint_dir, task_metrics_dir
from src.utils.reproducibility import set_seed

logger = get_logger("train_common")


def resolve_batch(cfg: dict) -> int | float:
    """Batch param for current ultralytics (int or float fraction only).

    'auto' maps to 0.7 -> YOLO picks the largest batch fitting that fraction
    of GPU memory (equivalent semantics to legacy batch=auto).
    """
    b = cfg.get("batch", "auto")
    if isinstance(b, str):
        return 0.7 if b.lower() == "auto" else float(b)
    return b


def run_yolo_train(cfg: dict, data_yaml: str | None = None, override: dict | None = None) -> dict:
    """Run Ultralytics training, return experiment record dict.

    cfg: config dict (train section under cfg['train'], task/model in cfg).
    data_yaml: path to data yaml for detect/segment; for classify use cfg path.
    """
    task = cfg["task"]
    model_name = cfg.get("model", "yolo11s.pt")
    tcfg = cfg.get("train", {})
    set_seed(int(tcfg.get("seed", 42)))
    import torch
    from ultralytics import YOLO

    device = tcfg.get("device", 0)
    if isinstance(device, int) and not torch.cuda.is_available():
        device = "cpu"
        logger.warning("GPU not available - falling back to CPU.")

    resolve = dict(
        imgsz=tcfg.get("imgsz", 640),
        epochs=tcfg.get("epochs", 100),
        patience=tcfg.get("patience", 20),
        amp=tcfg.get("amp", True),
        seed=tcfg.get("seed", 42),
        batch=resolve_batch(tcfg),
        workers=tcfg.get("workers", 4),
        device=device,
        project=str(OUTPUTS_DIR / "logs" / "yolo" / task),
        name="train",
        exist_ok=True,
        pretrained=tcfg.get("pretrained", True),
        cache=tcfg.get("cache", False),
        optimizer=tcfg.get("optimizer", "auto"),
        lr0=tcfg.get("lr0", 0.01),
    )
    aug = tcfg.get("augment", {})
    defaults = {"fliplr": False, "mosaic": 1.0, "mixup": 0.0, "close_mosaic": 0}
    for k in ["hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
              "fliplr", "mosaic", "mixup", "close_mosaic"]:
        # note: erasing/blur are version-dependent; kept out of kwargs
        resolve[k] = aug.get(k, defaults.get(k, 0.0))

    model = YOLO(model_name)
    logger.info(f"YOLO task={task} model={model_name} data={data_yaml or tcfg.get('data')}")
    kwargs = {k: v for k, v in resolve.items() if v is not None}
    if data_yaml:
        kwargs["data"] = data_yaml
    results = model.train(**kwargs)

    ckpt_dir = task_checkpoint_dir(task)
    best_src = results.save_dir / "weights" / "best.pt"
    last_src = results.save_dir / "weights" / "last.pt"
    if best_src.exists():
        shutil.copy(best_src, ckpt_dir / "best.pt")
    if last_src.exists():
        shutil.copy(last_src, ckpt_dir / "last.pt")

    record = {
        "experiment_id": f"{task}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": "NA",
        "dataset_name": cfg.get("data", {}).get("yaml_path", "NA"),
        "task": task,
        "model": model_name,
        "pretrained_weights": "coco" if tcfg.get("pretrained", True) else "none",
        "imgsz": tcfg.get("imgsz", 640),
        "batch": resolve_batch(tcfg),
        "epochs": tcfg.get("epochs", 100),
        "optimizer": tcfg.get("optimizer", "auto"),
        "learning_rate": tcfg.get("lr0", 0.01),
        "seed": tcfg.get("seed", 42),
        "amp": tcfg.get("amp", True),
        "checkpoint_path": str(ckpt_dir / "best.pt"),
        "config_path": cfg.get("_config_path", "NA"),
        "notes": "",
    }
    return record


def append_experiment(record: dict) -> None:
    """Append a row to outputs/experiments.csv, never dropping columns."""
    import pandas as pd

    columns = [
        "experiment_id", "timestamp", "git_commit", "dataset_name", "dataset_version",
        "split_version", "task", "model", "pretrained_weights", "imgsz", "batch",
        "epochs", "patience", "optimizer", "learning_rate", "seed", "amp",
        "train_samples", "val_samples", "test_samples", "precision", "recall", "f1",
        "map50", "map50_95", "dice", "iou", "auroc", "checkpoint_path", "config_path", "notes",
    ]
    row = {c: "NA" for c in columns}
    row.update({k: v for k, v in record.items() if k in columns})
    if EXPERIMENTS_CSV.exists():
        df = pd.read_csv(EXPERIMENTS_CSV)
    else:
        df = pd.DataFrame(columns=columns)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(EXPERIMENTS_CSV, index=False)


def save_metrics_json(task: str, metrics: dict) -> Path:
    p = task_metrics_dir(task) / "metrics.json"
    p.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    return p
