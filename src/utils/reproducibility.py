"""Reproducibility utilities: fixed seeds and environment report."""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import datetime
import json
import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Fix random, numpy and torch seeds. torch.cuda seeded via torch if available."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:
        return "NA"


def collect_environment() -> dict:
    """Collect python/torch/cuda/gpu information, tolerant of missing pieces."""
    env = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        env["torch_version"] = torch.__version__
        env["torch_version_cuda"] = str(torch.version.cuda)
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["cudnn_version"] = str(torch.backends.cudnn.version())
        if env["cuda_available"]:
            env["device_count"] = torch.cuda.device_count()
            env["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            vram = []
            for i in range(torch.cuda.device_count()):
                vram.append(torch.cuda.get_device_properties(i).total_memory // (1024 ** 2))
            env["gpu_vram_mb"] = vram
        else:
            env["device_count"] = 0
            env["gpu_names"] = []
            env["gpu_vram_mb"] = []
    except ImportError:
        env["torch_version"] = "NA"
        env["torch_version_cuda"] = "NA"
        env["cuda_available"] = False
        env["cudnn_version"] = "NA"

    env["nvidia_smi"] = _run(["nvidia-smi"]) if _run(["which", "nvidia-smi"]) != "NA" else "not installed"
    return env


def save_environment_report(out_path: Path | None = None) -> Path:
    """Write outputs/reports/environment.json and return the path."""
    from .paths import REPORTS_DIR

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    target = out_path or (REPORTS_DIR / "environment.json")
    report = collect_environment()
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return target
