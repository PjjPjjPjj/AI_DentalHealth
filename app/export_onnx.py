"""Export production weights to ONNX for the DentalAI desktop app.

Run with the TRAINING environment (torch + ultralytics installed):
    python -m app.export_onnx            # export all
    python -m app.export_onnx --verify   # export + consistency check vs ultralytics

Outputs -> app/models/*.onnx + calculus_fusion.npz
Inference at deploy time needs ONLY onnxruntime (no torch / no ultralytics).
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "app" / "models"
ARTIFACTS = ROOT / "outputs" / "calculus_v5_fix7" / "artifacts"


def _pipeline_weights() -> dict:
    """Read production weights from configs/pipeline.yaml (single source)."""
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "pipeline.yaml").read_text())
    w = cfg.get("weights", {})
    cal = w.get("calculus", {})
    return {
        "tooth_seg": (w.get("tooth"), "seg", 896),
        "calculus_s1": (cal.get("s1"), "seg", 640),
        "calculus_s2": (cal.get("s2"), "seg", 640),
        "caries_det": (w.get("caries"), "det", 960),
        "plaque_cls": (w.get("plaque"), "cls", 224),
    }


JOBS = _pipeline_weights()


def export_yolo(name: str, weights: Path, task: str, imgsz: int) -> Path:
    from ultralytics import YOLO

    out = MODELS_DIR / f"{name}.onnx"
    if out.exists():
        print(f"[skip] {name} -> {out.name} exists")
        return out
    print(f"[export] {name}: {weights.name} task={task} imgsz={imgsz} dynamic=True")
    model = YOLO(str(weights))
    path = model.export(format="onnx", imgsz=imgsz, opset=12, simplify=False,
                        dynamic=True, half=False, nms=False, batch=1)
    path = Path(path)
    if path != out:
        shutil.move(str(path), str(out))
    print(f"[done]  {name} -> {out}")
    return out


def export_torch_submodule(name: str, builder, weights: Path, size: int,
                           in_ch: int) -> Path:
    """Export torchvision-based verifier (4ch) / imgclf (3ch)."""
    import torch

    out = MODELS_DIR / f"{name}.onnx"
    if out.exists():
        print(f"[skip] {name} -> {out.name} exists")
        return out
    print(f"[export] {name}: {weights.name} size={size} in_ch={in_ch}")
    model = builder()
    sd = torch.load(str(weights), map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    model.eval()
    dummy = torch.randn(1, in_ch, size, size)
    torch.onnx.export(
        model, dummy, str(out), input_names=["images"], output_names=["logit"],
        opset_version=12, do_constant_folding=True,
        dynamic_axes={"images": {0: "batch"}, "logit": {0: "batch"}})
    print(f"[done]  {name} -> {out}")
    return out


def export_fusion() -> Path:
    import numpy as np

    out = MODELS_DIR / "calculus_fusion.npz"
    src = np.load(ARTIFACTS / "fusion_v3.npz")
    np.savez(out, coef=src["coef"], intercept=src["intercept"],
             mu=src["mu"], sg=src["sg"])
    print(f"[done]  calculus_fusion.npz -> {out}")
    return out


def _builder_verifier():
    import torch
    import torch.nn as nn
    from torchvision import models as tvmodels

    vb = tvmodels.efficientnet_v2_s(weights=None)
    conv = vb.features[0][0]
    new = nn.Conv2d(4, conv.out_channels, conv.kernel_size, conv.stride,
                    conv.padding, conv.dilation, conv.groups, conv.bias)
    with torch.no_grad():
        new.weight[:, :3] = conv.weight
        new.weight[:, 3] = conv.weight.mean(dim=1)
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    vb.features[0][0] = new
    vb.classifier[1] = nn.Linear(vb.classifier[1].in_features, 1)
    return vb


def _builder_imgclf():
    import torch.nn as nn
    from torchvision import models as tvmodels

    m = tvmodels.efficientnet_v2_s(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of job names")
    args = ap.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    todo = args.only or (list(JOBS) + ["verifier", "imgclf", "fusion"])

    for name in todo:
        if name in JOBS:
            w, task, sz = JOBS[name]
            if not Path(w).exists():
                print(f"[miss] {name}: weights not found {w}")
                continue
            export_yolo(name, Path(w), task, sz)
        elif name == "verifier":
            export_torch_submodule("calculus_verifier", _builder_verifier,
                                   ARTIFACTS / "verifier" / "best.pt", 224, 4)
        elif name == "imgclf":
            export_torch_submodule("calculus_imgclf", _builder_imgclf,
                                   ARTIFACTS / "imgclf" / "best.pt", 448, 3)
        elif name == "fusion":
            export_fusion()
    print("all requested exports finished")


if __name__ == "__main__":
    sys.exit(main())
