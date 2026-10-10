"""Insert stride-4 P2 branch into the v3b Stage1 (2-class) YOLO11m-seg.

Adapted from scripts/stage1_p2_clone.py (proven working on ultralytics 8.4.129).
Backbone/neck kept; new P2 C3k2 + 4-scale Segment head; every shape-matching
weight copied (incl. cls branch since nc stays 2).
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.nn.modules.block import C3k2
from ultralytics.nn.modules.head import Segment

SRC = "outputs/calculus_v3b_stage1/weights/best.pt"
OUT = "outputs/stage1_v3b_p2_clone.pt"


def copy_shape_weights(src: nn.Module, dst: nn.Module) -> int:
    cache = {}
    for m in src.modules():
        if hasattr(m, "weight") and m.weight is not None:
            cache.setdefault(tuple(m.weight.shape), []).append(m)
        if hasattr(m, "bias") and m.bias is not None:
            cache.setdefault(tuple(m.bias.shape), []).append(m)
    n = 0
    for m in dst.modules():
        if hasattr(m, "weight") and m.weight is not None:
            sh = tuple(m.weight.shape)
            if sh in cache and cache[sh]:
                m.weight.data.copy_(cache[sh][0].weight.data)
                if hasattr(m, "bias") and m.bias is not None and \
                        tuple(m.bias.shape) in cache and cache[tuple(m.bias.shape)]:
                    m.bias.data.copy_(cache[sh][0].bias.data)
                n += 1
    return n


def build() -> str:
    m = YOLO(SRC)
    mdl = m.model
    original_layers = list(mdl.model)
    old_head = original_layers[-1]
    layers = original_layers[:-1]
    p2 = C3k2(256, 128, n=1, c3k=False, e=0.25, shortcut=False)
    p2.f = 2
    p2.i = 23
    new_head = Segment(2, 32, 256, ch=(128, 256, 512, 512))
    copied = copy_shape_weights(old_head, new_head)
    with torch.no_grad():
        new_head.stride = torch.tensor([4.0, 8.0, 16.0, 32.0])
    new_head.f = [23, 16, 19, 22]
    new_head.i = 24
    new_seq = nn.Sequential(*(layers + [p2, new_head]))
    mdl.model = new_seq
    mdl.save = sorted(set(list(mdl.save) + [2, 23]))
    mdl.names = {0: "calculus", 1: "other_lesion"}
    m.model = mdl
    mdl.args = {}
    mdl.task = "segment"
    mdl.stride = torch.tensor([4.0, 8.0, 16.0, 32.0])
    torch.save({"model": mdl, "train_args": {}, "names": mdl.names}, OUT)
    mdl.to(0).eval()
    with torch.no_grad():
        o = mdl(torch.zeros(1, 3, 640, 640).to(0))
    print(f"smoke forward OK; copied={copied}",
          [tuple(t.shape) for t in o[0]] if isinstance(o, tuple) else None)
    return OUT


if __name__ == "__main__":
    build()
