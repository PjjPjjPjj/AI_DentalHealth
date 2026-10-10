"""V3 verifier training: EfficientNetV2-S, 4ch (RGB + S2 mask), BCE.

Balanced sampler (2*n_pos/epoch), head 2ep @1.5e-3 -> full @1e-4,
PR-AUC early stop. Output: outputs/calculus_v3_cls/verifier/
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (average_precision_score,
                             precision_recall_fscore_support, roc_auc_score)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models
from torchvision.transforms import functional as TF

V3 = Path("datasets/processed/calculus_v3_verifier")
OUT = Path("outputs/calculus_v3_cls")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
BS = 64
EPOCHS = 20


class VRDataset(Dataset):
    def __init__(self, meta: pd.DataFrame, split: str, train: bool):
        self.rows = meta[meta.split == split].reset_index(drop=True)
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        lab = "positive" if r.label == 1 else "negative"
        sub = V3 / r.split / lab
        img = cv2.cvtColor(cv2.imread(str(sub / f"{r.sample_id}.png")),
                           cv2.COLOR_BGR2RGB)
        msk = (cv2.imread(str(sub / f"{r.sample_id}_s2mask.png"),
                          cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        img = TF.to_tensor(Image.fromarray(img))
        msk = TF.to_tensor(Image.fromarray(msk * 255))
        if self.train:
            rng = np.random.default_rng(int(torch.randint(0, 1 << 30, (1,)).item()))
            ang = float(rng.uniform(-8, 8))
            tr = (float(0.05 * rng.uniform(-1, 1)), float(0.05 * rng.uniform(-1, 1)))
            sc = float(rng.uniform(0.9, 1.1))
            img = TF.affine(img, ang, tr, sc, 0, TF.InterpolationMode.BILINEAR, 0.0)
            msk = TF.affine(msk, ang, tr, sc, 0, TF.InterpolationMode.NEAREST, 0.0)
            if rng.random() < 0.5:
                img = TF.hflip(img); msk = TF.hflip(msk)
            if rng.random() < 0.1:
                img = TF.adjust_brightness(img, float(rng.uniform(0.9, 1.1)))
            if rng.random() < 0.1:
                img = TF.gaussian_blur(img, 3, sigma=float(rng.uniform(0.1, 0.5)))
        img = TF.resize(img, (224, 224), TF.InterpolationMode.BILINEAR)
        msk = TF.resize(msk, (224, 224), TF.InterpolationMode.NEAREST)
        img = TF.normalize(img, MEAN, STD)
        x = torch.cat([img, (msk > 0.5).float()], dim=0)
        return x, torch.tensor([float(r.label)]), r.sample_id


def build_model() -> nn.Module:
    m = models.efficientnet_v2_s(
        weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    conv = m.features[0][0]
    new = nn.Conv2d(4, conv.out_channels, conv.kernel_size, conv.stride,
                    conv.padding, conv.dilation, conv.groups, conv.bias)
    with torch.no_grad():
        new.weight[:, :3] = conv.weight
        new.weight[:, 3] = conv.weight.mean(dim=1)
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    m.features[0][0] = new
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    return m


def main() -> None:
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    meta = pd.read_csv(V3 / "metadata.csv")
    tr = meta[meta.split == "train"].reset_index(drop=True)
    va = meta[meta.split == "val"].reset_index(drop=True)
    dtr = VRDataset(tr, "train", True)
    dva = VRDataset(va, "val", False)
    n_pos = int((tr.label == 1).sum())
    sampler = WeightedRandomSampler(
        torch.ones(len(dtr), dtype=torch.double), num_samples=2 * n_pos,
        replacement=True)
    loader = DataLoader(dtr, batch_size=BS, sampler=sampler, num_workers=6,
                        pin_memory=True)
    vloader = DataLoader(dva, batch_size=BS, shuffle=False, num_workers=4)
    model = build_model().to(0)
    for p in model.features.parameters():
        p.requires_grad = False
    crit = nn.BCEWithLogitsLoss()
    head_opt = torch.optim.Adam(model.classifier.parameters(), lr=1.5e-3)
    val_labels = torch.tensor([float(x) for x in va.label.values])
    best_pr, no_imp = 0.0, 0
    d = OUT / "verifier"
    d.mkdir(parents=True, exist_ok=True)
    for ep in range(EPOCHS):
        if ep == 2:
            for p in model.parameters():
                p.requires_grad = True
            opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        else:
            opt = head_opt if ep < 2 else \
                torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        model.train()
        tot = 0.0
        for x, y, _ in loader:
            x, y = x.to(0), y.to(0).view(-1).float()
            opt.zero_grad()
            loss = crit(model(x).squeeze(1), y)
            loss.backward()
            opt.step()
            tot += loss.item()
        model.eval()
        scs = []
        with torch.no_grad():
            for x, y, _ in vloader:
                scs += model(x.to(0)).squeeze(1).sigmoid().cpu().tolist()
        scs = np.array(scs)
        prA = average_precision_score(val_labels, scs)
        rocA = roc_auc_score(val_labels, scs)
        prec, rec, f1, _ = precision_recall_fscore_support(
            val_labels, (scs > 0.5).astype(int), average="binary", zero_division=0)
        print(f"ep{ep + 1:02d} loss={tot / len(loader):.4f} prAUC={prA:.4f} "
              f"rocAUC={rocA:.4f} P@.5={prec:.3f} R@.5={rec:.3f} F1={f1:.3f}",
              flush=True)
        torch.save(model.state_dict(), d / "last.pt")
        if prA > best_pr:
            best_pr, no_imp = prA, 0
            torch.save(model.state_dict(), d / "best.pt")
        else:
            no_imp += 1
        if no_imp >= 6:
            break
    model.load_state_dict(torch.load(d / "best.pt", weights_only=False))
    model.eval()
    scs = []
    with torch.no_grad():
        for x, y, _ in vloader:
            scs += model(x.to(0)).squeeze(1).sigmoid().cpu().tolist()
    pd.DataFrame({"y": val_labels.numpy(), "score": np.array(scs)}).to_csv(
        d / "val_scores.csv", index=False)
    print(f"BEST PR-AUC={best_pr:.4f} pos_ratio={float(val_labels.mean()):.4f}")


if __name__ == "__main__":
    main()
