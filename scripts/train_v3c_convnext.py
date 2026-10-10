"""ConvNeXt-Tiny image classifier @448 (aspect-preserving pad), binary.

Output: outputs/calculus_v3_cls/convnext/{best.pt, val_scores.csv}
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

DATA = Path("datasets/processed/calculus_seg_v3")
OUT = Path("outputs/calculus_v3_cls")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
BS = 24
EPOCHS = 16
SZ = 448


def prep(img_rgb, train=False, rng=None):
    t = TF.to_tensor(Image.fromarray(img_rgb))
    if train and rng is not None:
        t = TF.affine(t, float(rng.uniform(-8, 8)),
                      (float(0.05 * rng.uniform(-1, 1)),
                       float(0.05 * rng.uniform(-1, 1))),
                      float(rng.uniform(0.9, 1.1)), 0,
                      TF.InterpolationMode.BILINEAR, 0.0)
        if rng.random() < 0.5:
            t = TF.hflip(t)
        if rng.random() < 0.3:
            t = TF.adjust_brightness(t, float(rng.uniform(0.85, 1.15)))
        if rng.random() < 0.3:
            t = TF.adjust_contrast(t, float(rng.uniform(0.85, 1.15)))
    # aspect-preserving: scale long side to SZ, pad right/bottom
    _, h, w = t.shape
    s = SZ / max(h, w)
    t = TF.resize(t, (max(1, int(h * s)), max(1, int(w * s))),
                  TF.InterpolationMode.BILINEAR)
    canvas = torch.zeros(3, SZ, SZ)
    canvas[:, :t.shape[1], :t.shape[2]] = t
    return TF.normalize(canvas, MEAN, STD)


class CnvDataset(Dataset):
    def __init__(self, meta: pd.DataFrame, split: str, train: bool):
        self.rows = meta[meta.split == split].reset_index(drop=True)
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        img = cv2.cvtColor(cv2.imread(r.image_path), cv2.COLOR_BGR2RGB)
        rng = np.random.default_rng(int(torch.randint(0, 1 << 30, (1,)).item())) \
            if self.train else None
        x = prep(img, self.train, rng)
        return x, torch.tensor([float(r.has_calculus)]), r.image_id


def main() -> None:
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    man = pd.read_csv(DATA / "split_manifest_v3.csv")
    tr = man[man.split == "train"].reset_index(drop=True)
    va = man[man.split == "val"].reset_index(drop=True)
    dtr = CnvDataset(tr, "train", True)
    dva = CnvDataset(va, "val", False)
    n_pos = int(tr.has_calculus.sum())
    dom_w = np.where(tr.domain.values == "other", 2.0, 1.0)
    base = np.where(tr.has_calculus.values > 0, 0.5, 1.0)
    sampler = WeightedRandomSampler(torch.tensor(base * dom_w, dtype=torch.double),
                                    num_samples=2 * n_pos, replacement=True)
    loader = DataLoader(dtr, batch_size=BS, sampler=sampler, num_workers=6,
                        pin_memory=True)
    vloader = DataLoader(dva, batch_size=BS, shuffle=False, num_workers=4)
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 1)
    model = model.to(0)
    crit = nn.BCEWithLogitsLoss()
    val_labels = torch.tensor([float(x) for x in va.has_calculus.values])
    best_pr, no_imp = 0.0, 0
    d = OUT / "convnext"
    d.mkdir(parents=True, exist_ok=True)
    opt = None
    for ep in range(EPOCHS):
        if ep == 0:
            for p in model.features.parameters():
                p.requires_grad = False
            opt = torch.optim.Adam(model.classifier.parameters(), lr=1.5e-3)
        if ep == 2:
            for p in model.parameters():
                p.requires_grad = True
            opt = torch.optim.Adam(model.parameters(), lr=8e-5, weight_decay=5e-4)
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
        if no_imp >= 5:
            break
    model.load_state_dict(torch.load(d / "best.pt", weights_only=False))
    model.eval()
    scs = []
    with torch.no_grad():
        for x, y, _ in vloader:
            scs += model(x.to(0)).squeeze(1).sigmoid().cpu().tolist()
    pd.DataFrame({"y": val_labels.numpy(), "score": np.array(scs)}).to_csv(
        d / "val_scores.csv", index=False)
    print(f"CONVNEXT BEST PR-AUC={best_pr:.4f}")


if __name__ == "__main__":
    main()
