"""EXP18: post-Stage2 verifier training (A: RGB-only, B: RGB+Stage2 mask).

EfficientNetV2-S, ImageNet pretrained. 160->224. BCEWithLogits.
Balanced 1:1 pos:neg sampler (no bucket reweighting, no hard oversampling).
A: 3-channel RGB; B: 4-channel [RGB + current Stage2 instance mask]
  (mask = THIS instance only, no S1 mask, no union), nearest resize,
  geometric augs strictly synchronized with RGB; color only on RGB.
Phases: head 2 ep @1.5e-3 -> unfreeze @1e-4; early stop max(PR-AUC);
threshold scan incl. Recall>=0.95 / >=0.90 operating points.
Outputs: outputs/calculus_cls/verifier_{A,B}/
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (average_precision_score,
                             precision_recall_fscore_support,
                             roc_auc_score)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.transforms import functional as TF

V2 = Path("datasets/processed/calculus_v2/verifier_dataset")
OUT = Path("outputs/calculus_cls")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
BS = 64
EPOCHS = 20


class VRDataset(Dataset):
    def __init__(self, meta, split, guided: bool, train: bool):
        self.rows = meta[meta.split == split].reset_index(drop=True)
        self.guided = guided
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        sp = r["split"]
        lab = "positive" if r.label == 1 else "negative"
        sub = V2 / sp / lab
        img = cv2.imread(str(sub / f"{r.sample_id}.png"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        msk = cv2.imread(str(sub / f"{r.sample_id}_s2mask.png"),
                         cv2.IMREAD_GRAYSCALE)
        msk = (msk > 127).astype(np.uint8)
        img = TF.to_tensor(cv_img2pil(img))
        msk = TF.to_tensor(cv_img2pil(msk * 255))
        if self.train and self.guided:
            rng = np.random.default_rng(int(torch.randint(0, 1 << 30, (1,)).item()))
            ang = float(rng.uniform(-8, 8))
            tr = float(rng.uniform(0.0, 0.05)), float(rng.uniform(0.0, 0.05))
            tr = (float(0.05 * rng.uniform(-1, 1)), float(0.05 * rng.uniform(-1, 1)))
            sc = float(rng.uniform(0.9, 1.1))
            img = TF.affine(img, ang, tr, sc, 0, TF.InterpolationMode.BILINEAR, 0.0)
            msk = TF.affine(msk, ang, tr, sc, 0, TF.InterpolationMode.NEAREST, 0.0)
            if rng.random() < 0.5:
                img = TF.hflip(img)
                msk = TF.hflip(msk)
        elif self.train:
            img = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomAffine(8, translate=(0.05, 0.05), scale=(0.9, 1.1)),
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.02),
                transforms.GaussianBlur(3, sigma=(0.1, 0.5)),
            ])(img)
        img = TF.resize(img, (224, 224), TF.InterpolationMode.BILINEAR)
        msk = TF.resize(msk, (224, 224), TF.InterpolationMode.NEAREST)
        if self.train:
            if float(torch.rand(1)) < 0.05:
                img = TF.gaussian_blur(img, 3, sigma=0.5)
            if float(torch.rand(1)) < 0.05:
                img = img + torch.randn_like(img) * 0.02
        img = TF.normalize(img, MEAN, STD)
        x = img
        if self.guided:
            msk = (msk > 0.5).float()
            x = torch.cat([img, msk], dim=0)
        return x, torch.tensor([float(r.label)]), r.sample_id


def cv_img2pil(arr):
    from PIL import Image
    return Image.fromarray(arr)


def build_model(guided: bool) -> nn.Module:
    m = models.efficientnet_v2_s(
        weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    if guided:
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
    mode = sys.argv[1] if len(sys.argv) > 1 else "A"
    guided = mode == "B"
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    meta = pd.read_csv(V2 / "metadata.csv")
    meta["label"] = meta.label.map({"positive": 1, "negative": 0})
    meta = meta.dropna(subset=["label"]).reset_index(drop=True)
    tr, va = meta[meta.split == "train"], meta[meta.split == "val"]
    dtr = VRDataset(tr, "train", guided, True)
    dva = VRDataset(va, "val", guided, False)
    weights = torch.tensor([1.0] * len(dtr), dtype=torch.double)
    n_pos = int((tr.label == 1).sum())
    sampler = WeightedRandomSampler(weights, num_samples=2 * n_pos, replacement=True)
    loader = DataLoader(dtr, batch_size=BS, sampler=sampler, num_workers=4,
                        pin_memory=True)
    vloader = DataLoader(dva, batch_size=BS, shuffle=False, num_workers=4)

    model = build_model(guided).to(0)
    for p in model.features.parameters():
        p.requires_grad = False
    crit = nn.BCEWithLogitsLoss()
    head_opt = torch.optim.Adam(model.classifier.parameters(), lr=1.5e-3)
    full_opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4)
    val_labels = torch.tensor([float(x) for x in va.label.values])
    best_pr, best_ep, no_imp = 0.0, 0, 0
    d = (OUT / f"verifier_{mode}")
    d.mkdir(parents=True, exist_ok=True)
    for ep in range(EPOCHS):
        if ep < 2:
            opt = head_opt
        else:
            if ep == 2:
                for p in model.parameters():
                    p.requires_grad = True
                full_opt = torch.optim.Adam(model.parameters(), lr=1e-4,
                                            weight_decay=1e-4)
            opt = full_opt
        model.train()
        t0 = time.time()
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
        print(f"[{mode}] ep{ep + 1:02d} loss={tot / len(loader):.4f} "
              f"prAUC={prA:.4f} rocAUC={rocA:.4f} P@.5={prec:.3f} "
              f"R@.5={rec:.3f} F1={f1:.3f} ({time.time() - t0:.0f}s)", flush=True)
        torch.save(model.state_dict(), d / f"last.pt")
        if prA > best_pr:
            best_pr, best_ep, no_imp = prA, ep + 1, 0
            torch.save(model.state_dict(), d / "best.pt")
        else:
            no_imp += 1
        if no_imp >= 6:
            break
    # eval best
    model.load_state_dict(torch.load(d / "best.pt", weights_only=False))
    model.eval()
    scs = []
    with torch.no_grad():
        for x, y, _ in vloader:
            scs += model(x.to(0)).squeeze(1).sigmoid().cpu().tolist()
    scs = np.array(scs)
    prA = average_precision_score(val_labels, scs)
    rocA = roc_auc_score(val_labels, scs)
    rows = []
    for t in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
              0.50, 0.60, 0.70, 0.80, 0.90):
        pred = (scs >= t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            val_labels, pred, average="binary", zero_division=0)
        rows.append({"threshold": t, "precision": round(prec, 4),
                     "recall": round(rec, 4), "f1": round(f1, 4)})
    df = pd.DataFrame(rows)
    df.to_csv(d / "scan.csv", index=False)
    np.save(d / "val_scores.npy", scs)
    pd.DataFrame({"y": val_labels.numpy(), "score": scs}).to_csv(
        d / "val_scores.csv", index=False)
    print(f"\n[{mode}] BEST(ep{best_ep}) ROC-AUC={rocA:.4f} PR-AUC={prA:.4f}")
    print(df.round(4).to_string(index=False), flush=True)
    for cand in (0.95, 0.90):
        ok = df[df.recall >= cand]
        if len(ok):
            b = ok.loc[ok.precision.idxmax()]
            print(f"recall>={cand}: thr={b['threshold']} P={b['precision']} "
                  f"R={b['recall']} F1={b['f1']}", flush=True)
    on = float(np.mean(val_labels.numpy()))
    print(f"verifier-val positive ratio={on:.4f} (random PR-AUC={on:.4f})",
          flush=True)


if __name__ == "__main__":
    main()
