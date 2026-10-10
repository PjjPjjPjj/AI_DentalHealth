"""Plan B STEP3-6: EfficientNetV2-S binary classifier (calculus candidate).

Inputs: Stage1-proposal bbox-centered 160x160 crops -> resize 224.
Loss: BCEWithLogits. Balanced sampling 1:1 (negative held at 3:1
hard:easy; positive bucket 1.3:1.1:1, hard-positive 2x, no heavy repeats).
Light augmentation only. Phased fine-tune: head 2 epochs, then unfreeze at
low LR; <=20 epochs; early-stop by val PR-AUC; best checkpoint saved.
Self-eval: STEP7 threshold scan + ROC/PR AUC.
Outputs: outputs/calculus_cls/ (best.pt, scan.csv, log)
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (auc, precision_recall_curve, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

V2 = Path("datasets/processed/calculus_v2/classifier_dataset")
OUT = Path("outputs/calculus_cls")
BS = 64
EPOCHS = 20
SEED = 42
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class CropDataset(Dataset):
    def __init__(self, meta, split, train: bool, transform):
        self.rows = meta[meta.split == split].reset_index(drop=True)
        self.split = split
        self.train = train
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import cv2
        r = self.rows.iloc[i]
        cls = "positive" if r.label == 1 else "negative"
        img = cv2.imread(str(V2 / self._sp() / cls / f"{r.crop_id}.png"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img = self.transform(img)
        return img, torch.tensor([float(r.label)]), str(r.crop_id)

    def _sp(self):
        return "train_classifier" if self.split == "train" else "val_classifier"


def train_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomAffine(degrees=8, translate=(0.05, 0.05),
                                scale=(0.9, 1.1), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.GaussianBlur(3, sigma=(0.1, 0.5)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def val_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    meta = pd.read_csv(V2 / "metadata.csv")
    tr_meta = meta[meta.split == "train"]
    va_meta = meta[meta.split == "val"]
    dtr = CropDataset(tr_meta, "train", True, None)
    dtr.transform = train_transform()
    dva = CropDataset(va_meta, "val", False, None)
    dva.transform = val_transform()

    w = {}
    for _, r in tr_meta.iterrows():
        base = 1.0
        if r.label == 1:
            base = {"small": 1.3, "medium": 1.1, "large": 1.0}[r.primary_bucket]
            if r.source_type == "hard_positive":
                base *= 2.0
        else:
            base = 3.0 if r.source_type == "hard_negative" else 1.0
        w[str(r.crop_id)] = base
    weights = torch.tensor([w[str(r.crop_id)] for _, r in tr_meta.iterrows()],
                           dtype=torch.double)
    n_pos = int((tr_meta.label == 1).sum())
    sampler = WeightedRandomSampler(weights, num_samples=2 * n_pos, replacement=True)
    loader = DataLoader(dtr, batch_size=BS, sampler=sampler, num_workers=4,
                        pin_memory=True, drop_last=False)
    vloader = DataLoader(dva, batch_size=BS, shuffle=False, num_workers=4)

    model = models.efficientnet_v2_s(
        weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model = model.to(0)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.parameters():
        pass
    for p in model.classifier.parameters():
        p.requires_grad = True
    crit = nn.BCEWithLogitsLoss()
    head_opt = torch.optim.Adam(model.classifier.parameters(), lr=1.5e-3)
    full_opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    OUT.mkdir(parents=True, exist_ok=True)
    val_targets = np.array([float(x.label) for _, x in va_meta.iterrows()])
    best_pr, best_epoch, no_imp = 0.0, 0, 0
    for ep in range(EPOCHS):
        opt = head_opt if ep < 2 else full_opt
        if ep == 2:
            for p in model.parameters():
                p.requires_grad = True
            full_opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
            opt = full_opt
        model.train()
        t0 = time.time()
        tot = 0.0
        for x, y, _ in loader:
            x, y = x.to(0), y.to(0).view(-1).float()
            opt.zero_grad()
            out = model(x).squeeze(1)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            tot += loss.item()
        model.eval()
        scores = []
        with torch.no_grad():
            for x, y, _ in vloader:
                scores += model(x.to(0)).squeeze(1).cpu().tolist()
        scores = np.array(scores)
        from sklearn.metrics import average_precision_score
        pr_au = average_precision_score(val_targets, scores)
        roc_au = roc_auc_score(val_targets, scores)
        prec, rec, f1, _ = precision_recall_fscore_support(
            val_targets, (scores > 0.5).astype(int), average="binary", zero_division=0)
        print(f"ep{ep + 1:02d} loss={tot / len(loader):.4f} prAUC={pr_au:.4f} "
              f"rocAUC={roc_au:.4f} P={prec:.3f} R={rec:.3f} F1={f1:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if pr_au > best_pr:
            best_pr, best_epoch, no_imp = pr_au, ep + 1, 0
            torch.save(model.state_dict(), OUT / "best.pt")
        else:
            no_imp += 1
        if no_imp >= 6:
            print("early stop (no PR-AUC improvement)", flush=True)
            break

    # STEP7 threshold scan on classifier validation
    model.load_state_dict(torch.load(OUT / "best.pt"))
    model.eval()
    scores = []
    with torch.no_grad():
        for x, y, _ in vloader:
            scores += model(x.to(0)).squeeze(1).cpu().sigmoid().tolist()
    scores = np.array(scores)
    rows = []
    for t in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
              0.50, 0.60, 0.70, 0.80, 0.90):
        pred = (scores >= t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            val_targets, pred, average="binary", zero_division=0)
        rows.append({"threshold": t, "precision": round(prec, 4),
                     "recall": round(rec, 4), "f1": round(f1, 4)})
    from sklearn.metrics import average_precision_score
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "scan.csv", index=False)
    prAUC = average_precision_score(val_targets, scores)
    rocAUC = roc_auc_score(val_targets, scores)
    print("\nROC-AUC=%.4f PR-AUC=%.4f" % (rocAUC, prAUC))
    print("threshold scan (val_classifier):")
    print(df.to_string(index=False))
    ok = df[df.recall >= 0.90]
    if len(ok):
        best = ok.loc[ok.precision.idxmax()]
        print(f"\nrecall>=0.90 pick: thr={best['threshold']} P={best['precision']} "
              f"R={best['recall']} F1={best['f1']}", flush=True)
        with open(OUT / "chosen.json", "w") as f:
            json.dump({"threshold": float(best["threshold"]),
                       "precision": float(best["precision"]),
                       "recall": float(best["recall"]),
                       "f1": float(best["f1"]),
                       "pr_auc": float(prAUC),
                       "roc_auc": float(rocAUC)}, f, indent=2)
    (OUT / "config.json").write_text(json.dumps({
        "model": "efficientnet_v2_s", "input": "160->224", "loss": "BCE",
        "sampler": "1:1 pos:neg; neg hard:easy 3:1; pos bucket 1.3:1.1:1; "
                    "hard-pos x2; epoch size 2*pos",
        "aug": "hflip .5, rot 8, trans .05, scale .9-1.1, colorjitter mild, "
               "gaussian-blur p.1; mosaic/mixup/cutmix/copypaste OFF",
        "phases": "head-only 2 ep @1.5e-3, then unfreeze @1e-4, seed 42",
        "best_epoch": best_epoch}, indent=2))


if __name__ == "__main__":
    main()
