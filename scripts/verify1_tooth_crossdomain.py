"""Verify 1: cross-domain qualitative check of iop_toothseg on real photos.

Samples images from caries6313 test, Oral-Mamba test, Mendeley val/test and
runs the tooth segmenter; saves overlay visualizations + per-image stats
(n teeth, mean conf, mask area fraction) to JSON for domain-gap judgement.
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
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

ROOT = Path(f"{_ROOT}")
TOOTH = ROOT / "outputs/iop_toothseg/weights/best.pt"
OUT = ROOT / "outputs/pipeline_v2_validation"
OUT.mkdir(parents=True, exist_ok=True)


def sample_images(n_per=6):
    picks = {}
    man = pd.read_csv(ROOT / "datasets/processed/calculus_seg_v3/split_manifest_v3.csv")
    om = man[(man.split == "test") & (man.has_calculus == True)]  # noqa: E712
    picks["oral_mamba"] = [str(p) for p in om.image_path.sample(n_per, random_state=7)]
    cd = ROOT / "datasets/processed/caries_det_v3"
    if (cd / "split_manifest.csv").exists():
        m2 = pd.read_csv(cd / "split_manifest.csv")
        col = "image_path" if "image_path" in m2.columns else m2.columns[0]
        sub = m2[m2.columns[m2.columns.str.contains("split")][0]] if any(m2.columns.str.contains("split")) else None
        picks["caries6313"] = [str(p) for p in m2[col].sample(n_per, random_state=7)]
    plq = pd.read_csv(ROOT / "datasets/processed/plaque_bin_v1/plaque_crops.csv")
    base = plq[(plq.split == "test") & (plq.is_aug == 0)].image_id.unique()
    rng = random.Random(7)
    sel = rng.sample(list(base), n_per)
    part = plq.drop_duplicates("image_id").set_index("image_id")
    picks["mendeley"] = []
    # reconstruct original image paths from raw dirs
    for iid in sel:
        cands = list((ROOT / "datasets/raw/Plaque").rglob(iid + ".jpg"))
        if cands:
            picks["mendeley"].append(str(cands[0]))
    return picks


def run(model, path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    H, W = img.shape[:2]
    res = model.predict(str(path), conf=0.3, iou=0.7, imgsz=896, device=0,
                        retina_masks=True, max_det=60, verbose=False)[0]
    if res.masks is None or len(res.boxes) == 0:
        return {"path": str(path), "W": W, "H": H, "n_teeth": 0,
                "mean_conf": 0.0, "mask_frac": 0.0}
    confs = res.boxes.conf.cpu().numpy()
    mk = res.masks.data.cpu().numpy() > 0.5
    frac = float(mk.sum()) / (H * W)
    vis = img.copy()
    overlay = np.zeros_like(img)
    colors = np.random.randint(60, 255, (len(confs), 3))
    for i in range(len(confs)):
        m = mk[i].astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, tuple(int(c) for c in colors[i]), 2)
        x0, y0, x1, y1 = [int(v) for v in res.boxes.xyxy[i]]
        cv2.putText(vis, f"{confs[i]:.2f}", (x0, max(12, y0 - 3)), 0, 0.45,
                    (255, 255, 255), 1)
    alpha = 0.25
    vis = cv2.addWeighted(vis, 1.0, (overlay), 0, 0)
    return {"path": str(path), "W": W, "H": H, "n_teeth": int(len(confs)),
            "mean_conf": round(float(confs.mean()), 3), "mask_frac": round(frac, 3)}


def main() -> None:
    model = YOLO(str(TOOTH))
    picks = sample_images()
    report = {}
    for dom, paths in picks.items():
        rows = []
        for p in paths:
            r = run(model, p)
            if r is None:
                continue
            rows.append(r)
            tag = f"{dom}_{Path(p).stem[:40]}"
            cv2.imwrite(str(OUT / f"tooth_{tag}.jpg"), cv2.imread(str(p)) if False else _vis_save(model, p))
        report[dom] = rows
        print(dom, [(r["n_teeth"], r["mean_conf"], r["mask_frac"]) for r in rows], flush=True)
    (OUT / "tooth_crossdomain.json").write_text(json.dumps(report, indent=2))


def _vis_save(model, p):
    img = cv2.imread(str(p))
    res = model.predict(str(p), conf=0.3, iou=0.7, imgsz=896, device=0,
                        retina_masks=True, max_det=60, verbose=False)[0]
    if res.masks is not None and len(res.boxes):
        confs = res.boxes.conf.cpu().numpy()
        mk = res.masks.data.cpu().numpy() > 0.5
        colors = np.random.randint(60, 255, (len(confs), 3))
        for i in range(len(confs)):
            m = mk[i].astype(np.uint8)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, tuple(int(c) for c in colors[i]), 2)
            x0, y0 = [int(v) for v in res.boxes.xyxy[i]][:2]
            cv2.putText(img, f"{confs[i]:.2f}", (x0, max(12, y0 - 3)), 0, 0.45, (255, 255, 255), 1)
    return img


if __name__ == "__main__":
    main()
