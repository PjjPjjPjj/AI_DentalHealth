"""Quick experiment: tiled Stage1 inference + tile-max imgclf on val (no retrain).

Hypothesis: 'other' domain = wide photos (408x612) with tiny lesions
(median 0.41% area). Tiling 2x2 with 20% overlap and per-tile S1 + per-tile
imgclf max should lift recall without retraining.

Outputs: outputs/calculus_v3_cls/tiled_val_experiment.json
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import json

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from scripts.v3_pipeline_lib import (DATA, OUT, RAW, bucket_of, imgclf_probs,
                                     make_imgclf, s1_pass)

WIDE_MIN = 500      # max(W,H) >= this -> tiled inference
OVERLAP = 0.2
C1 = 0.02


def tiles_for(W: int, H: int):
    tw, th = int(W * 0.6), int(H * 0.6)
    coords = []
    for yi in range(2):
        for xi in range(2):
            x0 = int(min(max(0, (W - tw) * xi), W - tw))
            y0 = int(min(max(0, (H - th) * yi), H - th))
            coords.append((x0, y0, x0 + tw, y0 + th))
    return coords


def nms_boxes(boxes: np.ndarray, confs: np.ndarray, thr=0.5):
    if len(boxes) == 0:
        return boxes, confs
    order = np.argsort(-confs)
    keep = []
    for i in order:
        b = boxes[i]
        ok = True
        for j in keep:
            bj = boxes[j]
            ix0, iy0 = max(b[0], bj[0]), max(b[1], bj[1])
            ix1, iy1 = min(b[2], bj[2]), min(b[3], bj[3])
            inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            ua = (b[2]-b[0])*(b[3]-b[1]) + (bj[2]-bj[0])*(bj[3]-bj[1]) - inter
            if ua > 0 and inter/ua >= thr:
                ok = False
                break
        if ok:
            keep.append(i)
    return boxes[keep], confs[keep]


def main() -> None:
    s1 = YOLO("outputs/calculus_v3_stage1/weights/best.pt")
    clf = make_imgclf()
    man = pd.read_csv(DATA / "split_manifest_v3.csv")
    rows = man[man.split == "val"].reset_index(drop=True)

    res = {("full", d): {"gt": 0, "tp": 0, "fp": 0, "n": 0, "pos": 0}
           for d in ("caries_0", "gingivitis", "other")}
    res.update({("tiled", d): {"gt": 0, "tp": 0, "fp": 0, "n": 0, "pos": 0}
                for d in ("caries_0", "gingivitis", "other")})
    img_rows = []

    for _, r in rows.iterrows():
        img = cv2.imread(r.image_path)
        H, W = img.shape[:2]
        m = cv2.imread(str(RAW / f"{r.image_id}.png"), cv2.IMREAD_UNCHANGED)
        m = np.zeros((H, W), np.uint8) if m is None else \
            (m[:, :, 0] if m.ndim == 3 else m)
        gts = []
        calc = (m == 3)
        if calc.any():
            n, comp = cv2.connectedComponents(calc.astype(np.uint8), 8)
            for i in range(1, n):
                px = int((comp == i).sum())
                if px >= 5:
                    gts.append((comp == i).astype(np.uint8))
        wide = max(W, H) >= WIDE_MIN

        for mode in ("full", "tiled"):
            st = res[(mode, r.domain)]
            st["n"] += 1
            st["pos"] += bool(r.has_calculus)
            st["gt"] += len(gts)
            if mode == "full" or not wide:
                boxes, confs, _ = s1_pass(s1, r.image_path, c1=C1)
            else:
                all_b, all_c = [], []
                for (x0, y0, x1, y1) in tiles_for(W, H):
                    tile = img[y0:y1, x0:x1]
                    tmp = f"/tmp/opencode/_tile.jpg"
                    cv2.imwrite(tmp, tile)
                    tb, tc, _ = s1_pass(s1, tmp, c1=C1)
                    if len(tb):
                        tb = tb + np.array([x0, y0, x0, y0], np.float32)
                        all_b.append(tb)
                        all_c.append(tc)
                if all_b:
                    boxes, confs = nms_boxes(np.concatenate(all_b),
                                             np.concatenate(all_c), 0.5)
                else:
                    boxes, confs = np.zeros((0, 4)), np.zeros(0)
            used = set()
            for b, cf in zip(boxes, confs):
                P = np.zeros((H, W), np.uint8)
                x0, y0 = int(b[0]), int(b[1])
                x1, y1 = int(b[2]), int(b[3])
                P[max(0,y0):y1, max(0,x0):x1] = 1  # box-level matching
                best = (0.0, None)
                for gi, G in enumerate(gts):
                    inter = int(P[G > 0].sum())
                    uni = int(((P > 0) | (G > 0)).sum())
                    io = inter / uni if uni else 0.0
                    if io > best[0]:
                        best = (io, gi)
                if best[0] >= 0.5 and best[1] not in used:
                    used.add(best[1])
                    st["tp"] += 1
                elif best[0] < 0.5:
                    st["fp"] += 1

        # imgclf full vs tile-max
        full_p = float(imgclf_probs(clf, [img])[0])
        tile_p = []
        if wide:
            tlist = []
            for (x0, y0, x1, y1) in tiles_for(W, H):
                tlist.append(img[y0:y1, x0:x1])
            tile_p = imgclf_probs(clf, tlist).tolist()
        img_rows.append({"y": int(r.has_calculus), "domain": r.domain,
                         "full": full_p,
                         "tilemax": float(max([full_p] + tile_p))})

    out = {"s1": {}, "imgclf": {}}
    for (mode, dom), st in res.items():
        out["s1"][f"{mode}|{dom}"] = {
            "recall": round(st["tp"] / max(st["gt"], 1), 3),
            "fp_img": round(st["fp"] / max(st["n"], 1), 2),
            "n": st["n"]}
    from sklearn.metrics import roc_auc_score
    y = np.array([r["y"] for r in img_rows])
    for key in ("full", "tilemax"):
        sc = np.array([r[key] for r in img_rows])
        accs = [(float(((sc >= t).astype(int) == y).mean()), float(t))
                for t in np.arange(0.05, 0.96, 0.01)]
        acc, thr = max(accs)
        out["imgclf"][key] = {
            "auroc": round(float(roc_auc_score(y, sc)), 4),
            "best_acc": round(acc, 4), "thr": round(thr, 2)}
    # per-domain tilemax accuracy
    for dom in ("caries_0", "gingivitis", "other"):
        sub = [r for r in img_rows if r["domain"] == dom]
        yd = np.array([r["y"] for r in sub])
        sd = np.array([r["tilemax"] for r in sub])
        accs = [float(((sd >= t).astype(int) == yd).mean())
                for t in np.arange(0.05, 0.96, 0.01)]
        out["imgclf"][f"tilemax|{dom}_acc"] = round(max(accs), 4)
    (OUT / "tiled_val_experiment.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
