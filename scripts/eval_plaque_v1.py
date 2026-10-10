"""Plaque V1 evaluation: val conf sweep -> frozen test, calculus-style metrics.

Instance P/R/F1 (IoU>=0.5 vs px2 components), buckets, fp_on_other (px1/3/4),
image AUROC (has_plaque = any px2).  Outputs: outputs/plaque_v1/eval_plaque.json
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from ultralytics import YOLO

from scripts.v3_pipeline_lib import RAW, bucket_of

OUT = Path("outputs/plaque_v1")
DATA = Path("datasets/processed/calculus_seg_v3")
IOU = 0.5
MIN_PX = 5


def sweep(model, rows, imgsz=640):
    tp_c, fp_c, n_gt = [], [], 0
    per_img = []
    ys, scs = [], []
    for r in rows.itertuples():
        img = cv2.imread(r.image_path)
        H, W = img.shape[:2]
        m = cv2.imread(str(RAW / f"{r.image_id}.png"), cv2.IMREAD_UNCHANGED)
        if m is None:
            m = np.zeros((H, W), np.uint8)
        if m.ndim == 3:
            m = m[:, :, 0]
        gts = []
        cm = (m == 2).astype(np.uint8)
        if cm.any():
            n, comp = cv2.connectedComponents(cm, 8)
            for i in range(1, n):
                px = int((comp == i).sum())
                if px >= MIN_PX:
                    G = (comp == i).astype(np.uint8)
                    gts.append({"mask": G,
                                "bucket": bucket_of(px / float(H * W))})
        n_gt += len(gts)
        res = model.predict(r.image_path, conf=0.01, iou=0.6, imgsz=imgsz,
                            device=0, retina_masks=True, max_det=150,
                            verbose=False)[0]
        det = []
        mk = res.masks.data.cpu().numpy() if res.masks is not None else None
        if res.boxes is not None and len(res.boxes) and mk is not None:
            clss = res.boxes.cls.cpu().numpy().astype(int)
            for i in np.where(clss == 0)[0]:
                P = np.zeros((H, W), np.uint8)
                P[mk[i] > 0.5] = 1
                det.append((float(res.boxes.conf[i]), P))
        det.sort(key=lambda t: -t[0])
        used = set()
        preds, matched_info = [], []
        for c, P in det:
            best, bi = 0.0, None
            for gi, G in enumerate(gts):
                if gi in used:
                    continue
                inter = int((P & G["mask"]).sum())
                uni = int((P | G["mask"]).sum())
                io = inter / uni if uni else 0.0
                if io > best:
                    best, bi = io, gi
            if best >= IOU and bi is not None:
                used.add(bi)
                tp_c.append(c)
                matched_info.append((bi, P))
            else:
                fp_c.append(c)
            preds.append((c, P))
        # image-level: max cls0 conf; label: any px2
        ys.append(int(bool(gts)))
        scs.append(float(max([d[0] for d in det], default=0.0)))
        other = np.isin(m, [1, 3, 4]).astype(np.uint8)
        per_img.append({"gts": gts, "preds": preds, "matched": matched_info,
                        "H": H, "W": W, "other": other})
    return np.array(tp_c), np.array(fp_c), n_gt, per_img, np.array(ys), np.array(scs)


def main() -> None:
    model = YOLO("outputs/plaque_v1/weights/best.pt")
    man = pd.read_csv(DATA / "split_manifest_v3.csv")
    out = {}
    best_conf = None
    for split in ("val", "test"):
        rows = man[man.split == split].reset_index(drop=True)
        tp, fp, n_gt, per_img, ys, scs = sweep(model, rows)
        # per-conf metrics recomputed from per-image structures
        def at(c):
            TP = FP = 0
            gtb = {b: 0 for b in ("small", "medium", "large")}
            tpb = {b: 0 for b in ("small", "medium", "large")}
            fp_on_other = 0
            preds_tot = 0
            for pi in per_img:
                cand = [(cf, P) for cf, P in pi["preds"] if cf >= c]
                used = set()
                for cf, P in cand:
                    preds_tot += 1
                    best, bi = 0.0, None
                    for gi, G in enumerate(pi["gts"]):
                        if gi in used:
                            continue
                        inter = int((P & G["mask"]).sum())
                        uni = int((P | G["mask"]).sum())
                        io = inter / uni if uni else 0.0
                        if io > best:
                            best, bi = io, gi
                    if best >= IOU and bi is not None:
                        used.add(bi)
                        tpb[pi["gts"][bi]["bucket"]] += 1
                    else:
                        FP += 1
                        if (P & (pi["other"] > 0)).sum() > 0.3 * max(int(P.sum()), 1):
                            fp_on_other += 1
                for gi, G in enumerate(pi["gts"]):
                    gtb[G["bucket"]] += 1
                TP += len(used)
            rec = TP / max(n_gt, 1)
            prec = TP / max(preds_tot, 1)
            return {"conf": round(float(c), 2), "P": round(prec, 4),
                    "R": round(rec, 4),
                    "F1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
                    "r_small": round(tpb["small"] / max(gtb["small"], 1), 4),
                    "r_medium": round(tpb["medium"] / max(gtb["medium"], 1), 4),
                    "r_large": round(tpb["large"] / max(gtb["large"], 1), 4),
                    "fp_on_other": fp_on_other}
        grid = [at(c) for c in np.arange(0.05, 0.91, 0.05)]
        best = max(grid, key=lambda g: g["F1"])
        out[split] = {"gt_instances": n_gt, "grid": grid, "best": best,
                      "img_auroc_maxconf": round(float(roc_auc_score(ys, scs)), 4)}
        if split == "val":
            best_conf = best["conf"]
        print(split, "gt:", n_gt, "best:", best, flush=True)
    # test at frozen val conf
    rows = man[man.split == "test"].reset_index(drop=True)
    tp, fp, n_gt, per_img, ys, scs = sweep(model, rows)

    def at_frozen(c):
        TP = FP = preds_tot = 0
        gtb = {b: 0 for b in ("small", "medium", "large")}
        tpb = {b: 0 for b in ("small", "medium", "large")}
        fp_on_other = 0
        for pi in per_img:
            used = set()
            for cf, P in pi["preds"]:
                if cf < c:
                    continue
                preds_tot += 1
                best, bi = 0.0, None
                for gi, G in enumerate(pi["gts"]):
                    if gi in used:
                        continue
                    inter = int((P & G["mask"]).sum())
                    uni = int((P | G["mask"]).sum())
                    io = inter / uni if uni else 0.0
                    if io > best:
                        best, bi = io, gi
                if best >= IOU and bi is not None:
                    used.add(bi)
                    tpb[pi["gts"][bi]["bucket"]] += 1
                else:
                    FP += 1
                    if (P & (pi["other"] > 0)).sum() > 0.3 * max(int(P.sum()), 1):
                        fp_on_other += 1
            for G in pi["gts"]:
                gtb[G["bucket"]] += 1
            TP += len(used)
        rec = TP / max(n_gt, 1)
        prec = TP / max(preds_tot, 1)
        return {"conf": round(float(c), 2), "P": round(prec, 4), "R": round(rec, 4),
                "F1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
                "r_small": round(tpb["small"] / max(gtb["small"], 1), 4),
                "r_medium": round(tpb["medium"] / max(gtb["medium"], 1), 4),
                "r_large": round(tpb["large"] / max(gtb["large"], 1), 4),
                "fp_img": round(FP / len(per_img), 3),
                "fp_on_other": fp_on_other}
    out["test_frozen"] = at_frozen(best_conf)
    (OUT / "eval_plaque.json").write_text(json.dumps(out, indent=2))
    print("FROZEN test:", json.dumps(out["test_frozen"], indent=1))


if __name__ == "__main__":
    main()
