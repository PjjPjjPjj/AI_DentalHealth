"""Caries C1 evaluation: conf sweep on val -> frozen -> test.

Baseline for comparison: yolo11s @640 (outputs/caries_det_v2/weights).
Reports: val F1-best conf, test P/R/F1 at that conf, mAP50/50-95 both models.
Outputs: outputs/caries_c1/eval_c1.json
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

import numpy as np
from ultralytics import YOLO

DATA = "datasets/processed/caries_det_v2/data.yaml"
OUT = Path("outputs/caries_c1")


def preds_conf(model, split):
    """Returns (y_true_boxes_count, list of (conf, matched)) per image via val."""
    res = model.val(data=DATA, split=split, imgsz=960 if "c1" in str(model.ckpt_path) else 640,
                    conf=0.01, iou=0.6, device=0, verbose=False, save_json=False)
    return res


def sweep(model, split, imgsz):
    """Collect per-detection conf + tp via val with save_json? Simpler:
    use model.predict on images and match vs GT manually."""
    import cv2
    import pandas as pd
    man = pd.read_csv("datasets/processed/caries_det_v2/split_manifest.csv")
    rows = man[man.split == split]
    tp_c, fp_c, n_gt = [], [], 0
    for r in rows.itertuples():
        img = cv2.imread(r.img)
        H, W = img.shape[:2]
        gts = []
        lf = Path(r.label)
        for line in lf.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            cx, cy, w, h = map(float, p[1:5])
            gts.append((cx * W, cy * H, w * W, h * H))
        n_gt += len(gts)
        res = model.predict(str(r.img), conf=0.01, iou=0.6, imgsz=imgsz,
                            device=0, verbose=False)[0]
        det = []
        if res.boxes is not None and len(res.boxes):
            bx = res.boxes.xywh.cpu().numpy()
            cf = res.boxes.conf.cpu().numpy()
            for b, c in zip(bx, cf):
                det.append((float(c), b))
        det.sort(key=lambda t: -t[0])
        used = set()
        for c, b in det:
            best, bi = 0.0, None
            for gi, g in enumerate(gts):
                if gi in used:
                    continue
                ix0 = max(b[0]-b[2]/2, g[0]-g[2]/2); iy0 = max(b[1]-b[3]/2, g[1]-g[3]/2)
                ix1 = min(b[0]+b[2]/2, g[0]+g[2]/2); iy1 = min(b[1]+b[3]/2, g[1]+g[3]/2)
                inter = max(0, ix1-ix0) * max(0, iy1-iy0)
                ua = b[2]*b[3] + g[2]*g[3] - inter
                io = inter/ua if ua else 0
                if io > best:
                    best, bi = io, gi
            if best >= 0.5 and bi is not None:
                used.add(bi)
                tp_c.append(float(c))
            else:
                fp_c.append(float(c))
    return np.array(tp_c), np.array(fp_c), n_gt


def pr_at(tp_c, fp_c, n_gt, conf):
    tp = int((tp_c >= conf).sum())
    fp = int((fp_c >= conf).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(n_gt, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"conf": round(float(conf), 3), "P": round(prec, 4),
            "R": round(rec, 4), "F1": round(f1, 4)}


def main() -> None:
    m_new = YOLO("outputs/caries_c1/weights/best.pt")
    m_base = YOLO("outputs/caries_det_v2/weights/best.pt")
    out = {}
    for name, model, imgsz in (("baseline_s640", m_base, 640), ("c1_m960", m_new, 960)):
        tp, fp, n_gt = sweep(model, "val", imgsz)
        grid = [pr_at(tp, fp, n_gt, c) for c in np.arange(0.05, 0.91, 0.05)]
        best = max(grid, key=lambda g: g["F1"])
        out[name] = {"val_best": best, "val_grid": grid}
        print(name, "val best:", best, flush=True)
    # frozen conf from the NEW model's val -> apply both to test
    conf = out["c1_m960"]["val_best"]["conf"]
    for name, model, imgsz in (("baseline_s640", m_base, 640), ("c1_m960", m_new, 960)):
        tp, fp, n_gt = sweep(model, "test", imgsz)
        out[name]["test"] = pr_at(tp, fp, n_gt, conf)
        m = model.val(data=DATA, split="test", imgsz=imgsz, device=0, verbose=False)
        out[name]["test_mAP"] = {"mAP50": round(float(m.box.map50), 4),
                                 "mAP50-95": round(float(m.box.map), 4)}
        print(name, "test:", out[name]["test"], out[name]["test_mAP"], flush=True)
    (OUT / "eval_c1.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "val_grid"}
                      for k, v in out.items()}, indent=2))


if __name__ == "__main__":
    main()
