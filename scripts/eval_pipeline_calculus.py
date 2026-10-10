"""Fresh instance-level evaluation of the PACKAGED CalculusCascadeModel
(the one now inside the unified pipeline, fix4 config) on the 700-image test
split. Compares against the reference: final_fix4_metrics.json.

Usage: PYTHONPATH=. python scripts/eval_pipeline_calculus.py
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, f"{_ROOT}")
ROOT = Path(f"{_ROOT}")
from src.inference.models import CalculusCascadeModel  # noqa: E402
from src.utils.config import load_yaml  # noqa: E402
import scripts.v3_pipeline_lib as _lib  # noqa: E402

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

OUT = ROOT / "outputs/calculus_pipeline_eval"
OUT.mkdir(parents=True, exist_ok=True)
IOU_MATCH = 0.5


def main() -> None:
    cfg = load_yaml(ROOT / "configs/pipeline.yaml")
    model = CalculusCascadeModel(**cfg["inference"]["calculus"],
                                 **cfg["weights"]["calculus"])
    man = pd.read_csv(_lib.DATA / "split_manifest_v3.csv")
    rows = man[man.split == "test"].reset_index(drop=True)

    gtb = {b: 0 for b in _lib.BUCKETS}
    tpb = {b: 0 for b in _lib.BUCKETS}
    tp = fp = preds_tot = 0
    dices = []
    n_img = len(rows)
    per_img = []
    for ri, r in rows.iterrows():
        img = cv2.imread(r.image_path)
        H, W = img.shape[:2]
        gts, _ = _lib.image_gts(r.image_path, str(_lib.RAW / f"{r.image_id}.png"), H, W)
        dets = model.predict(r.image_path)
        preds = [d["mask"] for d in dets]
        preds_tot += len(preds)
        pairs = []
        for pi, P in enumerate(preds):
            for gi, G in enumerate(gts):
                inter = int((P & G["mask"]).sum())
                uni = int((P | G["mask"]).sum())
                io = inter / uni if uni else 0.0
                if io >= IOU_MATCH:
                    pairs.append((io, pi, gi))
        pairs.sort(reverse=True)
        used_p, used_g = set(), set()
        for io, pi, gi in pairs:
            if pi in used_p or gi in used_g:
                continue
            used_p.add(pi); used_g.add(gi)
            P, G = preds[pi], gts[gi]
            inter = int((P & G["mask"]).sum())
            dices.append(2 * inter / max(int(P.sum()) + int(G["mask"].sum()), 1))
            tpb[G["bucket"]] += 1
        for G in gts:
            gtb[G["bucket"]] += 1
        tp += len(used_g)
        fp += len(preds) - len(used_p)
        if (ri + 1) % 100 == 0:
            print(f"  {ri+1}/{n_img}", flush=True)
        per_img.append({"image_id": r.image_id, "n_pred": len(preds), "n_gt": len(gts),
                        "tp": len(used_g)})
    rec = tp / max(sum(gtb.values()), 1)
    prec = tp / max(preds_tot, 1)
    out = {
        "model": "packaged CalculusCascadeModel (fix4 config) from configs/pipeline.yaml",
        "test_n": n_img,
        "instance": {
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
            "r_small": round(tpb["small"] / max(gtb["small"], 1), 4),
            "r_medium": round(tpb["medium"] / max(gtb["medium"], 1), 4),
            "r_large": round(tpb["large"] / max(gtb["large"], 1), 4),
            "tp_dice": round(float(np.mean(dices)), 4) if dices else 0.0,
            "fp_img": round(fp / n_img, 3),
            "pred_img": round(preds_tot / n_img, 3),
            "n_gt": sum(gtb.values()),
        },
        "reference_final_fix4_cascade": {
            "precision": 0.4242, "recall": 0.4982, "f1": 0.4583,
            "r_small": 0.4367, "r_medium": 0.6, "r_large": 0.5,
            "tp_dice": 0.817, "fp_img": 0.271,
        },
    }
    (OUT / "pipeline_calculus_eval.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
