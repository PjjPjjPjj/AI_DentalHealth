"""Build SANO negative tooth crops for plaque specificity (plaque_bin_v2).

Pipeline (matches inference-time protocol, verify3_plaque_integration.py):
  tooth bbox from iop_toothseg (conf .25 iou .5 @896) -> direct bbox crop,
  min side 8 px -> P(plaque) from plaque_bin_v1 best.pt.

Label policy:
  - SANO folder-level label = healthy (audit-confirmed: all 225 imgs clean)
  - eval crops : keep ALL (honest evaluation); model-P logged for review
  - train crops: drop P(plaque) >= 0.99 (suspected visible plaque/stain),
                 keep the rest incl. hard negatives (P in [0.5, 0.99))

Output:
  datasets/processed/plaque_sano/{train,eval}/*.jpg + sano_crops.csv
  annotation_queue/sano_eval_review/  (contact sheets w/ P overlay)
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
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

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

ROOT = Path(f"{_ROOT}")
RAW = ROOT / "datasets/raw/Plaque/SANO"
OUT = ROOT / "datasets/processed/plaque_sano"
TOOTH_W = ROOT / "outputs/iop_toothseg/weights/best.pt"
PLAQUE_W = ROOT / "outputs/plaque_bin_v1/weights/best.pt"
MIN_SIDE = 8
TRAIN_DROP_P = 0.99


def main() -> None:
    sys.path.insert(0, str(ROOT))
    from src.inference.models import ToothModel, PlaqueModel

    split = json.loads((RAW / "split.json").read_text())
    tm = ToothModel(TOOTH_W)
    pm = PlaqueModel(PLAQUE_W)

    (OUT / "train").mkdir(parents=True, exist_ok=True)
    (OUT / "eval").mkdir(parents=True, exist_ok=True)
    rows = []

    for subset in ("train", "eval"):
        for name in split[subset]:
            fname = Path(name).name.replace(" ", "_")
            img_path = RAW / "images" / fname
            img = cv2.imread(str(img_path))
            if img is None:
                print("UNDECODABLE:", fname)
                continue
            H, W = img.shape[:2]
            teeth = tm.predict(img)
            stem = Path(fname).stem
            if not teeth:
                rows.append({"subset": subset, "image": fname, "tooth_idx": -1,
                             "crop": None, "p_plaque_v1": None, "status": "no_teeth"})
                continue
            crops, metas = [], []
            for ti, t in enumerate(teeth):
                x0, y0, x1, y1 = [int(max(0.0, v)) for v in t["bbox"]]
                x1, y1 = min(W, x1), min(H, y1)
                if x1 - x0 < MIN_SIDE or y1 - y0 < MIN_SIDE:
                    continue
                crops.append(img[y0:y1, x0:x1].copy())
                metas.append((ti, x0, y0, x1, y1, float(t["conf"])))
            if not crops:
                continue
            probs = pm._proba_array(crops)
            for (ti, x0, y0, x1, y1, tconf), crop, p in zip(metas, crops, probs):
                cid = f"sano_{stem}__t{ti:02d}"
                rows.append({"subset": subset, "image": fname, "tooth_idx": ti,
                             "crop": f"{subset}/{cid}.jpg", "p_plaque_v1": round(float(p), 4),
                             "tooth_conf": round(tconf, 3),
                             "box_w": x1 - x0, "box_h": y1 - y0,
                             "status": "kept"})
                cv2.imwrite(str(OUT / subset / f"{cid}.jpg"), crop,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"{subset} {fname}: teeth={len(teeth)} kept={len(metas)}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sano_crops.csv", index=False)

    # ---- label policy application (logical, files stay on disk) ----
    tr = df[(df.subset == "train") & (df.status == "kept")]
    ev = df[(df.subset == "eval") & (df.status == "kept")]
    tr_drop = int((tr.p_plaque_v1 >= TRAIN_DROP_P).sum())
    print(f"\ntrain crops: {len(tr)}  (drop P>={TRAIN_DROP_P}: {tr_drop} -> use {len(tr)-tr_drop})")
    print(f"eval  crops: {len(ev)}  (all kept, honest eval)")
    print(f"eval P(plaque) percentiles:",
          {q: round(float(np.percentile(ev.p_plaque_v1, q)), 3) for q in [5, 25, 50, 75, 95]})

    # ---- review contact sheets for eval subset ----
    q = ROOT / "annotation_queue/sano_eval_review"
    q.mkdir(parents=True, exist_ok=True)
    TH, COLS = 160, 12
    import math
    ev_rows = ev.sort_values("p_plaque_v1", ascending=False).reset_index(drop=True)
    for chunk_i in range(math.ceil(len(ev_rows) / 120)):
        part = ev_rows.iloc[chunk_i * 120:(chunk_i + 1) * 120]
        rows_n = math.ceil(len(part) / COLS)
        sheet = np.full((rows_n * TH, COLS * TH, 3), 30, np.uint8)
        for i, r in part.iterrows():
            im = cv2.imread(str(OUT / r.crop))
            if im is None:
                continue
            im = cv2.resize(im, (TH - 6, TH - 26))
            rr, cc = divmod(i % len(part), COLS)
            y, x = rr * TH + 2, cc * TH + 2
            sheet[y:y + im.shape[0], x:x + im.shape[1]] = im
            tag = f"P={r.p_plaque_v1:.2f}"
            cv2.putText(sheet, tag, (x, y + im.shape[0] + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0) if r.p_plaque_v1 < 0.5 else (0, 0, 255), 1)
        cv2.imwrite(str(q / f"eval_review_{chunk_i:02d}.jpg"), sheet)
    print("review sheets ->", q)


if __name__ == "__main__":
    main()
