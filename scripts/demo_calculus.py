"""Calculus cascade demo: pick 4 representative test images, run the packaged
CalculusCascadeModel, save individual overlays + a 2x2 panel figure.

GT masks drawn as thin red contours for honest comparison; predictions as
orange mask overlay + fused conf.
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
OUT = ROOT / "outputs/demo/calculus_demo"
OUT.mkdir(parents=True, exist_ok=True)
from src.inference.models import CalculusCascadeModel  # noqa: E402
from src.utils.config import load_yaml  # noqa: E402
import scripts.v3_pipeline_lib as _lib  # noqa: E402

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))


def draw(img, gts, dets):
    vis = img.copy()
    for G in gts:  # GT: thin red contour
        cnts, _ = cv2.findContours(G["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 0, 255), 1)
    ov = vis.copy()
    for d in dets:  # pred: orange fill+contour
        m = (d["mask"] > 0)
        ov[m] = (0.45 * np.array([0, 165, 255]) + 0.55 * ov[m]).astype(np.uint8)
        cnts, _ = cv2.findContours(d["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, cnts, -1, (0, 140, 255), 2)
        ys, xs = np.where(m)
        if len(xs):
            cv2.putText(ov, f"{d['conf']:.2f}", (int(xs.min()), max(12, int(ys.min()) - 3)),
                        0, 0.5, (0, 140, 255), 1)
    return cv2.addWeighted(ov, 0.85, vis, 0.15, 0)


def main() -> None:
    cfg = load_yaml(ROOT / "configs/pipeline.yaml")
    model = CalculusCascadeModel(**cfg["inference"]["calculus"],
                                 **cfg["weights"]["calculus"])
    man = pd.read_csv(_lib.DATA / "split_manifest_v3.csv")
    test = man[(man.split == "test") & (man.has_calculus == True)].copy()  # noqa: E712

    picks = {}
    # 1) most instances
    picks["most_lesions"] = test.sort_values("n_calc_inst", ascending=False).iloc[0]
    # 2) two known large-lesion cases (post-fix recoveries)
    for iid in ("caries_0_4307", "caries_0_7816"):
        r = test[test.image_id == iid]
        if len(r):
            picks[f"large_{iid}"] = r.iloc[0]
    # 3) one gingivitis-domain image
    g = test[test.domain == "gingivitis"]
    if len(g):
        picks["gingivitis_domain"] = g.sort_values("n_calc_inst", ascending=False).iloc[0]

    panels = []
    report = []
    for tag, r in picks.items():
        img = cv2.imread(r.image_path)
        H, W = img.shape[:2]
        gts, _ = _lib.image_gts(r.image_path, str(_lib.RAW / f"{r.image_id}.png"), H, W)
        dets = model.predict(r.image_path)
        vis = draw(img, gts, dets)
        cv2.putText(vis, f"{r.image_id}  pred={len(dets)} GT={len(gts)}",
                    (8, 22), 0, 0.7, (255, 255, 0), 2)
        cv2.putText(vis, "red=GT  orange=prediction", (8, H - 10), 0, 0.55, (255, 255, 255), 1)
        cv2.imwrite(str(OUT / f"calculus_demo_{tag}_{r.image_id}.jpg"), vis)
        panels.append(cv2.resize(vis, (900, int(900 * H / W))))
        confs = [round(d["conf"], 3) for d in dets]
        report.append({"tag": tag, "image_id": r.image_id, "domain": r.domain,
                       "size": f"{W}x{H}", "n_pred": len(dets), "n_gt": len(gts),
                       "confs": confs})
        print(r.image_id, f"pred={len(dets)} gt={len(gts)} confs={confs}", flush=True)

    # 2x2 panel
    ph = max(p.shape[0] for p in panels)
    pw = max(p.shape[1] for p in panels)
    canvas = np.full((2 * ph + 18, 2 * pw + 18, 3), 245, np.uint8)
    for k, p in enumerate(panels):
        r, c = divmod(k, 2)
        canvas[r * (ph + 12):r * (ph + 12) + p.shape[0],
               c * (pw + 12):c * (pw + 12) + p.shape[1]] = p
    cv2.imwrite(str(OUT / "calculus_demo_panel.jpg"), canvas)
    (OUT / "calculus_demo_summary.json").write_text(json.dumps(report, indent=2))
    print("panel saved:", OUT / "calculus_demo_panel.jpg")


if __name__ == "__main__":
    main()
