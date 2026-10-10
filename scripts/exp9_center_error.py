"""EXP9: Stage1 crop-center error diagnosis (VAL only, no training).

Per GT instance:
  - best Stage1 candidate = (final matched pred's parent stage1 box) if the
    GT is matched at IoU>=0.5 (group A), else the stage1 box with the largest
    bbox-IoU to the GT box (group B = unmatched / IoU<0.5).
  - cx_gt/cy_gt vs cx_pred/cy_pred -> dx, dy, euclid error, error/160.
  - inclusion = |GT mask inside the pred-centered crop| / |GT mask|,
    crop size follows pipeline (side = max(bw,bh)*1.8, min 160, clamped).

Outputs per bucket (small/medium/large) and per group (A/B):
center error stats, inclusion stats, fraction inclusion>=0.95/0.75/0.50.
Outputs: outputs/calculus_yolo11m_twostage/exp9_center_error.csv
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

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.inference.twostage import MIN_WINDOW, TwoStageSegmenter, TARGET, MARGIN
from src.utils.paths import PROCESSED_DIR

DATA = PROCESSED_DIR / "calculus_seg"
RUN = Path("outputs/calculus_yolo11m_twostage")
BUCKET_EDGES = (0.00552, 0.01565)
BUCKETS = ("small", "medium", "large")
NMS_IOU = 0.5
IOU_THR = 0.5
C1, C2 = 0.05, 0.05


def bucket_of(ratio: float) -> str:
    if ratio < BUCKET_EDGES[0]:
        return "small"
    if ratio < BUCKET_EDGES[1]:
        return "medium"
    return "large"


def box_iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua


def window_at(cx: float, cy: float, side: float, H: int, W: int):
    x0 = int(round(cx - side / 2)); y0 = int(round(cy - side / 2))
    x0 = max(0, min(x0, W - int(side))); y0 = max(0, min(y0, H - int(side)))
    x1 = min(x0 + int(side), W); y1 = min(y0 + int(side), H)
    return x0, y0, x1, y1


def collect_val() -> list[dict]:
    seg = TwoStageSegmenter("outputs/calculus_yolo11m_full/weights/best.pt",
                            "outputs/calculus_yolo11m_crop/weights/best.pt",
                            c1=C1, c2=C2, device=0, fusion="mask_nms",
                            nms_iou=NMS_IOU, score_mode="s2")
    man = pd.read_csv(DATA / "split_manifest.csv")
    rows = man[(man.split == "val") & man.has_calculus]
    out_recs = []  # per-image: {img,H,W,gts,preds,kept_parent}
    for _, r in rows.iterrows():
        img = cv2.imread(r["image_path"])
        H, W = img.shape[:2]
        boxes, confs = seg._stage1_full(r["image_path"])
        gt = cv2.imread(r["mask_original_path"], cv2.IMREAD_UNCHANGED) > 0
        gts = []
        n, comp, _, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8), 8)
        for i in range(1, n):
            if (comp == i).sum() < 5:
                continue
            m = (comp == i).astype(np.uint8)
            gx, gy, gw, gh = cv2.boundingRect(m)
            px = int(m.sum())
            gts.append({"mask": m, "box": (gx, gy, gx + gw, gy + gh),
                        "px": px, "bucket": bucket_of(px / float(H * W))})
        if not gts:
            continue
        recs = []
        for k, (b, sc1) in enumerate(zip(boxes, confs)):
            cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
            side = float(max(b[2] - b[0], b[3] - b[1]) * MARGIN)
            side = max(side, MIN_WINDOW)
            side = float(min(side, max(H, W)))
            x0, y0, x1, y1 = window_at(cx, cy, side, H, W)
            ch, cw = y1 - y0, x1 - x0
            crop = cv2.resize(img[y0:y1, x0:x1], (TARGET, TARGET),
                              interpolation=cv2.INTER_AREA)
            for inst in seg._stage2_instances(crop):
                m = cv2.resize(inst["bin640"], (cw, ch),
                               interpolation=cv2.INTER_NEAREST)
                recs.append({"x0": x0, "y0": y0, "cw": cw, "ch": ch, "bin": m,
                             "parent": k, "s1_conf": float(sc1),
                             "s2_conf": inst["s2_conf"], "score": inst["s2_conf"]})
        out_recs.append({"img": img, "H": H, "W": W, "boxes": boxes,
                         "gts": gts, "recs": recs})
    return out_recs


def main() -> None:
    data = collect_val()
    from src.inference.fusion import greedy_nms, rec_bin
    stats = {b: {"A": [], "B": []} for b in BUCKETS}
    n_missing = {"small": 0, "medium": 0, "large": 0}
    for d in data:
        H, W = d["H"], d["W"]
        keep = greedy_nms(d["recs"], "mask_nms", NMS_IOU, H, W)
        preds = [rec_bin(c, H, W) for c in keep]
        gts = d["gts"]
        pairs = []
        for pi, P in enumerate(preds):
            for gi, Gm in enumerate(gts):
                inter = int((P & Gm["mask"]).sum())
                uni = int((P | Gm["mask"]).sum())
                io = inter / uni if uni else 0.0
                if io >= IOU_THR:
                    pairs.append((io, pi, gi))
        pairs.sort(reverse=True)
        used_p, used_g, matched = set(), set(), []
        for io, pi, gi in pairs:
            if pi in used_p or gi in used_g:
                continue
            used_p.add(pi); used_g.add(gi)
            matched.append((pi, gi, io))
        match_by_gi = {gi: (io,) for _, gi, io in matched}
        # parent box idx by kept pred index
        parent_by_pi = {pi: keep[pi]["parent"] for pi in range(len(keep))}
        for gi, G in enumerate(gts):
            b = G["bucket"]
            gbox = G["box"]
            cx_gt = gbox[0] + gbox[2] / 2.0
            cy_gt = gbox[1] + gbox[3] / 2.0
            group = "A" if gi in match_by_gi else "B"
            parent = None
            if group == "A":
                pi = next(p for p, gg, _ in matched if gg == gi)
                parent = d["boxes"][parent_by_pi[pi]]
            else:
                if len(d["boxes"]) == 0:
                    n_missing[b] += 1
                    continue
                best, best_iou = None, -1.0
                for bx in d["boxes"]:
                    bxi = box_iou(gbox, (bx[0], bx[1], bx[2], bx[3]))
                    if bxi > best_iou:
                        best_iou, best = bxi, bx
                parent = best
            cx_pred = (parent[0] + parent[2]) / 2.0
            cy_pred = (parent[1] + parent[3]) / 2.0
            dx = cx_gt - cx_pred
            dy = cy_gt - cy_pred
            err = float(np.hypot(dx, dy))
            side = float(max(parent[2] - parent[0], parent[3] - parent[1]) * MARGIN)
            side = max(side, MIN_WINDOW)
            side = float(min(side, max(H, W)))
            x0, y0, x1, y1 = window_at(cx_pred, cy_pred, side, H, W)
            inc = int(G["mask"][y0:y1, x0:x1].sum()) / float(G["px"])
            stats[b][group].append({"err": err, "nerr": err / 160.0,
                                    "adx": abs(dx), "ady": abs(dy),
                                    "inc": inc, "px": G["px"]})
    rows = []
    for b in BUCKETS:
        for gp in ("A", "B"):
            v = stats[b][gp]
            if not v:
                continue
            errs = np.array([x["err"] for x in v])
            adx = np.array([x["adx"] for x in v])
            ady = np.array([x["ady"] for x in v])
            inc = np.array([x["inc"] for x in v])
            rows.append({"bucket": b, "group": f"{gp} (n={len(v)})",
                         "err_mean": round(float(errs.mean()), 2),
                         "err_med": round(float(np.median(errs)), 2),
                         "err_p25": round(float(np.percentile(errs, 25)), 2),
                         "err_p75": round(float(np.percentile(errs, 75)), 2),
                         "err_p90": round(float(np.percentile(errs, 90)), 2),
                         "abs_dx_mean": round(float(adx.mean()), 2),
                         "abs_dy_mean": round(float(ady.mean()), 2),
                         "inclusion_mean": round(float(inc.mean()), 4),
                         "inclusion_med": round(float(np.median(inc)), 4),
                         "frac_inc_ge_095": round(float((inc >= 0.95).mean()), 4),
                         "frac_inc_ge_075": round(float((inc >= 0.75).mean()), 4),
                         "frac_inc_ge_050": round(float((inc >= 0.50).mean()), 4)})
    # all-group (A+B) per bucket
    for b in BUCKETS:
        v = stats[b]["A"] + stats[b]["B"]
        if not v:
            continue
        errs = np.array([x["err"] for x in v])
        inc = np.array([x["inc"] for x in v])
        rows.append({"bucket": b, "group": f"ALL (n={len(v)})",
                     "err_mean": round(float(errs.mean()), 2),
                     "err_med": round(float(np.median(errs)), 2),
                     "err_p25": round(float(np.percentile(errs, 25)), 2),
                     "err_p75": round(float(np.percentile(errs, 75)), 2),
                     "err_p90": round(float(np.percentile(errs, 90)), 2),
                     "abs_dx_mean": round(float(np.mean([x["adx"] for x in v])), 2),
                     "abs_dy_mean": round(float(np.mean([x["ady"] for x in v])), 2),
                     "inclusion_mean": round(float(inc.mean()), 4),
                     "inclusion_med": round(float(np.median(inc)), 4),
                     "frac_inc_ge_095": round(float((inc >= 0.95).mean()), 4),
                     "frac_inc_ge_075": round(float((inc >= 0.75).mean()), 4),
                     "frac_inc_ge_050": round(float((inc >= 0.50).mean()), 4)})
    df = pd.DataFrame(rows)
    df.to_csv(RUN / "exp9_center_error.csv", index=False)
    pd.set_option("display.width", 200)
    print("EXP9: crop center error & GT inclusion, group A=final IoU>=0.5, B=Iou<0.5/unmatched")
    print(df.to_string(index=False))
    print("missing-candidate GTs (B, no stage1 box at all):", n_missing)


if __name__ == "__main__":
    main()
