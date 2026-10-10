"""Sweep thr_large (large-lesion fused threshold) on val AND test with the
fix4 cascade. Candidates are collected once per image (no selection), then
selection is a pure post-filter -> cheap offline sweep.

Answers: is thr_large=0.45 too high for large-lesion recall?
Output: outputs/calculus_pipeline_eval/thr_large_sweep.json
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
OUT = ROOT / "outputs/calculus_pipeline_eval"
from src.utils.config import load_yaml  # noqa: E402
import scripts.v3_pipeline_lib as _lib  # noqa: E402

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

THR_STD = 0.6
LARGE_EDGE = 0.01565
S1_MIN = 0.02
THRS = [0.45, 0.40, 0.35, 0.30, 0.25]
IOU_MATCH = 0.5


def collect_candidates(model_cfg, s1m, s2m, vb, image_path, rng_img=None):
    cfg = load_yaml(ROOT / "configs/pipeline.yaml")
    icfg = cfg["inference"]["calculus"]
    import src.inference.models as M
    m = M.CalculusCascadeModel(**icfg, **cfg["weights"]["calculus"])
    return m  # we just reuse its internals below


def image_candidates(m, image_path):
    """Replicates CalculusCascadeModel.predict up to fused scores (no selection)."""
    import cv2
    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression
    _lib = m._lib
    image = cv2.imread(str(image_path))
    H, W = image.shape[:2]
    boxes, confs, masks = _lib.s1_pass(m.s1m, str(image_path))
    feats, bins, rects, crop_local = [], [], [], []
    for (b, sc1, msk) in zip(boxes, confs, masks):
        bx = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
        side = _lib.SIDE
        bw, bh = bx[2] - bx[0], bx[3] - bx[1]
        if max(bw, bh) >= _lib.S2_ADAPT_TRIG * _lib.SIDE:
            side = int(min(max(W, H), max(_lib.SIDE, _lib.S2_ADAPT_MULT * max(bw, bh))))
        x0 = int(round((bx[0] + bx[2]) / 2 - side / 2))
        y0 = int(round((bx[1] + bx[3]) / 2 - side / 2))
        x0 = max(0, min(x0, W - side)); y0 = max(0, min(y0, H - side))
        x1, y1 = min(x0 + side, W), min(y0 + side, H)
        crop_img = image[y0:y1, x0:x1]
        crop = cv2.resize(crop_img, (_lib.TARGET, _lib.TARGET), interpolation=cv2.INTER_AREA)
        insts = _lib.s2_instances(m.s2m, crop)
        bw_, bh_ = bx[2] - bx[0], bx[3] - bx[1]
        saturated2 = (x1 - x0) * (y1 - y0) >= 0.9 * W * H
        rel2 = max(bw_, bh_) / max(x1 - x0, y1 - y0)
        do_tta = saturated2 or rel2 >= _lib.S2_TTA_TRIG
        if do_tta:
            flipped = _lib.s2_instances(m.s2m, crop[:, ::-1])
            for it in flipped:
                it["bin640"] = np.ascontiguousarray(it["bin640"][:, ::-1])
            insts = insts + flipped
        sel = [{"x0": 0, "y0": 0, "cw": _lib.TARGET, "ch": _lib.TARGET,
                "bin": i["bin640"], "s2_conf": i["s2_conf"],
                "score": i["s2_conf"]} for i in insts]
        keep = _lib.greedy_nms(sel, "mask_nms", _lib.NMS_IOU, _lib.TARGET, _lib.TARGET)
        if do_tta:
            keep = _lib.tta_fuse(keep, sel, 0.6)
        rectl = {"x0": x0, "y0": y0, "cw": x1 - x0, "ch": y1 - y0}
        for k in keep:
            P = k["bin"]
            m160 = (cv2.resize(P, (_lib.SIDE, _lib.SIDE), interpolation=cv2.INTER_NEAREST) > 0)
            s1m_ = (cv2.resize(msk[y0:y1, x0:x1], (_lib.SIDE, _lib.SIDE),
                               interpolation=cv2.INTER_AREA) > 0.5)
            gx, gy, gw, gh = cv2.boundingRect(P.astype(np.uint8))
            feats.append({
                "s1_conf": float(sc1), "s2_conf": float(k["s2_conf"]),
                "s1_area_ratio": float(s1m_.sum()) / (_lib.SIDE * _lib.SIDE),
                "s2_area_ratio": float(m160.sum()) / (_lib.SIDE * _lib.SIDE),
                "s1s2_iou": float((s1m_ & m160).sum() / max((s1m_ | m160).sum(), 1)),
                "bbox_area_ratio": (gw * gh) / float(_lib.TARGET * _lib.TARGET),
                "bbox_aspect": float(min(gw, gh) / max(gw, gh)),
                "verifier_prob": None})
            bins.append(cv2.resize(P.astype(np.uint8), (rectl["cw"], rectl["ch"]),
                                   interpolation=cv2.INTER_NEAREST))
            rects.append(rectl)
            crop_local.append({"crop_img": crop_img, "mask160": m160})
    if not feats:
        return []
    if m.vb is not None:
        vp = _lib.verifier_probs(m.vb, crop_local)
        for i in range(len(feats)):
            feats[i]["verifier_prob"] = float(vp[i])
    from sklearn.linear_model import LogisticRegression
    m3 = LogisticRegression(max_iter=1000, class_weight="balanced")
    m3.coef_, m3.intercept_, m3.classes_ = m.fz["coef"], m.fz["intercept"], np.array([0, 1])
    X = np.array([[f[c] for c in _lib.FCOLS] for f in feats], float)
    fused = m3.predict_proba((X - m.fz["mu"]) / m.fz["sg"])[:, 1]
    s1c = np.array([f["s1_conf"] for f in feats])
    ar = np.array([f["s2_area_ratio"] for f in feats])
    recs = []
    for i in range(len(feats)):
        r = rects[i]
        recs.append({"x0": r["x0"], "y0": r["y0"], "cw": r["cw"], "ch": r["ch"],
                     "bin": bins[i], "fused": float(fused[i]), "s1_conf": float(s1c[i]),
                     "area_ratio": float(ar[i])})
    return recs


def eval_thr(recs_per_img, gts_per_img, thr_large, thr_std=THR_STD):
    from src.inference.fusion import greedy_nms, rec_bin
    tp = fp = preds_tot = 0
    gtb = {b: 0 for b in _lib.BUCKETS}
    tpb = {b: 0 for b in _lib.BUCKETS}
    td = []
    for recs, gts in zip(recs_per_img, gts_per_img):
        H, W = gts["H"], gts["W"]
        ar = np.array([r["area_ratio"] for r in recs]) if recs else np.zeros(0)
        s1c = np.array([r["s1_conf"] for r in recs]) if recs else np.zeros(0)
        fused = np.array([r["fused"] for r in recs]) if recs else np.zeros(0)
        thr = np.where(ar >= LARGE_EDGE, thr_large, thr_std) if len(recs) else np.zeros(0)
        ok = (s1c >= S1_MIN) & (fused >= thr) if len(recs) else np.zeros(0, bool)
        sel = [recs[i] for i in np.argwhere(ok).ravel()]
        preds = [rec_bin(k, H, W) for k in greedy_nms(sel, "mask_nms", 0.5, H, W)] if sel else []
        preds_tot += len(preds)
        pairs = []
        for pi, P in enumerate(preds):
            for gi, G in enumerate(gts["gts"]):
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
            P, G = preds[pi], gts["gts"][gi]
            inter = int((P & G["mask"]).sum())
            td.append(2 * inter / max(int(P.sum()) + int(G["mask"].sum()), 1))
            tpb[G["bucket"]] += 1
        for G in gts["gts"]:
            gtb[G["bucket"]] += 1
        tp += len(used_g)
        fp += len(preds) - len(used_p)
    rec = tp / max(sum(gtb.values()), 1)
    prec = tp / max(preds_tot, 1)
    return {"thr_large": thr_large,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
            "r_large": round(tpb["large"] / max(gtb["large"], 1), 4),
            "r_medium": round(tpb["medium"] / max(gtb["medium"], 1), 4),
            "r_small": round(tpb["small"] / max(gtb["small"], 1), 4),
            "tp_dice": round(float(np.mean(td)), 4) if td else 0.0,
            "fp_img": round(fp / len(recs_per_img), 3)}


def run_split(m, vb, man, split):
    rows = man[man.split == split].reset_index(drop=True)
    recs_per_img, gts_per_img = [], []
    for ri, r in rows.iterrows():
        img = cv2.imread(r.image_path)
        H, W = img.shape[:2]
        gts, _ = _lib.image_gts(r.image_path, str(_lib.RAW / f"{r.image_id}.png"), H, W)
        recs = image_candidates(m, r.image_path)
        recs_per_img.append(recs)
        gts_per_img.append({"H": H, "W": W, "gts": gts})
        if (ri + 1) % 100 == 0:
            print(f"  {split} {ri+1}/{len(rows)}", flush=True)
    return [eval_thr(recs_per_img, gts_per_img, t) for t in THRS]


def main() -> None:
    from src.inference.models import CalculusCascadeModel
    cfg = load_yaml(ROOT / "configs/pipeline.yaml")
    m = CalculusCascadeModel(**cfg["inference"]["calculus"], **cfg["weights"]["calculus"])
    man = pd.read_csv(_lib.DATA / "split_manifest_v3.csv")
    out = {}
    for split in ("val", "test"):
        out[split] = run_split(m, m.vb, man, split)
        for r in out[split]:
            print(split, r, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "thr_large_sweep.json").write_text(json.dumps(out, indent=2))
    print("saved", OUT / "thr_large_sweep.json")


if __name__ == "__main__":
    main()
