# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
f"""Plaque single-TOOTH-level eval v2: tooth_id pairing + gingival-window sweep
+ tooth-level Platt recalibration.

Fixes the v1 protocol (outputs/plaque_test_toothlevel): v1 paired each GT *box*
to the max-IoU segmented tooth (match rate 16%, AUROC diluted to 0.799 by
band-vs-whole-tooth granularity mismatch). v2 groups GT boxes by the dataset's
own tooth_id into TOOTH-level units (label=1 if any band positive, screening
semantics), matches via union-box IoU / centroid containment, and evaluates:

  step1  tooth-level AUROC with whole-crop score (new baseline)
  step2  gingival-window score sweep on val (frozen geometry -> test)
  step3  Platt recalibration on val tooth pairs (+SANO-refit negatives),
         threshold re-frozen at val sens>=0.95 max-spec, applied to test+SANO

Outputs: outputs/plaque_toothlevel_v2/{tooth_records.csv,results.json,summary.md}
Run: PYTHONPATH={_ROOT} python3 scripts/eval_plaque_toothlevel_v2.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, f"{_ROOT}")
from src.inference.models import PlaqueModel, ToothModel  # noqa: E402

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

ROOT = Path(f"{_ROOT}")
DATA = ROOT / "datasets/processed/plaque_bin_v1"
RAW_DIRS = [ROOT / "datasets/raw/Plaque/mendeley-dataset-materials_Part_1/data",
            ROOT / "datasets/raw/Plaque/mendeley-dataset-materials_Part_2/data"]
SANO_CSV = ROOT / "datasets/processed/plaque_sano/sano_crops.csv"
SANO_DIR = ROOT / "datasets/processed/plaque_sano"
OUT = ROOT / "outputs/plaque_toothlevel_v2"

FHS = (0.25, 1 / 3, 0.5)                  # gingival-band height fractions
SANO_REFIT_N_IMG = 20                     # last-20-image slice == official 503-crop val negs


def find_image(image_id: str) -> Path | None:
    for d in RAW_DIRS:
        for img in (d / "images").glob(f"*/{image_id}.jpg"):
            return img
    return None


def find_label(image_id: str) -> Path | None:
    for d in RAW_DIRS:
        cands = list((d / "labels").glob(f"*/{image_id}.txt"))
        if cands:
            return cands[0]
    return None


def platt(raw: np.ndarray, ab) -> np.ndarray:
    a, b = ab
    p = np.clip(raw, 1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + np.exp(-(a * np.log(p / (1 - p)) + b)))


def base_images(split: str) -> list[str]:
    rows = pd.read_csv(DATA / "plaque_crops.csv")
    t = rows[(rows.split == split) & (rows.is_aug == 0)]
    t = t[~t.image_id.str.contains("dup")]
    return sorted(t.image_id.unique())


def gt_teeth(lbl: Path, W: int, H: int) -> dict[int, dict]:
    """Group raw GT lines by tooth_id -> tooth-level unit."""
    groups: dict[int, dict] = {}
    for line in lbl.read_text().splitlines():
        f = line.split()
        if len(f) != 6:
            continue
        y, cx, cy, w, h, tid = int(f[0]), *(float(v) for v in f[1:5]), int(f[5])
        bx = np.array([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
        g = groups.setdefault(tid, {"boxes": [], "labels": set(), "fdi": None})
        g["boxes"].append(bx)
        g["labels"].add(y)
    out = {}
    for tid, g in groups.items():
        B = np.stack(g["boxes"])
        union = np.array([B[:, 0].min(), B[:, 1].min(), B[:, 2].max(), B[:, 3].max()])
        out[tid] = {
            "union": union,
            "y_any": int(1 in g["labels"]),
            "mixed": int(len(g["labels"]) > 1),
            "n_boxes": len(B),
            "cx": float((union[0] + union[2]) / 2),
            "cy": float((union[1] + union[3]) / 2),
        }
    return out


def arch_of(tb: np.ndarray, H: int) -> str:
    return "upper" if (tb[1] + tb[3]) / 2 < H / 2 else "lower"


def band_crop(img: np.ndarray, tb, arch: str, fh: float) -> np.ndarray:
    x0, y0, x1, y1 = [int(round(v)) for v in tb]
    h = y1 - y0
    bh = max(8, int(round(fh * h)))
    if arch == "upper":
        return img[y0:y0 + bh, x0:x1]
    return img[y1 - bh:y1, x0:x1]


def match_gt_tooth(gt: dict, tb_all: np.ndarray) -> tuple[int, float, str]:
    """Return (seg_tooth_idx, iou, match_type) for one GT tooth unit."""
    u = gt["union"]
    ix0 = np.maximum(tb_all[:, 0], u[0]); iy0 = np.maximum(tb_all[:, 1], u[1])
    ix1 = np.minimum(tb_all[:, 2], u[2]); iy1 = np.minimum(tb_all[:, 3], u[3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    aG = (u[2] - u[0]) * (u[3] - u[1])
    aT = (tb_all[:, 2] - tb_all[:, 0]) * (tb_all[:, 3] - tb_all[:, 1])
    iou = inter / np.maximum(aG + aT - inter, 1e-9)
    j = int(np.argmax(iou))
    if iou[j] >= 0.3:
        return j, float(iou[j]), "iou"
    cx_in = (tb_all[:, 0] <= gt["cx"]) & (gt["cx"] <= tb_all[:, 2]) & \
            (tb_all[:, 1] <= gt["cy"]) & (gt["cy"] <= tb_all[:, 3])
    scx = (tb_all[:, 0] + tb_all[:, 2]) / 2
    scy = (tb_all[:, 1] + tb_all[:, 3]) / 2
    s_in = (u[0] <= scx) & (scx <= u[2]) & (u[1] <= scy) & (scy <= u[3])
    contain = cx_in | s_in
    if contain.any():
        cand = np.where(contain)[0]
        k = cand[np.argmax(iou[cand])]
        return int(k), float(iou[k]), "contain"
    return -1, float(iou[j]), "none"


def collect_split(tm, pm, split: str) -> list[dict]:
    recs: list[dict] = []
    for iid in base_images(split):
        img_path, lbl = find_image(iid), find_label(iid)
        if img_path is None or lbl is None:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        gt = gt_teeth(lbl, W, H)
        if not gt:
            continue
        teeth = tm.predict(img)
        if not teeth:
            for tid, g in gt.items():
                recs.append({"split": split, "image_id": iid, "tooth_id": tid,
                             "matched": 0, "match_iou": 0.0, "match_type": "no_seg",
                             "y_any": g["y_any"], "mixed": g["mixed"], "n_boxes": g["n_boxes"]})
            continue
        tb = np.array([t["bbox"] for t in teeth], float)
        crops, keep = [], []
        for ti in range(len(teeth)):
            x0, y0, x1, y1 = [int(max(0, v)) for v in tb[ti]]
            x1, y1 = min(W, x1), min(H, y1)
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            crops.append(img[y0:y1, x0:x1])
            keep.append(ti)
        if not crops:
            continue
        archs = {ki: arch_of(tb[ki], H) for ki in keep}
        feed, tags = [], []
        for ki, c in zip(keep, crops):
            feed.append(c); tags.append((ki, "full"))
            for fh in FHS:
                feed.append(band_crop(img, tb[ki], archs[ki], fh)); tags.append((ki, f"band{fh:.2f}"))
        raw = np.array(pm._proba_array_raw(feed))
        score = dict(zip(tags, raw))
        for tid, g in gt.items():
            j, miou, mtype = match_gt_tooth(g, tb)
            r = {"split": split, "image_id": iid, "tooth_id": tid,
                 "y_any": g["y_any"], "mixed": g["mixed"], "n_boxes": g["n_boxes"],
                 "gt_cx": g["cx"], "gt_cy": g["cy"]}
            if mtype == "none" or j not in keep:
                r.update({"matched": 0, "match_iou": round(miou, 4), "match_type": mtype})
            else:
                r.update({"matched": 1, "match_iou": round(miou, 4), "match_type": mtype,
                          "arch": archs[j], "seg_conf": round(teeth[j]["conf"], 4),
                          "p_full": round(float(score[(j, "full")]), 5)})
                for fh in FHS:
                    r[f"p_band{fh:.2f}"] = round(float(score[(j, f"band{fh:.2f}")]), 5)
            recs.append(r)
    return recs


def collect_sano(pm, subset: str, limit_imgs: int | None = None) -> list[dict]:
    """SANO crops have no arch info -> try top AND bottom gingival bands,
    keep per-fh max. Returns one record per crop (y=0, all healthy)."""
    df = pd.read_csv(SANO_CSV)
    df = df[df.subset == subset]
    imgs = sorted(df.image.unique())
    if limit_imgs:
        imgs = imgs[-limit_imgs:]
    keep_names = set(df[df.image.isin(imgs)].crop.str.split("/").str[-1])
    paths = []
    for sub in ("train", "eval"):
        for p in sorted((SANO_DIR / sub).glob("*.jpg")):
            if p.name in keep_names:
                paths.append(p)
    recs: list[dict] = []
    BATCH = 96
    for i in range(0, len(paths), BATCH):
        names = [p.name for p in paths[i:i + BATCH]]
        chunk = [cv2.imread(str(p)) for p in paths[i:i + BATCH]]
        idx_ok = [k for k, c in enumerate(chunk) if c is not None]
        if not idx_ok:
            continue
        feed, spans = [], []
        for k in idx_ok:
            c = chunk[k]
            feed.append(c); spans.append((k, "full", -1.0))
            h = c.shape[0]
            for fh in FHS:
                bh = max(8, int(round(fh * h)))
                feed.append(c[:bh]); spans.append((k, f"band{fh:.2f}", 0.0))
                feed.append(c[-bh:]); spans.append((k, f"band{fh:.2f}", 1.0))
        raw = np.array(pm._proba_array_raw(feed))
        per: dict[int, dict] = defaultdict(dict)
        for (k, tag, _side), v in zip(spans, raw):
            if tag == "full":
                per[k]["p_full"] = float(v)
            else:
                per[k][tag] = max(per[k].get(tag, 0.0), float(v))
        for k in idx_ok:
            r = {"subset": subset, "crop": names[k], "y": 0,
                 "p_full": round(per[k]["p_full"], 5)}
            for fh in FHS:
                r[f"p_band{fh:.2f}"] = round(per[k].get(f"band{fh:.2f}", np.nan), 5)
            recs.append(r)
        print(f"sano/{subset}: {i + len(idx_ok)}/{len(paths)}", flush=True)
    return recs


def pair_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Matched pairs with derived score variants (raw scale)."""
    p = df[df.matched == 1].copy()
    for fh in FHS:
        c = f"p_band{fh:.2f}"
        p[f"max_full_{c}"] = p[["p_full", c]].max(axis=1)
    return p


def auroc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    if len(set(y.tolist())) < 2 or len(y) == 0:
        return None
    return round(float(roc_auc_score(y, s)), 4)


def ap(y, s):
    y, s = np.asarray(y), np.asarray(s)
    if len(set(y.tolist())) < 2 or len(y) == 0:
        return None
    return round(float(average_precision_score(y, s)), 4)


def thr_metrics(y, p, thr):
    y, p = np.asarray(y), np.asarray(p)
    pred = p >= thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    sens = tp / max(1, tp + fn); spec = tn / max(1, tn + fp)
    prec = tp / max(1, tp + fp)
    return dict(sens=round(sens, 4), spec=round(spec, 4), prec=round(prec, 4),
                f1=round(2 * prec * sens / max(1e-9, prec + sens), 4),
                tp=tp, fp=fp, fn=fn, tn=tn)


def fit_platt(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression
    z = np.log(np.clip(s, 1e-6, 1 - 1e-6) / (1 - np.clip(s, 1e-6, 1 - 1e-6)))
    lr = LogisticRegression(max_iter=1000)
    lr.fit(z.reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "tooth_records.csv"
    if cache.exists() and "--recache" not in sys.argv:
        big = pd.read_csv(cache)
        df = big[big.subset.isna()].drop(columns=[c for c in ["subset"] if c in big])
        df["matched"] = df["matched"].astype(int)
        sano_ev_df = big[big.subset == "eval"]
        sano_rf_df = big[big.subset == "train"]
        print(f"loaded cache: mendeley={len(df)} sano_eval={len(sano_ev_df)} "
              f"sano_refit={len(sano_rf_df)}")
    else:
        pm = PlaqueModel(weights=str(ROOT / "outputs/plaque_bin_v2_sano/weights/best.pt"),
                         imgsz=224, windows=[[0, 0, 1, 1]])
        tm = ToothModel(weights=str(ROOT / "outputs/iop_toothseg/weights/best.pt"),
                        conf=0.30, iou=0.70, imgsz=896)
        all_recs = []
        for split in ("val", "test"):
            recs = collect_split(tm, pm, split)
            all_recs.extend(recs)
            m = sum(r["matched"] for r in recs)
            print(f"{split}: gt_teeth={len(recs)} matched={m} ({m / max(1, len(recs)):.1%})",
                  flush=True)
        df = pd.DataFrame(all_recs)
        sano_ev_df = pd.DataFrame(collect_sano(pm, "eval"))
        sano_rf_df = pd.DataFrame(collect_sano(pm, "train", limit_imgs=SANO_REFIT_N_IMG))
        pd.concat([df, sano_ev_df, sano_rf_df]).to_csv(cache, index=False)
        print("saved tooth_records.csv")

    for d in (sano_ev_df, sano_rf_df):  # derived max(full, band) variants
        for fh in FHS:
            c = f"p_band{fh:.2f}"
            if c in d:
                d[f"max_full_{c}"] = d[["p_full", c]].max(axis=1)

    # ---------- step1: tooth_id-paired baseline ----------
    # KEY FACT: Mendeley val/test have ZERO all-negative teeth at tooth level
    # (every tooth with a negative band also has >=1 positive band), so a
    # within-Mendeley tooth-level AUROC is UNDEFINED. Report match/coverage +
    # mixed-teeth stats; discrimination can only be measured vs SANO healthy.
    res: dict = {"protocol": "tooth_id grouping (any-band-positive = tooth positive); "
                            "match = union-box IoU>=0.3 or centroid containment",
                 "key_fact": "mendeley tooth-level has no all-negative teeth; "
                             "negatives come from SANO healthy crops only"}
    step1 = {}
    for split in ("val", "test"):
        p = pair_frame(df[df.split == split])
        n_gt = int((df.split == split).sum()); n_m = len(p)
        un = df[(df.split == split) & (df.matched == 0)]
        step1[split] = {
            "n_gt_teeth": n_gt, "n_matched": n_m,
            "match_rate": round(n_m / max(1, n_gt), 4),
            "match_type_iou": int((p.match_type == "iou").sum()),
            "match_type_contain": int((p.match_type == "contain").sum()),
            "unmatched_reason": {k: int(v) for k, v in un.match_type.value_counts().items()},
            "mixed_teeth": int(p.mixed.sum()),
            "label_dist": {"pos": int(p.y_any.sum()), "neg": int((1 - p.y_any).sum())},
            "p_full_median_pos": round(float(p.p_full.median()), 4),
        }
    res["step1_tooth_id_pairing"] = step1

    # ---------- step2: gingival-window sweep ----------
    # Selection set = val matched positive teeth + SANO-refit healthy teeth
    # (raw scale). This mirrors deployment: diseased teeth must score high,
    # healthy teeth must score low.
    pv = pair_frame(df[df.split == "val"])
    pt = pair_frame(df[df.split == "test"])
    y_sel = np.concatenate([pv.y_any.values, np.zeros(len(sano_rf_df))])
    variants = ["p_full"] + [f"p_band{fh:.2f}" for fh in FHS] + \
               [f"max_full_p_band{fh:.2f}" for fh in FHS]
    sweep = {}
    for v in variants:
        s_sel = np.concatenate([pv[v].values, sano_rf_df[v].values])
        okm = ~np.isnan(s_sel)
        sweep[v] = {"sel_auroc": auroc(y_sel[okm], s_sel[okm]),
                    "val_pos_median": round(float(pv[v].median()), 4),
                    "sano_refit_fp_p95": round(float(np.nanpercentile(sano_rf_df[v], 95)), 4)}
    best = max(sweep, key=lambda k: sweep[k]["sel_auroc"] or -1)
    s_test = np.concatenate([pt[best].values, sano_ev_df[best].values])
    y_test2 = np.concatenate([pt.y_any.values, np.zeros(len(sano_ev_df))])
    sweep[best].update({"test_sens_median": round(float(pt[best].median()), 4),
                        "combined_test_sano_auroc": auroc(y_test2, s_test)})
    res["step2_window_sweep"] = {"selected_variant": best, "sweep": sweep}

    # ---------- step3: tooth-level Platt recalibration + threshold re-freeze ----------
    y_rf = np.zeros(len(sano_rf_df))
    s_rf = sano_rf_df[best].values
    okm = ~np.isnan(s_rf)
    y_rf, s_rf = y_rf[okm], s_rf[okm]
    ab = fit_platt(np.concatenate([yv := pv.y_any.values, y_rf]),
                   np.concatenate([sv := pv[best].values, s_rf]))
    cv_sel = platt(sv, ab)                       # val positives (calibrated)
    cs_rf = platt(s_rf, ab)                      # sano-refit negatives
    ct_pos = platt(pt[best].values, ab)          # test positives
    se = sano_ev_df[best].values
    se = se[~np.isnan(se)]
    cs_ev = platt(se, ab)                        # sano-eval negatives
    # freeze: sens>=0.95 on val positives, max spec on sano-refit
    grid = np.arange(0.01, 1.0, 0.01)
    sens_v = [(cv_sel >= t).mean() for t in grid]
    ok = np.array(sens_v) >= 0.95
    thr = float(grid[ok][np.argmax([(cs_rf < t).mean() for t in grid[ok]])]) if ok.any() else 0.5
    y_all_test = np.concatenate([pt.y_any.values, np.zeros(len(cs_ev))])
    s_all_test = np.concatenate([ct_pos, cs_ev])
    res["step3_tooth_platt"] = {
        "variant": best, "platt_a_b": [round(v, 4) for v in ab],
        "n_refit_pos": int(yv.sum()), "n_refit_neg": int(len(y_rf)),
        "frozen_thr_calib": round(thr, 3),
        "val_pos_sens_at_thr": round(float((cv_sel >= thr).mean()), 4),
        "sano_refit_spec_at_thr": round(float((cs_rf < thr).mean()), 4),
        "test_pos_sens_at_thr": round(float((ct_pos >= thr).mean()), 4),
        "test_pos_median": round(float(np.median(ct_pos)), 4),
        "sano_eval_spec_at_thr": round(float((cs_ev < thr).mean()), 4),
        "sano_eval_fp": int((cs_ev >= thr).sum()), "sano_eval_n": int(len(cs_ev)),
        "combined_test_sano": {"auroc": auroc(y_all_test, s_all_test),
                               "ap": ap(y_all_test, s_all_test),
                               **thr_metrics(y_all_test, s_all_test, thr)},
    }
    res["reference_v1_band_paired"] = {"auroc_all_0.3": 0.799, "auroc_iou_ge0.5": 0.924,
                                       "match_rate": 0.161,
                                       "note": "v1 paired BAND labels to whole crops "
                                               "(granularity mismatch); not comparable"}

    (OUT / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
