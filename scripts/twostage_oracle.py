"""Oracle diagnostics for the calculus two-stage pipeline (VALIDATION ONLY).

Collects, for every val image (pos+neg):
  - stage-1 candidate boxes at c1 in C1_GRID (boxes deduped across c1)
  - stage-2 soft mask + binary mask for each unique candidate box,
    restored to full-image coordinates as ROI (bin uint8, soft uint8=prob*255)
  - per candidate: s1_conf, s2_conf, fg_prob (mean soft inside bin mask),
    composite score = s1_conf * fg_prob

Then reports:
  Oracle A  stage-1 candidate coverage per size bucket vs c1 (overlap>0 rule)
  Oracle B  stage-2 mask upper bound per covered GT (best per-candidate
            IoU/Dice, ignoring fusion) at the current c1=0.2
  Oracle C  fusion ablation per size bucket at c1=0.2:
            union / best-score-single / box-NMS / mask-NMS /
            prob-max / prob-weighted
  Scan     per-c1 table: bucket recalls, avg proposals/img, FP/img,
            final Recall@IoU0.5 & Dice of best fusion, est. inference time

GT is used ONLY for oracle evaluation, never for candidate selection.
Outputs into outputs/calculus_yolo11m_twostage/oracle_val/
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

import argparse
import json
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from src.inference.twostage import MARGIN, MIN_WINDOW, TARGET
from src.utils.paths import PROCESSED_DIR

DATA = PROCESSED_DIR / "calculus_seg"
RUN = Path("outputs/calculus_yolo11m_twostage")
OUT = RUN / "oracle_val"
CACHE = OUT / "oracle_val_cache.pkl"
C1_GRID = [0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30]
CUR_C1 = 0.2
C2 = 0.05
BUCKET_EDGES = (0.00552, 0.01565)  # same as test lesion_size_analysis
BUCKETS = ("small", "medium", "large")
NMS_IOU = 0.5
SCORE_THR_IMG = 0.05  # image-score threshold fixed for FP rate in scans


def bucket_of(ratio: float) -> str:
    if ratio < BUCKET_EDGES[0]:
        return "small"
    if ratio < BUCKET_EDGES[1]:
        return "medium"
    return "large"


def box_overlap_gt(gbox, cand) -> bool:
    gx, gy, gx2, gy2 = gbox
    bx0, by0, bx1, by1 = cand
    return max(0, min(gx2, bx1) - max(gx, bx0)) * max(0, min(gy2, by1) - max(gy, by0)) > 0


def box_iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua


from src.inference.fusion import (  # noqa: E402  shared implementations
    MODES, fuse_mask, greedy_nms, rec_bin, rec_soft)


def collect() -> dict:
    s1 = YOLO("outputs/calculus_yolo11m_full/weights/best.pt")
    s2 = YOLO("outputs/calculus_yolo11m_crop/weights/best.pt")
    man = pd.read_csv(DATA / "split_manifest.csv")
    rows = man[man.split == "val"]
    cache: dict = {}
    for i, (_, r) in enumerate(rows.iterrows()):
        img = cv2.imread(r["image_path"])
        H, W = img.shape[:2]
        per_c1: dict[float, list] = {}
        unique: dict[tuple, dict] = {}
        t_s1 = 0.0
        for c in C1_GRID:
            t0 = time.time()
            res = s1.predict(r["image_path"], conf=c, iou=0.6, imgsz=TARGET,
                             device=0, retina_masks=True, max_det=100,
                             verbose=False)[0]
            t_s1 += time.time() - t0
            if res.boxes is None or len(res.boxes) == 0:
                per_c1[c] = []
                continue
            bb = res.boxes.xyxy.cpu().numpy()
            cc = res.boxes.conf.cpu().numpy()
            lst = []
            for k in range(len(bb)):
                bx = tuple(int(round(v)) for v in bb[k])
                lst.append(bx)
                if bx not in unique:
                    unique[bx] = {"s1_conf": float(cc[k])}
            per_c1[c] = lst
        # stage-2 per unique box
        for bx in list(unique):
            x0, y0, x1, y1 = bx
            side = float(max(x1 - x0, y1 - y0) * MARGIN)
            side = max(side, MIN_WINDOW)
            side = float(min(side, max(H, W)))
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            ax0 = int(round(cx - side / 2)); ay0 = int(round(cy - side / 2))
            ax0 = max(0, min(ax0, W - int(side))); ay0 = max(0, min(ay0, H - int(side)))
            wx1 = min(ax0 + int(side), W); wy1 = min(ay0 + int(side), H)
            ch, cw = wy1 - ay0, wx1 - ax0
            crop = cv2.resize(img[ay0:wy1, ax0:wx1], (TARGET, TARGET),
                              interpolation=cv2.INTER_AREA)
            res = s2.predict(crop, conf=C2, iou=0.6, imgsz=TARGET, device=0,
                             retina_masks=True, max_det=100, verbose=False)[0]
            rec = {"x0": ax0, "y0": ay0, "cw": cw, "ch": ch,
                   "s1_conf": unique[bx]["s1_conf"]}
            if res.boxes is not None and len(res.boxes) and res.masks is not None:
                mk = res.masks.data.cpu().numpy()
                binc = np.zeros((TARGET, TARGET), np.uint8)
                softc = np.zeros((TARGET, TARGET), np.float32)
                s2c = 0.0
                for k in range(mk.shape[0]):
                    s2c = max(s2c, float(res.boxes.conf[k]))
                    soft = mk[k].astype(np.float32)
                    softc = np.maximum(softc, soft)
                    binc = np.maximum(binc, (soft > 0.5).astype(np.uint8))
                binroi = cv2.resize(binc, (cw, ch), interpolation=cv2.INTER_NEAREST)
                softroi = cv2.resize(softc, (cw, ch), interpolation=cv2.INTER_LINEAR)
                fgp = float(softroi[binroi > 0].mean()) if binroi.any() else 0.0
                rec.update({"bin": binroi, "soft": (softroi * 255).astype(np.uint8),
                            "s2_conf": s2c, "fg_prob": fgp})
            else:
                rec.update({"bin": None, "soft": None, "s2_conf": 0.0, "fg_prob": 0.0})
            unique[bx] = rec
        cache[r["image_id"]] = {
            "H": H, "W": W, "per_c1": per_c1, "unique": unique,
            "stage1_ms": 1000.0 * t_s1 / len(C1_GRID),
        }
        if (i + 1) % 50 == 0:
            print(f"  collected {i + 1}/{len(rows)}", flush=True)
    return cache


def load_gt_inst(img_p: str, mask_p: str) -> list[dict]:
    img = cv2.imread(img_p)
    H, W = img.shape[:2]
    gt = cv2.imread(mask_p, cv2.IMREAD_UNCHANGED) > 0
    if not gt.any():
        return []
    n, comp, _, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        if (comp == i).sum() < 5:
            continue
        m = (comp == i).astype(np.uint8)
        gx, gy, gw, gh = cv2.boundingRect(m)
        out.append({"mask": m, "box": (gx, gy, gx + gw, gy + gh),
                    "px": int(m.sum()), "ratio": int(m.sum()) / float(H * W),
                    "bucket": bucket_of(int(m.sum()) / float(H * W))})
    return out


def analyze(cache: dict) -> None:
    man = pd.read_csv(DATA / "split_manifest.csv")
    rows = man[man.split == "val"]

    # ---------- per-image candidate lists ----------
    per_img = {}
    for _, r in rows.iterrows():
        im = cache[r["image_id"]]
        cands_c1 = {}
        for c, keys in im["per_c1"].items():
            cs = []
            for bx in keys:
                rec = im["unique"][bx]
                if rec["bin"] is None:
                    continue
                rec["score"] = rec["s1_conf"] * rec["fg_prob"]
                cs.append(rec)
            cands_c1[c] = cs
        per_img[(r["image_id"])] = {
            "has": bool(r["has_calculus"]),
            "img_p": r["image_path"], "mask_p": r["mask_original_path"],
            "cands": cands_c1, "H": im["H"], "W": im["W"],
            "stage1_ms": im["stage1_ms"],
        }

    rows_pos = [r for _, r in rows.iterrows() if r["has_calculus"]]

    # ---------- Oracle A ----------
    a_rows = []
    for c in C1_GRID:
        cnt = {b: [0, 0] for b in BUCKETS}
        for r in rows_pos:
            gts = load_gt_inst(r["image_path"], r["mask_original_path"])
            cands = per_img[r["image_id"]]["cands"][c]
            cboxes = [(cc["x0"], cc["y0"], cc["x0"] + cc["cw"], cc["y0"] + cc["ch"])
                      for cc in cands]
            for g in gts:
                cnt[g["bucket"]][1] += 1
                if any(box_overlap_gt(g["box"], b) for b in cboxes):
                    cnt[g["bucket"]][0] += 1
        for b in BUCKETS:
            n_cov, n_tot = cnt[b]
            a_rows.append({"c1": c, "bucket": b, "n_gt": n_tot,
                           "covered": n_cov,
                           "recall": (n_cov / n_tot) if n_tot else 0.0})
    A = pd.DataFrame(a_rows)

    # ---------- Oracle B (c1 = CUR_C1) ----------
    b_stats = []
    for r in rows_pos:
        gts = load_gt_inst(r["image_path"], r["mask_original_path"])
        cands = per_img[r["image_id"]]["cands"][CUR_C1]
        cboxes = [(cc["x0"], cc["y0"], cc["x0"] + cc["cw"], cc["y0"] + cc["ch"])
                  for cc in cands]
        for g in gts:
            if not any(box_overlap_gt(g["box"], b) for b in cboxes):
                continue
            mask = g["mask"]
            best_iou, best_dice = 0.0, 0.0
            for cc in cands:
                cb = rec_bin(cc, per_img[r["image_id"]]["H"], per_img[r["image_id"]]["W"])
                inter = int((mask & cb).sum())
                uni = int((mask | cb).sum())
                iou = inter / uni if uni else 0.0
                dice = 2 * inter / (int(mask.sum()) + int(cb.sum())) if (int(cb.sum()) or int(mask.sum())) else 1.0
                best_iou = max(best_iou, iou)
                best_dice = max(best_dice, dice)
            b_stats.append({"bucket": g["bucket"], "best_iou": best_iou,
                            "best_dice": best_dice,
                            "hit@0.3": int(best_iou >= 0.3),
                            "hit@0.5": int(best_iou >= 0.5),
                            "hit@0.7": int(best_iou >= 0.7)})
    Btab = pd.DataFrame(b_stats)

    # ---------- Oracle C fusion ablation ----------
    c_rows = []
    for r in rows_pos:
        gts = load_gt_inst(r["image_path"], r["mask_original_path"])
        if not gts:
            continue
        cands = per_img[r["image_id"]]["cands"][CUR_C1]
        masks = {m: fuse_mask(cands, per_img[r["image_id"]]["H"],
                              per_img[r["image_id"]]["W"], m) for m in MODES}
        for m in MODES:
            P = masks[m] > 0
            for g in gts:
                mask = g["mask"]
                inter = int((mask & P).sum())
                uni = int((mask | P).sum())
                iou = inter / uni if uni else 0.0
                dice = 2 * inter / (int(mask.sum()) + int(P.sum())) if (int(mask.sum()) + int(P.sum())) else 1.0
                c_rows.append({"mode": m, "bucket": g["bucket"],
                               "dice": dice, "iou": iou, "hit05": int(iou >= 0.5)})
    C = pd.DataFrame(c_rows)
    c_stats = C.groupby(["mode", "bucket"]).agg(
        dice_mean=("dice", "mean"), iou_mean=("iou", "mean"),
        rec05=("hit05", "mean"), n=("dice", "count")).reset_index()

    # ---------- c1 scan: FP/image, avg proposals, final metrics, time ----------
    scan_rows = []
    for c in C1_GRID:
        n_cand = 0.0
        n_img = 0.0
        n_fp_img = 0
        n_neg = 0
        for _, r in rows.iterrows():
            im = per_img[r["image_id"]]
            cs = im["cands"][c]
            n_cand += len(cs)
            n_img += 1
            score = max([cc.get("score", 0.0) for cc in cs]) if cs else 0.0
            if not im["has"]:
                n_neg += 1
                if score >= SCORE_THR_IMG:
                    n_fp_img += 1
        inst_dices, inst_ious, inst_hits = [], [], []
        for r in rows_pos:
            gts = load_gt_inst(r["image_path"], r["mask_original_path"])
            if not gts:
                continue
            im = per_img[r["image_id"]]
            P = fuse_mask(im["cands"][c], im["H"], im["W"], "mask_nms") > 0
            for g in gts:
                m = g["mask"]
                inter = int((m & P).sum())
                uni = int((m | P).sum())
                iou = inter / uni if uni else 0.0
                dice = 2 * inter / (int(m.sum()) + int(P.sum())) if (int(m.sum()) + int(P.sum())) else 1.0
                inst_ious.append(iou)
                inst_dices.append(dice)
                inst_hits.append(int(iou >= 0.5))
        scan_rows.append({
            "c1": c,
            "small_rec": float(A[(A.c1 == c) & (A.bucket == "small")]["recall"].iloc[0]),
            "medium_rec": float(A[(A.c1 == c) & (A.bucket == "medium")]["recall"].iloc[0]),
            "large_rec": float(A[(A.c1 == c) & (A.bucket == "large")]["recall"].iloc[0]),
            "avg_proposals_img": float(n_cand / max(n_img, 1e-6)),
            "fp_img_rate": float(n_fp_img / max(n_neg, 1e-6)),
            "recall_iou05": float(np.mean(inst_hits)) if inst_hits else 0.0,
            "dice": float(np.mean(inst_dices)) if inst_dices else 0.0,
            "mean_iou": float(np.mean(inst_ious)) if inst_ious else 0.0,
        })
    for s in scan_rows:
        s["est_ms_img"] = (float(np.mean([per_img[r["image_id"]]["stage1_ms"]
                                          for _, r in rows.iterrows()]))
                           + s["avg_proposals_img"] * STAGE2_MS_EST)
    scan = pd.DataFrame(scan_rows)

    OUT.mkdir(parents=True, exist_ok=True)
    A.to_csv(OUT / "oracle_a_stage1_coverage.csv", index=False)
    Btab.to_csv(OUT / "oracle_b_stage2_upper.csv", index=False)
    c_stats.to_csv(OUT / "oracle_c_fusion_ablation.csv", index=False)
    scan.to_csv(OUT / "c1_scan.csv", index=False)

    lines = [
        "## Oracle A: Stage1 proposal upper bound (VAL, bbox overlap>0)",
        A.pivot_table(index="c1", columns="bucket", values="recall").round(4).to_string(),
        "",
        "## Oracle B: Stage2 mask upper bound (c1=0.2, covered GT only)",
        Btab.groupby("bucket").agg(
            n=("best_iou", "count"), best_iou=("best_iou", "mean"),
            best_dice=("best_dice", "mean"), r03=("hit@0.3", "mean"),
            r05=("hit@0.5", "mean"), r07=("hit@0.7", "mean")).round(4).to_string(),
        "",
        "## Oracle C: fusion ablation per bucket (c1=0.2)   [dice / rec@0.5 / meanIoU]",
        c_stats.pivot_table(index="mode", columns="bucket",
                            values=["dice_mean", "rec05", "iou_mean"]).round(4).to_string(),
        "",
        "## c1 scan (VAL; final = mask-nms fusion, image score=composite, thr=0.05)",
        scan.round(4).to_string(),
    ]
    report = "\n".join(lines)
    print(report, flush=True)
    (OUT / "oracle_diagnostics.md").write_text(report + "\n")


STAGE2_MS_EST = 90.0  # measured roughly on V100 (est. only)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-collect", action="store_true")
    ap.add_argument("--scan-stage2-ms", default=None)
    args = ap.parse_args()
    if args.scan_stage2_ms:
        global STAGE2_MS_EST
        STAGE2_MS_EST = float(args.scan_stage2_ms)
    if not args.skip_collect and not CACHE.exists():
        cache = collect()
        OUT.mkdir(parents=True, exist_ok=True)
        with open(CACHE, "wb") as f:
            pickle.dump(cache, f)
        print("collection done, cache saved", flush=True)
    elif CACHE.exists():
        print("cache loading", flush=True)
        cache = pickle.loads(CACHE.read_bytes())
    else:
        cache = None
    if cache is not None:
        analyze(cache)


if __name__ == "__main__":
    main()
