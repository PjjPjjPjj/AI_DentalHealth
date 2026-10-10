"""Pipeline audit experiments (P0/P1, validation-only, no training).

EXP1  coordinate-pipeline identity check (no model):
      GT crop -> resize 640 -> current restore -> paste -> Dice vs GT.
      Should be ~1.0; <0.95 means the geom transform chain is buggy.

EXP2  crop-space vs restore-space Stage2 Dice:
      per candidate: Dice_crop  = Dice(stage2 bin 640 vs GT crop->640)
                    Dice_restore= Dice(stage2 restored ROI vs GT full)
      per GT instance (covered), pick best candidate.

EXP3  candidate quality decoupling:
      GT-bbox crop (1.8 margin) -> stage2   vs   predicted-box crop -> stage2
      per bucket Dice (crop space). Big gap => Stage1 box/crop quality problem.

EXP4  Oracle B2: original per-instance stage2 masks (no np.maximum union!)
      vs merged candidate masks (Oracle B path). Best instance mask IoU
      per GT, per bucket: if B2 >> B, crop-internal multi-mask union hurts.

Also reports per-image vs per-instance GT counts (val/test) to explain
the instance-count discrepancy vs lesion_size_analysis.csv.

Outputs into outputs/calculus_yolo11m_twostage/pipeline_audit/
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
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from src.inference.twostage import MARGIN, MIN_WINDOW, TARGET, TwoStageSegmenter
from src.utils.paths import PROCESSED_DIR

DATA = PROCESSED_DIR / "calculus_seg"
RUN = Path("outputs/calculus_yolo11m_twostage")
OUT = RUN / "pipeline_audit"
CACHE = OUT / "audit_cache2.pkl"
C1_GRID = [0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30]
C1 = 0.05
C2 = 0.05
BUCKET_EDGES = (0.00552, 0.01565)
BUCKETS = ("small", "medium", "large")


def bucket_of(ratio: float) -> str:
    if ratio < BUCKET_EDGES[0]:
        return "small"
    if ratio < BUCKET_EDGES[1]:
        return "medium"
    return "large"


def dice(a: np.ndarray, b: np.ndarray) -> float:
    pa, pb = int((a > 0).sum()), int((b > 0).sum())
    inter = int(((a > 0) & (b > 0)).sum())
    return 2 * inter / (pa + pb) if (pa + pb) else 1.0


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(((a > 0) & (b > 0)).sum())
    uni = int(((a > 0) | (b > 0)).sum())
    return inter / uni if uni else 0.0


# ----------------------------------------------------------------- EXP1
def exp1_identity_check() -> pd.DataFrame:
    man = pd.read_csv(DATA / "split_manifest.csv")
    rows = man[(man.split == "val") & man.has_calculus]
    rows_l = []
    for _, r in rows.iterrows():
        img = cv2.imread(r["image_path"])
        H, W = img.shape[:2]
        gt = cv2.imread(r["mask_original_path"], cv2.IMREAD_UNCHANGED) > 0
        if not gt.any():
            continue
        n, comp, stats, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8), 8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 5:
                continue
            m = (comp == i).astype(np.uint8)
            gx, gy, gw, gh = cv2.boundingRect(m)
            box = (gx, gy, gx + gw, gy + gh)
            seg = TwoStageSegmenter  # only for _window logic, static
            x0, y0, x1, y1 = _static_window(box, H, W)
            ch, cw = y1 - y0, x1 - x0
            crop = m[y0:y1, x0:x1]
            s = cv2.resize(crop, (TARGET, TARGET), interpolation=cv2.INTER_AREA)
            s = (s > 0.5).astype(np.uint8)
            back = cv2.resize(s, (cw, ch), interpolation=cv2.INTER_NEAREST)
            buf = np.zeros((H, W), np.uint8)
            buf[y0:y1, x0:x1] = back
            d = dice(m, buf)
            i = iou(m, buf)
            rows_l.append({"bucket": bucket_of(int(m.sum()) / float(H * W)),
                           "dice": d, "iou": i, "gt_px": int(m.sum()),
                           "cw": cw, "ch": ch})
    df = pd.DataFrame(rows_l)
    tab = df.groupby("bucket").agg(n=("dice", "count"), dice=("dice", "mean"),
                                   iou=("iou", "mean"),
                                   dice_min=("dice", "min")).round(4)
    return tab


def _static_window(b, H: int, W: int) -> tuple[int, int, int, int]:
    x, y, x2, y2 = b
    cx, cy = (x + x2) / 2.0, (y + y2) / 2.0
    side = max(x2 - x, y2 - y) * MARGIN
    side = max(side, MIN_WINDOW)
    side = float(min(side, max(H, W)))
    x0 = int(round(cx - side / 2)); y0 = int(round(cy - side / 2))
    x0 = max(0, min(x0, W - int(side))); y0 = max(0, min(y0, H - int(side)))
    x1 = min(x0 + int(side), W); y1 = min(y0 + int(side), H)
    return x0, y0, x1, y1


# ----------------------------------------------------------------- collect for exp2/3/4
def collect2() -> dict:
    s1 = YOLO("outputs/calculus_yolo11m_full/weights/best.pt")
    s2 = YOLO("outputs/calculus_yolo11m_crop/weights/best.pt")
    man = pd.read_csv(DATA / "split_manifest.csv")
    rows = man[(man.split == "val") & man.has_calculus]
    cache: dict = {}
    for i, (_, r) in enumerate(rows.iterrows()):
        img = cv2.imread(r["image_path"])
        H, W = img.shape[:2]
        gt = cv2.imread(r["mask_original_path"], cv2.IMREAD_UNCHANGED) > 0
        per_c1: dict[float, list] = {}
        unique: dict[tuple, dict] = {}
        for c in C1_GRID:
            res = s1.predict(r["image_path"], conf=c, iou=0.6, imgsz=TARGET,
                             device=0, retina_masks=True, max_det=100,
                             verbose=False)[0]
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
                   "s1_conf": unique[bx]["s1_conf"], "insts": [], "bin640": None}
            if res.boxes is not None and len(res.boxes) and res.masks is not None:
                mk = res.masks.data.cpu().numpy()
                binc = np.zeros((TARGET, TARGET), np.uint8)
                s2c = 0.0
                for k in range(mk.shape[0]):
                    s2c = max(s2c, float(res.boxes.conf[k]))
                    mi = (mk[k] > 0.5).astype(np.uint8)
                    binc = np.maximum(binc, mi)
                    rec["insts"].append(mi)
                rec["bin640"] = binc
                rec["s2_conf"] = s2c
            else:
                rec["s2_conf"] = 0.0
            unique[bx] = dict(unique[bx], **rec)
        cache[r["image_id"]] = {
            "H": H, "W": W, "gt": gt, "per_c1": per_c1, "unique": unique,
            "img_path": r["image_path"],
        }
        if (i + 1) % 30 == 0:
            print(f"  collected {i + 1}/{len(rows)}", flush=True)
    return cache


def gt_instances(cache_r: dict) -> list[dict]:
    gt = cache_r["gt"]
    H, W = cache_r["H"], cache_r["W"]
    n, comp, _, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        if (comp == i).sum() < 5:
            continue
        m = (comp == i).astype(np.uint8)
        gx, gy, gw, gh = cv2.boundingRect(m)
        px = int(m.sum())
        out.append({"mask": m, "box": (gx, gy, gx + gw, gy + gh), "px": px,
                    "ratio": px / float(H * W),
                    "bucket": bucket_of(px / float(H * W))})
    return out


def rois_from_rec(rec: dict, H: int, W: int) -> tuple[np.ndarray, list]:
    buf = np.zeros((H, W), np.uint8)
    if rec["bin640"] is not None:
        m = cv2.resize(rec["bin640"], (rec["cw"], rec["ch"]),
                       interpolation=cv2.INTER_NEAREST)
        buf[rec["y0"]:rec["y0"] + rec["ch"], rec["x0"]:rec["x0"] + rec["cw"]] = m
    inst_rois = []
    for mi in rec.get("insts", []):
        b = np.zeros((H, W), np.uint8)
        m = cv2.resize(mi, (rec["cw"], rec["ch"]), interpolation=cv2.INTER_NEAREST)
        b[rec["y0"]:rec["y0"] + rec["ch"], rec["x0"]:rec["x0"] + rec["cw"]] = m
        inst_rois.append(b)
    return buf, inst_rois


# ----------------------------------------------------------------- EXP4+EXP2
def exp2_and_exp4(cache: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_l, b2_rows = [], []
    for image_id, c in cache.items():
        H, W = c["H"], c["W"]
        gts = gt_instances(c)
        for g in gts:
            best_merged_iou, best_merged_dc = 0.0, 0.0
            best_inst_iou = 0.0
            best_crop_dc, best_rest_dc = 0.0, 0.0
            for rec in c["unique"].values():
                if rec["bin640"] is None:
                    continue
                x0, y0, x1, y1 = rec["x0"], rec["y0"], rec["x0"] + rec["cw"], rec["y0"] + rec["ch"]
                gm = g["mask"]
                roi, inst_rois = rois_from_rec(rec, H, W)
                mi = int((gm & roi).sum())
                uni = int((gm | roi).sum())
                imiou = mi / uni if uni else 0.0
                imdice = 2 * mi / (int(gm.sum()) + int(roi.sum())) if (int(gm.sum()) + int(roi.sum())) else 1.0
                if imiou > best_merged_iou:
                    best_merged_iou, best_merged_dc = imiou, imdice
                for ir in inst_rois:
                    ii = int((gm & ir).sum())
                    iu = int((gm | ir).sum())
                    i2 = ii / iu if iu else 0.0
                    if i2 > best_inst_iou:
                        best_inst_iou = i2
                # crop-space dice: GT crop -> 640 vs stage2 bin640
                if rec["bin640"] is not None and gm[y0:y1, x0:x1].any():
                    tc = cv2.resize(gm[y0:y1, x0:x1].astype(np.uint8), (TARGET, TARGET),
                                    interpolation=cv2.INTER_AREA)
                    tc = (tc > 0.5).astype(np.uint8)
                    dp = dice(tc, rec["bin640"])
                    if dp > best_crop_dc:
                        best_crop_dc = dp
            rows_l.append({
                "image_id": image_id, "bucket": g["bucket"],
                "merged_best_iou": best_merged_iou,
                "merged_best_dice": best_merged_dc,
                "crop_space_best_dice": best_crop_dc,
                "gt_px": g["px"]})
            b2_rows.append({
                "image_id": image_id, "bucket": g["bucket"],
                "inst_best_iou": best_inst_iou,
                "gt_px": g["px"]})
    E = pd.DataFrame(rows_l)
    B2 = pd.DataFrame(b2_rows)
    return E, B2


# ----------------------------------------------------------------- EXP3 GT-bbox crop
def exp3_gtbox_crop(cache: dict) -> pd.DataFrame:
    s2 = YOLO("outputs/calculus_yolo11m_crop/weights/best.pt")
    rows_l = []
    for image_id, c in cache.items():
        H, W = c["H"], c["W"]
        img = cv2.imread(c["img_path"])
        gt = c["gt"]
        for g in gt_instances(c):
            box = g["box"]
            x0, y0, x1, y1 = _static_window(box, H, W)
            ch, cw = y1 - y0, x1 - x0
            crop = cv2.resize(img[y0:y1, x0:x1], (TARGET, TARGET),
                              interpolation=cv2.INTER_AREA)
            res = s2.predict(crop, conf=C2, iou=0.6, imgsz=TARGET, device=0,
                             retina_masks=True, max_det=100, verbose=False)[0]
            if res.boxes is None or len(res.boxes) == 0:
                rows_l.append({"bucket": g["bucket"], "dice_crop": 0.0,
                               "gt_px": g["px"], "conf": 0.0})
                continue
            binc = np.zeros((TARGET, TARGET), np.uint8)
            mk = res.masks.data.cpu().numpy()
            for k in range(mk.shape[0]):
                binc = np.maximum(binc, (mk[k] > 0.5).astype(np.uint8))
            tc = cv2.resize(gt[y0:y1, x0:x1].astype(np.uint8), (TARGET, TARGET),
                            interpolation=cv2.INTER_AREA)
            tc = (tc > 0.5).astype(np.uint8)
            rows_l.append({"bucket": g["bucket"],
                           "dice_crop": dice(tc, binc),
                           "gt_px": g["px"],
                           "conf": float(res.boxes.conf.max().item())})
    return pd.DataFrame(rows_l)


# ----------------------------------------------------------------- counts
def explain_counts() -> None:
    man = pd.read_csv(DATA / "split_manifest.csv")
    for split in ("val", "test"):
        pos = man[(man.split == split) & man.has_calculus]
        n_inst = {"small": 0, "medium": 0, "large": 0}
        for _, r in pos.iterrows():
            img = cv2.imread(r["image_path"])
            H, W = img.shape[:2]
            gt = cv2.imread(r["mask_original_path"], cv2.IMREAD_UNCHANGED) > 0
            n, comp, _, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8), 8)
            for i in range(1, n):
                if (comp == i).sum() < 5:
                    continue
                px = int((comp == i).sum())
                n_inst[bucket_of(px / float(H * W))] += 1
        print(f"{split}: images={len(pos)} instances={sum(n_inst.values())} "
              f"({n_inst})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-collect", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.skip_collect or not CACHE.exists():
        cache = collect2()
        with open(CACHE, "wb") as f:
            pickle.dump(cache, f)
    else:
        print("loading cache2", flush=True)
        cache = pickle.loads(CACHE.read_bytes())

    print("== counts =="); explain_counts()
    print("\n== EXP1: coordinate identity check (GT->crop->640->restore->Dice) ==")
    t1 = exp1_identity_check()
    print(t1.to_string())
    t1.to_csv(OUT / "exp1_identity.csv")

    print("\n== EXP2+EXP4: merged(candidate-level) vs B2(instance-level) vs crop-space ==")
    E, B2 = exp2_and_exp4(cache)
    print("EXP2/4 per bucket (best candidate mask, c1=0.05 pool):")
    t = E.groupby("bucket")[["merged_best_iou", "crop_space_best_dice"]].agg(
        ["mean", "count"]).round(4)
    t2 = B2.groupby("bucket")[["inst_best_iou"]].agg(["mean", "count"]).round(4)
    t2.columns = ["inst_best_iou_mean", "n"]
    print(pd.concat([t, t2], axis=1).to_string())
    E.to_csv(OUT / "exp2_merged_vs_crop.csv", index=False)
    B2.to_csv(OUT / "exp4_b2_instance.csv", index=False)

    print("\n== EXP3: GT-bbox crop -> stage2 dice (crop space) ==")
    e3 = exp3_gtbox_crop(cache)
    tt = e3.groupby("bucket").agg(n=("dice_crop", "count"),
                                  dice_crop=("dice_crop", "mean"),
                                  conf=("conf", "mean")).round(4)
    print(tt.to_string())
    tt.to_csv(OUT / "exp3_gtbox_crop.csv")


if __name__ == "__main__":
    main()
