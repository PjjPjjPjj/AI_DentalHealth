"""Eka calculus-candidate mining with the v3c pipeline (step 3) + review
queue export (step 4).

Per work image (1536px):
  S1 v3c tiled (512px windows, stride 384, c1=0.02, cls0) -> merge NMS 0.5
  per box: S2 window side = clip(4*max(bw,bh), 256, 1024) -> resize 640 ->
           class-0 instances -> intra-window mask-NMS 0.5
  per instance: tooth-BB overlap (anatomy filter >=20% of mask), verifier
  prob, composite score = .5*s2 + .3*verifier + .2*s1
Review queue buckets by score: high >=0.55, mid 0.30-0.55, low 0.10-0.30,
each candidate = overlay crop PNG + binary mask PNG + review.csv row.

Outputs:
  outputs/eka_review_queue/{high,mid,low}/
  outputs/eka_review_queue/review.csv
  outputs/reports/eka_candidates_summary.json
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
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from scripts.v3_pipeline_lib import make_verifier, verifier_probs
from src.inference.fusion import greedy_nms

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

WORK = Path(f"{_ROOT}/datasets/processed/eka_work")
QUEUE = Path(f"{_ROOT}/outputs/eka_review_queue")
S1_W = "outputs/calculus_v3c_stage1/weights/best.pt"
S2_W = "outputs/calculus_v3b_stage2/weights/best.pt"
C1 = 0.02
TILE = 512
STRIDE = 384
TARGET = 640
NMS_IOU = 0.5
ANAT_MIN = 0.20


def tiles_for(W, H, tile, stride):
    xs = sorted(set(list(range(0, max(W - tile, 0) + 1, stride)) + [max(W - tile, 0)]))
    ys = sorted(set(list(range(0, max(H - tile, 0) + 1, stride)) + [max(H - tile, 0)]))
    return [(x, y, min(x + tile, W), min(y + tile, H)) for y in ys for x in xs]


def nms_xyxy(boxes, confs, thr=0.5):
    if len(boxes) == 0:
        return boxes, confs, np.zeros(0, int)
    order = np.argsort(-confs)
    keep = []
    for i in order:
        b = boxes[i]
        ok = True
        for j in keep:
            bj = boxes[j]
            ix0, iy0 = max(b[0], bj[0]), max(b[1], bj[1])
            ix1, iy1 = min(b[2], bj[2]), min(b[3], bj[3])
            inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            ua = ((b[2] - b[0]) * (b[3] - b[1])
                  + (bj[2] - bj[0]) * (bj[3] - bj[1]) - inter)
            if ua > 0 and inter / ua >= thr:
                ok = False
                break
        if ok:
            keep.append(i)
    keep = np.array(sorted(keep), int)
    return boxes[keep], confs[keep], keep


def extract_cls0(res, ox=0.0, oy=0.0):
    if res.boxes is None or len(res.boxes) == 0 or res.masks is None:
        return (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), [])
    cls = res.boxes.cls.cpu().numpy().astype(int)
    keep = np.where(cls == 0)[0]
    if len(keep) == 0:
        return (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), [])
    mk = res.masks.data.cpu().numpy()
    boxes = res.boxes.xyxy.cpu().numpy()[keep].copy()
    boxes[:, [0, 2]] += ox
    boxes[:, [1, 3]] += oy
    return (boxes, res.boxes.conf.cpu().numpy()[keep],
            [(mk[i] > 0.5).astype(np.uint8) for i in keep])


def s1_tiled(model, img):
    H, W = img.shape[:2]
    all_b, all_c, all_m = [], [], []
    tmp = "/tmp/opencode/_eka_tile.jpg"
    for (x0, y0, x1, y1) in tiles_for(W, H, TILE, STRIDE):
        cv2.imwrite(tmp, img[y0:y1, x0:x1])
        res = model.predict(tmp, conf=C1, iou=0.6, imgsz=TARGET, device=0,
                            retina_masks=True, max_det=40, verbose=False)[0]
        b, c, m = extract_cls0(res, float(x0), float(y0))
        if len(b):
            all_b.append(b); all_c.append(c); all_m.extend(m)
    if not all_b:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), []
    boxes = np.concatenate(all_b, 0)
    confs = np.concatenate(all_c, 0)
    boxes, confs, keep = nms_xyxy(boxes, confs, 0.5)
    return boxes, confs, [all_m[i] for i in keep]


def main() -> None:
    tt = pd.read_csv(WORK / "tooth_table.csv")
    for b in ("high", "mid", "low"):
        (QUEUE / b).mkdir(parents=True, exist_ok=True)
    s1 = YOLO(S1_W)
    s2 = YOLO(S2_W)
    vb = make_verifier()

    rows = []
    n_done = 0
    for r in tt.itertuples():
        img = cv2.imread(r.work_path)
        if img is None:
            continue
        H, W = img.shape[:2]
        try:
            teeth = json.loads(r.teeth_json) if isinstance(r.teeth_json, str) else []
        except Exception:
            teeth = []
        tooth_masks = []
        for t in teeth:
            bx = t.get("Bounding_Box_Coordinates")
            if not bx or len(bx) != 4:
                continue
            x, y, w, h = bx
            x0, y0 = int(x * W), int(y * H)
            m = np.zeros((H, W), np.uint8)
            m[y0:y0 + int(h * H), x0:x0 + int(w * W)] = 1
            tooth_masks.append((m, t.get("Class_ID")))

        boxes, confs, masks = s1_tiled(s1, img)
        cands = []
        for bi, (b, cf) in enumerate(zip(boxes, confs)):
            bw, bh = b[2] - b[0], b[3] - b[1]
            side = float(np.clip(4 * max(bw, bh), 256, 1024))
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            x0 = int(np.clip(cx - side / 2, 0, max(W - side, 0)))
            y0 = int(np.clip(cy - side / 2, 0, max(H - side, 0)))
            x1, y1 = min(x0 + int(side), W), min(y0 + int(side), H)
            win = img[y0:y1, x0:x1]
            if win.size == 0:
                continue
            crop = cv2.resize(win, (TARGET, TARGET), interpolation=cv2.INTER_AREA)
            res = s2.predict(crop, conf=0.05, iou=0.6, imgsz=TARGET, device=0,
                             retina_masks=True, max_det=40, verbose=False)[0]
            if res.boxes is None or len(res.boxes) == 0 or res.masks is None:
                continue
            cls = res.boxes.cls.cpu().numpy().astype(int)
            mk = res.masks.data.cpu().numpy()
            sel = []
            for i in np.where(cls == 0)[0]:
                mi = (mk[i] > 0.5).astype(np.uint8)
                if mi.sum() == 0:
                    continue
                sel.append({"x0": 0, "y0": 0, "cw": TARGET, "ch": TARGET,
                            "bin": mi, "s2_conf": float(res.boxes.conf[i]),
                            "score": float(res.boxes.conf[i])})
            keep = greedy_nms(sel, "mask_nms", NMS_IOU, TARGET, TARGET)
            for k in keep:
                P640 = k["bin"]
                Pwin = (cv2.resize(P640, (x1 - x0, y1 - y0),
                                   interpolation=cv2.INTER_NEAREST) > 0)
                # tooth overlap
                best_tooth, gin = 0.0, -1
                Psum = max(int(Pwin.sum()), 1)
                for tm, gid in tooth_masks:
                    ov = int((Pwin & (tm[y0:y1, x0:x1] > 0)).sum()) / Psum
                    if ov > best_tooth:
                        best_tooth, gin = ov, gid
                cands.append({"bi": bi, "box": [float(x0), float(y0),
                                                float(x1), float(y1)],
                              "s1_conf": float(cf), "s2_conf": float(k["s2_conf"]),
                              "mask_win": Pwin, "win": (x0, y0, x1, y1),
                              "tooth_ov": best_tooth, "gin_class": gin})
        if cands:
            vp = verifier_probs(vb, [{"crop_img": img[c["win"][1]:c["win"][3],
                                                  c["win"][0]:c["win"][2]],
                                      "mask160": c["mask_win"]} for c in cands])
            for c, v in zip(cands, vp):
                c["verifier_prob"] = float(v)
                c["score"] = round(0.5 * c["s2_conf"] + 0.3 * float(v)
                                   + 0.2 * c["s1_conf"], 4)
        for ci, c in enumerate(cands):
            if c["tooth_ov"] < ANAT_MIN:
                bucket = None
            elif c["score"] >= 0.55:
                bucket = "high"
            elif c["score"] >= 0.30:
                bucket = "mid"
            elif c["score"] >= 0.10:
                bucket = "low"
            else:
                bucket = None
            if bucket is None:
                continue
            cid = f"{r.md5[:12]}_{c['bi']}_{ci}"
            x0, y0, x1, y1 = c["win"]
            ov = img[y0:y1, x0:x1].copy()
            mm = c["mask_win"].astype(np.uint8)
            cont, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(ov, cont, -1, (0, 0, 255), 2)
            cv2.putText(ov, f"{c['score']:.2f}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imwrite(str(QUEUE / bucket / f"{cid}_overlay.jpg"), ov)
            cv2.imwrite(str(QUEUE / bucket / f"{cid}_mask.png"),
                        (mm * 255).astype(np.uint8))
            rows.append({"cand_id": cid, "md5": r.md5, "orig_split": r.orig_split,
                         "work_path": r.work_path,
                         "bucket": bucket, "score": c["score"],
                         "s1_conf": c["s1_conf"], "s2_conf": c["s2_conf"],
                         "verifier_prob": c["verifier_prob"],
                         "tooth_ov": round(c["tooth_ov"], 3),
                         "gin_class": c["gin_class"],
                         "box_work": json.dumps(c["box"])})
        n_done += 1
        if n_done % 100 == 0:
            print(f"{n_done}/{len(tt)} images, {len(rows)} candidates",
                  flush=True)
    rdf = pd.DataFrame(rows)
    rdf.to_csv(QUEUE / "review.csv", index=False)
    summary = {
        "images_processed": n_done,
        "candidates_total": int(len(rdf)),
        "buckets": rdf.bucket.value_counts().to_dict() if len(rdf) else {},
        "images_with_high": int(rdf[rdf.bucket == 'high'].md5.nunique()) if len(rdf) else 0,
        "images_with_any": int(rdf.md5.nunique()) if len(rdf) else 0,
        "anatomy_filtered_out": "tooth_ov<0.20 dropped",
    }
    (Path(f"{_ROOT}/outputs/reports/eka_candidates_summary.json")
     .write_text(json.dumps(summary, indent=2)))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
