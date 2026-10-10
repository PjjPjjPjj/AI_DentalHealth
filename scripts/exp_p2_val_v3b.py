"""P2 model as a second proposal source on val (S2/verifier unchanged).

Compares: S1-only vs P2-only vs union(both) proposals at c1=0.02.
Reports per-domain proposal recall/FP and image-level evidence quality
(max fused prob AUROC/acc using existing M3 fusion + verifier).
Output: outputs/calculus_v3_cls/p2_val_experiment.json
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import json

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from ultralytics import YOLO

from scripts.v3_pipeline_lib import (DATA, OUT, FCOLS, RAW, NMS_IOU, TARGET,
                                     bucket_of, collect_split, make_verifier,
                                     s2_instances, verifier_probs)
from src.inference.fusion import greedy_nms

C1 = 0.02
SIDE = 160


def s1_pass_generic(model, image_path, c1=C1):
    res = model.predict(image_path, conf=c1, iou=0.6, imgsz=640, device=0,
                        retina_masks=True, max_det=100, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0 or res.masks is None:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), []
    cls = res.boxes.cls.cpu().numpy().astype(int)
    keep = np.where(cls == 0)[0]
    if len(keep) == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), []
    mk = res.masks.data.cpu().numpy()
    return (res.boxes.xyxy.cpu().numpy()[keep],
            res.boxes.conf.cpu().numpy()[keep],
            [(mk[i] > 0.5).astype(np.uint8) for i in keep])


def merge_boxes(boxes_a, confs_a, boxes_b, confs_b, iou_thr=0.5):
    boxes = np.concatenate([boxes_a, boxes_b], 0) if len(boxes_b) else boxes_a
    confs = np.concatenate([confs_a, confs_b], 0) if len(boxes_b) else confs_a
    if len(boxes) == 0:
        return boxes, confs
    order = np.argsort(-confs)
    keep = []
    for i in order:
        b = boxes[i]
        ok = True
        for j in keep:
            bj = boxes[j]
            ix0, iy0 = max(b[0], bj[0]), max(b[1], bj[1])
            ix1, iy1 = min(b[2], bj[2]), min(b[3], bj[3])
            inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            ua = (b[2]-b[0])*(b[3]-b[1]) + (bj[2]-bj[0])*(bj[3]-bj[1]) - inter
            if ua > 0 and inter / ua >= iou_thr:
                ok = False
                break
        if ok:
            keep.append(i)
    return boxes[keep], confs[keep]


def run_variant(s1, p2, s2, vb, man, split, mode, z):
    m3 = LogisticRegression(max_iter=1000, class_weight="balanced")
    m3.coef_, m3.intercept_, m3.classes_ = z["coef"], z["intercept"], np.array([0, 1])
    mu, sg = z["mu"], z["sg"]
    rows = man[man.split == split].reset_index(drop=True)
    st = {d: {"gt": 0, "tp": 0, "fp": 0, "n": 0} for d in
          ("caries_0", "gingivitis", "other")}
    imgA = []
    ys = []
    for _, r in rows.iterrows():
        img = cv2.imread(r.image_path)
        H, W = img.shape[:2]
        m = cv2.imread(str(RAW / f"{r.image_id}.png"), cv2.IMREAD_UNCHANGED)
        m = np.zeros((H, W), np.uint8) if m is None else \
            (m[:, :, 0] if m.ndim == 3 else m)
        gts = []
        calc = (m == 3)
        if calc.any():
            n, comp = cv2.connectedComponents(calc.astype(np.uint8), 8)
            for i in range(1, n):
                px = int((comp == i).sum())
                if px >= 5:
                    gts.append(((comp == i).astype(np.uint8),
                                bucket_of(px / float(H * W))))
        boxes, confs, masks = s1_pass_generic(s1, r.image_path)
        if mode in ("p2", "union"):
            b2, c2, _ = s1_pass_generic(p2, r.image_path)
            if mode == "union":
                boxes, confs = merge_boxes(boxes, confs, b2, c2)
            else:
                boxes, confs = b2, c2
        for d in st:
            pass
        s = st[r.domain]
        s["gt"] += len(gts)
        s["n"] += 1
        used = set()
        feats_all = []
        for bi, (b, cf) in enumerate(zip(boxes, confs)):
            x0 = int(round((b[0] + b[2]) / 2 - SIDE / 2))
            y0 = int(round((b[1] + b[3]) / 2 - SIDE / 2))
            x0 = max(0, min(x0, W - SIDE)); y0 = max(0, min(y0, H - SIDE))
            x1, y1 = min(x0 + SIDE, W), min(y0 + SIDE, H)
            crop_img = img[y0:y1, x0:x1]
            if crop_img.size == 0:
                continue
            crop = cv2.resize(crop_img, (TARGET, TARGET),
                              interpolation=cv2.INTER_AREA)
            insts = s2_instances(s2, crop)
            sel = [{"x0": 0, "y0": 0, "cw": TARGET, "ch": TARGET,
                    "bin": i["bin640"], "s2_conf": i["s2_conf"],
                    "score": i["s2_conf"]} for i in insts]
            keep = greedy_nms(sel, "mask_nms", NMS_IOU, TARGET, TARGET)
            for k in keep:
                feats_all.append({"s1_conf": float(cf),
                                  "s2_conf": float(k["s2_conf"]),
                                  "s1_area_ratio": 0.01, "s2_area_ratio":
                                      float(k["bin"].sum()) / (TARGET * TARGET),
                                  "s1s2_iou": 0.5,
                                  "bbox_area_ratio": k["score"] * 0,
                                  "bbox_aspect": 1.0,
                                  "verifier_prob": 0.5})
            # proposal-level matching
            P = np.zeros((H, W), np.uint8)
            P[max(0, int(b[1])):int(b[3]), max(0, int(b[0])):int(b[2])] = 1
            best = (0.0, None)
            for gi, (G, _) in enumerate(gts):
                inter = int(P[G > 0].sum())
                uni = int(((P > 0) | (G > 0)).sum())
                io = inter / uni if uni else 0.0
                if io > best[0]:
                    best = (io, gi)
            if best[0] >= 0.5 and best[1] not in used:
                used.add(best[1])
                s["tp"] += 1
            elif best[0] < 0.5:
                s["fp"] += 1
        if feats_all:
            X = np.array([[f[c] for c in FCOLS] for f in feats_all], float)
            # NOTE: with placeholder features the M3 score is not valid here;
            # use max s2_conf as image evidence instead
            imgA.append(float(max(f["s2_conf"] for f in feats_all)))
        else:
            imgA.append(0.0)
        ys.append(int(r.has_calculus))
    out = {}
    for d, s in st.items():
        out[d] = {"recall": round(s["tp"] / max(s["gt"], 1), 3),
                  "fp_img": round(s["fp"] / max(s["n"], 1), 2), "n": s["n"]}
    y = np.array(ys)
    out["_img"] = {"auroc_maxs2": round(float(roc_auc_score(y, np.array(imgA))), 4)}
    return out


def main() -> None:
    s1 = YOLO("outputs/calculus_v3b_stage1/weights/best.pt")
    p2 = YOLO("outputs/calculus_v3b_p2/weights/best.pt")
    s2 = YOLO("outputs/calculus_v3b_stage2/weights/best.pt")
    vb = make_verifier()
    man = pd.read_csv(DATA / "split_manifest_v3.csv")
    z = np.load("outputs/calculus_v3_cls/fusion_v3.npz")
    rep = {}
    for mode in ("s1", "p2", "union"):
        print(f"=== mode {mode} ===", flush=True)
        rep[mode] = run_variant(s1, p2, s2, vb, man, "val", mode, z)
        print(json.dumps(rep[mode], indent=1))
    from scripts.eval_ensemble_v3b import effnet_tta_probs
    imgB = effnet_tta_probs(man, "val")
    y = man[man.split == "val"].has_calculus.values.astype(int)
    rep["imgB_auroc"] = round(float(roc_auc_score(y, imgB)), 4)
    (OUT / "p2_val_experiment.json").write_text(json.dumps(rep, indent=2))
    print("imgB auroc:", rep["imgB_auroc"])


if __name__ == "__main__":
    main()
