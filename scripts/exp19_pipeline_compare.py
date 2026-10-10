"""Plan B STEP8-9: A/B/C/D strict comparison on main VAL (no retraining).

A: Stage1 c1=0.05 -> Stage2 V1
B: c1=0.05 -> classifier(thr .05/.10) -> Stage2 V1
C: c1=0.02 -> Stage2 V1
D: c1=0.02 -> classifier(thr .05/.10) -> Stage2 V1

Stage2 = V1 best, imgsz 640, independent instances, mask-NMS 0.5.
c2 coarse grid {0.30,0.40,0.50,0.60}. TP-dice sanity assert >= 0.667.
Classified crop = per stage1 box (bbox-center 160); classifier = trained
EfficientNetV2-S (outputs/calculus_cls/best.pt), never touching val/test.
Error analysis for B/D: TP_keep / TP_reject / FP_keep / FP_reject +
TP_reject details (bucket, stage1_conf, cls score, GT area).
Outputs: outputs/calculus_cls/pipeline_compare.csv + summary printed.
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

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision import models, transforms

from src.inference.fusion import greedy_nms, rec_bin
from src.inference.twostage import TARGET, TwoStageSegmenter
from src.utils.paths import PROCESSED_DIR

DATA = PROCESSED_DIR / "calculus_seg"
OUT = Path("outputs/calculus_cls")
C1S = (0.05, 0.02)
C2_GRID = (0.30, 0.40, 0.50, 0.60)
THRS = (0.05, 0.10)
NMS_IOU = 0.5
SIDE = 160
IOU_MATCH = 0.5
BUCKET_EDGES = (0.00552, 0.01565)
BUCKETS = ("small", "medium", "large")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def bucket_of(r: float) -> str:
    if r < BUCKET_EDGES[0]:
        return "small"
    if r < BUCKET_EDGES[1]:
        return "medium"
    return "large"


def load_cls() -> torch.nn.Module:
    m = models.efficientnet_v2_s(
        weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, 1)
    m.load_state_dict(torch.load(OUT / "best.pt", weights_only=False))
    m = m.to(0).eval()
    return m


def main() -> None:
    seg = TwoStageSegmenter("outputs/calculus_yolo11m_full/weights/best.pt",
                            "outputs/calculus_yolo11m_crop/weights/best.pt",
                            c1=C1S[0], c2=0.05, device=0, fusion="mask_nms",
                            nms_iou=NMS_IOU, score_mode="s2")
    cls = load_cls()
    tf = transforms.Compose([transforms.ToPILImage(),
                             transforms.Resize((224, 224)),
                             transforms.ToTensor(),
                             transforms.Normalize(MEAN, STD)])
    man = pd.read_csv(DATA / "split_manifest.csv")
    rows = man[man.split == "val"]
    n_img = 0
    stage2_cache: dict = {}  # boxkey -> recs(instances for that crop)
    boxes_of = {}  # c1 -> per image box lists
    all_cls_inputs, all_cls_keys = [], []

    for _, r in rows.iterrows():
        img = cv2.imread(r["image_path"])
        H, W = img.shape[:2]
        gtb = cv2.imread(r["mask_original_path"], cv2.IMREAD_UNCHANGED)
        has = bool(r["has_calculus"])
        gts = []
        if has and gtb is not None and (gtb > 0).any():
            gtb = gtb > 0
            n, comp, _, _ = cv2.connectedComponentsWithStats(gtb.astype(np.uint8), 8)
            for i in range(1, n):
                if (comp == i).sum() < 5:
                    continue
                m = (comp == i).astype(np.uint8)
                px = int(m.sum())
                gts.append({"mask": m, "bucket": bucket_of(px / float(H * W))})
        for c1 in C1S:
            seg.c1 = c1
            boxes, confs = seg._stage1_full(r["image_path"])
            boxes_of.setdefault(c1, {})[r["image_id"]] = (boxes, confs)
            for b, sc1 in zip(boxes, confs):
                key = (r["image_id"], tuple(int(v) for v in b))
                if key in stage2_cache:
                    continue
                cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
                x0 = int(round(cx - SIDE / 2)); y0 = int(round(cy - SIDE / 2))
                x0 = max(0, min(x0, W - SIDE)); y0 = max(0, min(y0, H - SIDE))
                x1, y1 = min(x0 + SIDE, W), min(y0 + SIDE, H)
                ch, cw = y1 - y0, x1 - x0
                crop_img = img[y0:y1, x0:x1]
                crop = cv2.resize(crop_img, (TARGET, TARGET),
                                  interpolation=cv2.INTER_AREA)
                t0 = time.time()
                insts = seg._stage2_instances(crop)
                stage2_ms = (time.time() - t0) * 1000
                recs = []
                for inst in insts:
                    mi = cv2.resize(inst["bin640"], (cw, ch),
                                    interpolation=cv2.INTER_NEAREST)
                    recs.append({"x0": x0, "y0": y0, "cw": cw, "ch": ch,
                                 "bin": mi, "conf": inst["s2_conf"]})
                stage2_cache[key] = {"recs": recs, "ms": stage2_ms}
                # cls input (raw crop)
                all_cls_keys.append(key)
                all_cls_inputs.append(tf(crop_img))
        n_img += 1
    # batch classify
    scores_all = {}
    batch_cnt = 0
    with torch.no_grad():
        for i in range(0, len(all_cls_inputs), 128):
            xb = torch.stack(all_cls_inputs[i:i + 128]).to(0)
            sc = cls(xb).squeeze(1).sigmoid().cpu().tolist()
            for k, s in zip(all_cls_keys[i:i + 128], sc):
                scores_all[k] = s
            batch_cnt += 1

    # per-image GT masks cached
    # evaluate schemes
    rows_out = []
    schemes = [("A", None), ("B", 0.05), ("B", 0.10)]
    for c1 in C1S:
        base = "A" if c1 == 0.05 else "C"
        for name, thr in [("A", None)] if c1 == 0.05 else \
                ([("D", 0.05), ("D", 0.10)] if False else []):
            pass
    cfg_list = [(name, c1, thr, cc) for name, c1, thr in
                [("A", 0.05, None), ("B", 0.05, 0.05), ("B", 0.05, 0.10),
                 ("C", 0.02, None), ("D", 0.02, 0.05), ("D", 0.02, 0.10)]
                for cc in C2_GRID]
    for name, c1, thr, c in cfg_list:
        tp = {b: 0 for b in BUCKETS}
        gt_n = {b: 0 for b in BUCKETS}
        tp_dice, gt_dice, sem = [], [], []
        n_preds = n_unm = 0
        tim = 0.0
        for _, r in rows.iterrows():
            imid = r["image_id"]
            boxes, confs = boxes_of[c1][imid]
            img = cv2.imread(r["image_path"])
            H, W = img.shape[:2]
            gtb = cv2.imread(r["mask_original_path"], cv2.IMREAD_UNCHANGED)
            gts = []
            if gtb is not None and (gtb > 0).any():
                gtb = gtb > 0
                n, comp, _, _ = cv2.connectedComponentsWithStats(
                    gtb.astype(np.uint8), 8)
                for i in range(1, n):
                    if (comp == i).sum() < 5:
                        continue
                    m = (comp == i).astype(np.uint8)
                    px = int(m.sum())
                    gts.append({"mask": m,
                                "bucket": bucket_of(px / float(H * W))})
            recs_all = []
            for b, sc1 in zip(boxes, confs):
                key = (imid, tuple(int(v) for v in b))
                chk = stage2_cache.get(key)
                if thr is not None and scores_all.get(key, 0.0) < thr:
                    continue
                tim += chk["ms"]
                for x in chk["recs"]:
                    if x["conf"] >= c:
                        recs_all.append({"x0": x["x0"], "y0": x["y0"],
                                         "cw": x["cw"], "ch": x["ch"],
                                         "bin": x["bin"], "score": x["conf"]})
            keep = greedy_nms(recs_all, "mask_nms", NMS_IOU, H, W)
            preds = [rec_bin(k, H, W) for k in keep]
            n_preds += len(preds)
            pairs = []
            for pi, P in enumerate(preds):
                for gi, G in enumerate(gts):
                    inter = int((P & G["mask"]).sum())
                    uni = int((P | G["mask"]).sum())
                    io = inter / uni if uni else 0.0
                    if io >= IOU_MATCH:
                        pairs.append((io, pi, gi))
            pairs.sort(reverse=True)
            used_p, used_g, matched = set(), set(), []
            for io, pi, gi in pairs:
                if pi in used_p or gi in used_g:
                    continue
                used_p.add(pi); used_g.add(gi)
                matched.append((pi, gi, io))
            n_unm += len(preds) - len(matched)
            for pi, gi, io in matched:
                P = preds[pi]; G = gts[gi]
                inter = int((P & G["mask"]).sum())
                dc = 2 * inter / (int(P.sum()) + int(G["mask"].sum())) \
                    if (int(P.sum()) + int(G["mask"].sum())) else 1.0
                assert dc >= 0.6667 - 1e-3
                tp[G["bucket"]] += 1
                tp_dice.append(dc)
            union = np.zeros((H, W), np.uint8)
            for P in preds:
                union |= P
            for gi, G in enumerate(gts):
                gt_n[G["bucket"]] += 1
                inter = int((G["mask"] & union).sum())
                uni = int((G["mask"] | union).sum())
                io = inter / uni if uni else 0.0
                dc = 2 * inter / (int(G["mask"].sum()) + int(union.sum())) \
                    if (int(G["mask"].sum()) + int(union.sum())) else 1.0
                sem.append(dc)
                gt_dice.append(dc if gi in [g for _, g, _ in matched] else 0.0)
        tp_tot = sum(tp.values())
        gt_tot = sum(gt_n.values()) if not gts else sum(gt_n.values())
        rec = tp_tot / gt_tot if gt_tot else 0.0
        prec = tp_tot / n_preds if n_preds else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows_out.append({"scheme": name, "c1": c1, "clas_thr": thr,
                         "precision": round(prec, 4), "recall": round(rec, 4),
                         "f1": round(f1, 4),
                         "tp_dice": round(float(np.mean(tp_dice)), 4),
                         "gt_dice": round(float(np.mean(gt_dice)), 4),
                         "sem_dice": round(float(np.mean(sem)), 4),
                         "fp_img": round(n_unm / n_img, 3),
                         "pred_img": round(n_preds / n_img, 3),
                         "r_small": round(tp["small"] / gt_n["small"], 4),
                         "r_medium": round(tp["medium"] / gt_n["medium"], 4),
                         "r_large": round(tp["large"] / gt_n["large"], 4),
                         "ms_img": round(tim / n_img, 1)})
    df = pd.DataFrame(rows_out)
    df.to_csv(OUT / "pipeline_compare.csv", index=False)
    print(df.round(4).to_string(index=False))

    # error analysis per B/D thr
    ea_rows = []
    for c1, thr in [(0.05, 0.05), (0.05, 0.10), (0.02, 0.05), (0.02, 0.10)]:
        tpk = tpr = fpk = fpr = 0
        rej = []
        for _, r in rows.iterrows():
            imid = r["image_id"]
            img = cv2.imread(r["image_path"])
            H, W = img.shape[:2]
            boxes, confs = boxes_of[c1][imid]
            gtb = cv2.imread(r["mask_original_path"], cv2.IMREAD_UNCHANGED)
            neg_mask = (gtb if gtb is not None else np.zeros((H, W), np.uint8)) > 0
            for b, sc1 in zip(boxes, confs):
                key = (imid, tuple(int(v) for v in b))
                keep = scores_all.get(key, 0.0) >= thr
                bx = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                has_gt = neg_mask[bx[1]:bx[3], bx[0]:bx[2]].any()
                if has_gt:
                    if keep:
                        tpk += 1
                    else:
                        tpr += 1
                        # details for reject
                        if gtb is not None:
                            gt_nfull = int((gtb > 0)[bx[1]:bx[3], bx[0]:bx[2]].sum())
                        else:
                            gt_nfull = 0
                        rej.append({"c1": c1, "thr": thr, "key": key,
                                    "has_small": bool((gtb > 0)[bx[1]:bx[3], bx[0]:bx[2]].any()) if gtb is not None else False,
                                    "s1_conf": round(float(sc1), 4),
                                    "cls": round(float(scores_all.get(key, 0.0)), 4),
                                    "gt_px_in_crop": gt_nfull})
                else:
                    if keep:
                        fpk += 1
                    else:
                        fpr += 1
        ea_rows.append({"scheme": "B" if c1 == 0.05 else "D", "c1": c1,
                        "thr": thr, "TP_keep": tpk, "TP_reject": tpr,
                        "FP_keep": fpk, "FP_reject": fpr, "rej": rej})
    eadf = pd.DataFrame(ea_rows)
    eadf.to_csv(OUT / "classifier_error_analysis.csv", index=False)
    print("\nclassifier keep/reject analysis:")
    print(eadf[["scheme", "c1", "thr", "TP_keep", "TP_reject",
                "FP_keep", "FP_reject"]].to_string(index=False))
    print("\nTP_reject details:")
    for _, row in eadf.iterrows():
        rj = row["rej"]
        if not rj:
            print(f"  {row['scheme']} c1={row['c1']} thr={row['thr']}: no rejects")
            continue
        gtx = np.array([x["gt_px_in_crop"] for x in rj])
        s1c = np.array([x["s1_conf"] for x in rj])
        cl = np.array([x["cls"] for x in rj])
        small_frac = float((gtx < 400).mean())
        print(f"  {row['scheme']} c1={row['c1']} thr={row['thr']}: n={len(rj)} "
              f"gt_px_in_crop med={np.median(gtx):.0f} p25={np.percentile(gtx,25):.0f} "
              f"p75={np.percentile(gtx,75):.0f} frac(gt_px<400)={small_frac:.2f} "
              f"s1_conf med={np.median(s1c):.3f} cls med={np.median(cl):.3f}")


if __name__ == "__main__":
    main()
