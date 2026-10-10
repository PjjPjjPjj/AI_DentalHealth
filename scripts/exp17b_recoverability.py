"""EXP17B: Stage2-recoverability audit of EXP17 candidate groups.

For each of the 1197 Stage1 positive candidates (clear/partial/bad by
best Stage1 mask-IoU with GT), run the REAL Stage2 V1 on the bbox-centered
160 crop, keep independent instances (conf>=0.05 for capability; 0.40
variant also recorded), 1-to-1 greedy match against crop GT instances in
640-space, record best Stage2 IoU & matched Dice.

Reports per group (and per group x size bucket): n, Stage2 TP
(best IoU>=0.5), TP rate, best-IoU stats, TP-only Dice; key outputs:
P(Stage2 TP | bad), P(Stage2 TP | partial).
Outputs: outputs/calculus_cls/exp17b_recoverability.csv
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

from src.inference.twostage import TARGET, TwoStageSegmenter
from src.utils.paths import PROCESSED_DIR

DATA = PROCESSED_DIR / "calculus_seg"
OUT = Path("outputs/calculus_cls")
C1 = 0.05
SIDE = 160
BUCKET_EDGES = (0.00552, 0.01565)
BUCKETS = ("small", "medium", "large")


def bucket_of(r: float) -> str:
    if r < BUCKET_EDGES[0]:
        return "small"
    if r < BUCKET_EDGES[1]:
        return "medium"
    return "large"


def instances_full(mask: np.ndarray, min_area=5):
    n, comp, _, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        if (comp == i).sum() < min_area:
            continue
        out.append((comp == i).astype(np.uint8))
    return out


def main() -> None:
    seg = TwoStageSegmenter("outputs/calculus_yolo11m_full/weights/best.pt",
                            "outputs/calculus_yolo11m_crop/weights/best.pt",
                            c1=C1, c2=0.05, device=0, nms_iou=0.5, score_mode="s2")
    man = pd.read_csv(DATA / "split_manifest.csv")
    rows = man[man.split == "train"]
    recs = []
    for ri, (_, r) in enumerate(rows.iterrows()):
        img = cv2.imread(r["image_path"])
        H, W = img.shape[:2]
        gt = cv2.imread(r["mask_original_path"], cv2.IMREAD_UNCHANGED)
        gtb = (gt > 0) if gt is not None else np.zeros((H, W), np.uint8)
        full_gts = instances_full(gtb)
        boxes, confs, masks = seg._stage1_masks(r["image_path"])
        for (b, _sc1, msk) in zip(boxes, confs, masks):
            if msk is None or len(msk.shape) > 2 or msk.sum() == 0:
                continue
            msk = msk.astype(np.uint8)
            best_s1 = (0.0, None)
            for fm in full_gts:
                inter = int((msk & fm).sum())
                uni = int((msk | fm).sum())
                mio = inter / uni if uni else 0.0
                if mio > best_s1[0]:
                    best_s1 = (mio, fm)
            bx = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
            x0 = int(round((bx[0] + bx[2]) / 2 - SIDE / 2))
            y0 = int(round((bx[1] + bx[3]) / 2 - SIDE / 2))
            x0 = max(0, min(x0, W - SIDE)); y0 = max(0, min(y0, H - SIDE))
            x1, y1 = x0 + SIDE, y0 + SIDE
            crop_gts = [fm for fm in full_gts
                        if int(fm[y0:y1, x0:x1].sum()) > 0]
            max_inc = max((float(int(fm[y0:y1, x0:x1].sum()) / fm.sum())
                           for fm in crop_gts), default=0.0)
            if max_inc < 0.8:
                continue  # same positive rule as classifier dataset
            mIoU = best_s1[0]
            group = ("clear" if mIoU >= 0.3 else
                     ("partial" if mIoU >= 0.1 else "bad"))
            # GT instances in crop -> 640 space components
            gtc = gtb[y0:y1, x0:x1].astype(np.uint8)
            ng, comp, _, _ = cv2.connectedComponentsWithStats(gtc, 8)
            gt_insts = []
            bucket_tags = []
            for i in range(1, ng):
                if (comp == i).sum() < 5:
                    continue
                m = (comp == i).astype(np.uint8)
                gm640 = cv2.resize(m, (TARGET, TARGET),
                                   interpolation=cv2.INTER_AREA) > 0.5
                gt_insts.append(gm640.astype(np.uint8))
                # bucket from full GT px of best overlapping full mask
                px = int(m.sum()) * (H * W) / (SIDE * SIDE)
                ratio = px / float(H * W)
                bucket_tags.append(bucket_of(ratio))
            crop = cv2.resize(img[y0:y1, x0:x1], (TARGET, TARGET),
                              interpolation=cv2.INTER_AREA)
            insts = seg._stage2_instances(crop)
            # greedy 1-to-1 matching (best IoU first)
            pairs = []
            for pi, inst in enumerate(insts):
                for gi, gm in enumerate(gt_insts):
                    inter = int((inst["bin640"] & gm).sum())
                    uni = int((inst["bin640"] | gm).sum())
                    io = inter / uni if uni else 0.0
                    if io >= 0.3:  # consider all with note
                        pairs.append((io, pi, gi))
            pairs.sort(reverse=True)
            used_p, used_g, matched = set(), set(), []
            for io, pi, gi in pairs:
                if pi in used_p or gi in used_g:
                    continue
                used_p.add(pi); used_g.add(gi)
                matched.append((pi, gi, io))
            best_io = max([m[2] for m in matched], default=0.0)
            best_dc = 0.0
            if matched:
                pi, gi = matched[0][0], matched[0][1]
                P = insts[pi]["bin640"]; G = gt_insts[gi]
                inter = int((P & G).sum())
                best_dc = 2 * inter / (int(P.sum()) + int(G.sum())) \
                    if (int(P.sum()) + int(G.sum())) else 1.0
            tp05 = int(best_io >= 0.5)
            tp05_40 = int(best_io >= 0.5 and max(
                [i["s2_conf"] for i in insts], default=0.0) >= 0.40)
            bucket = bucket_tags[0] if bucket_tags else "small"
            recs.append({"group": group, "bucket": bucket, "mIoU": mIoU,
                         "best_io": best_io, "best_dice": best_dc,
                         "tp05": tp05, "tp05_c240": tp05_40,
                         "max_conf": max([i["s2_conf"] for i in insts],
                                         default=0.0)})
        if (ri + 1) % 500 == 0:
            print(f"  {ri + 1}/{len(rows)}", flush=True)
    df = pd.DataFrame(recs)
    df.to_csv(OUT / "exp17b_recoverability.csv", index=False)
    print(f"\nEXP17B: candidates audited = {len(df)}")
    for g in ["clear", "partial", "bad"]:
        sub = df[df.group == g]
        tp = int(sub.tp05.sum())
        tp40 = int(sub.tp05_c240.sum())
        print(f"\n== {g} (n={len(sub)}) ==")
        print(f"   Stage2 TP rate (conf>=0.05): {tp / len(sub):.4f} "
              f"({tp}/{len(sub)}) | conf>=0.40: {tp40 / len(sub):.4f} ({tp40})")
        print(f"   best IoU mean/med: {sub.best_io.mean():.4f} / "
              f"{sub.best_io.median():.4f}; TP-only Dice: "
              f"{sub[sub.tp05 == 1].best_dice.mean():.4f}")
        bt = sub.groupby("bucket")["tp05"].agg(["sum", "count"])
        bt["rate"] = (bt["sum"] / bt["count"]).round(4)
        print("   per-bucket TP rate:", bt.to_dict("index"))
    print("\nP(Stage2 TP | bad) =", f"{df[df.group=='bad'].tp05.mean():.4f}")
    print("P(Stage2 TP | partial) =", f"{df[df.group=='partial'].tp05.mean():.4f}")


if __name__ == "__main__":
    main()
