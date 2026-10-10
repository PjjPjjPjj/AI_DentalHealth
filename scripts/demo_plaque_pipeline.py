# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
f"""Demo: single-tooth segmentation -> plaque classification, step-by-step images.

Picks the Mendeley test-base image with the most matched teeth (from
plaque_toothlevel_v2 records), then renders the deployed pipeline:

  step1_input.jpg               original intraoral photo
  step2_tooth_segmentation.jpg  iop_toothseg masks + tooth indices
  step3_tooth_crops.jpg         single-tooth crops = EXACT input of the
                                plaque classifier (saved individually too)
  step4_plaque_scores.jpg       per-tooth P(plaque) after cls+Platt
  step5_result.jpg              final annotated image + image-level stats
  README.md                     pipeline description (CN)

Run: PYTHONPATH={_ROOT} python3 scripts/demo_plaque_pipeline.py [image_id]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, f"{_ROOT}")
from scripts.eval_plaque_toothlevel_v2 import find_label, gt_teeth, match_gt_tooth  # noqa: E402
from src.inference.models import PlaqueModel, ToothModel  # noqa: E402

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

ROOT = Path(f"{_ROOT}")
REC = ROOT / "outputs/plaque_toothlevel_v2/tooth_records.csv"
OUT = ROOT / "outputs/demo_plaque_pipeline"
RAW_DIRS = [ROOT / "datasets/raw/Plaque/mendeley-dataset-materials_Part_1/data",
            ROOT / "datasets/raw/Plaque/mendeley-dataset-materials_Part_2/data"]
THR = 0.746          # deployed calibrated threshold
DISPLAY_W = 1600
GRID_N = 12
CELL = 260


def find_image(image_id: str) -> Path | None:
    for d in RAW_DIRS:
        for img in (d / "images").glob(f"*/{image_id}.jpg"):
            return img
    return None


def pick_image(requested: str | None) -> tuple[str, pd.DataFrame]:
    df = pd.read_csv(REC)
    df = df[(df.split == "test") & (df.matched == 1)]
    if requested:
        row = df[df.image_id == requested]
        if row.empty:
            raise SystemExit(f"image_id {requested} not in matched test records")
        return requested, row
    cnt = df.groupby("image_id").size().sort_values(ascending=False)
    return str(cnt.index[0]), df[df.image_id == cnt.index[0]]


def draw_text(img, text, org, color=(255, 255, 255), scale=0.8, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "crops").mkdir(exist_ok=True)
    iid, recs = pick_image(sys.argv[1] if len(sys.argv) > 1 else None)
    img_path = find_image(iid)
    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]
    sc = DISPLAY_W / W
    disp = cv2.resize(img, (DISPLAY_W, int(H * sc)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT / "step1_input.jpg"), disp)

    tm = ToothModel(weights=str(ROOT / "outputs/iop_toothseg/weights/best.pt"),
                    conf=0.30, iou=0.70, imgsz=896)
    pm = PlaqueModel(weights=str(ROOT / "outputs/plaque_bin_v2_sano/weights/best.pt"),
                     imgsz=224, windows=[[0, 0, 1, 1]],
                     calibration=str(ROOT / "outputs/plaque_bin_v2_sano/calibration_platt.npz"))

    teeth = tm.predict(img)
    tb = np.array([t["bbox"] for t in teeth], float)
    # order teeth left->right (view axis) for stable indexing
    order = np.argsort(tb[:, 0])
    crops, keeps = [], []
    for rank, ti in enumerate(order):
        x0, y0, x1, y1 = [int(max(0, v)) for v in tb[ti]]
        x1, y1 = min(W, x1), min(H, y1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        crops.append(img[y0:y1, x0:x1])
        keeps.append((rank, ti))
    probs = pm._proba_array(crops)
    # strict GT pairing: same protocol as eval (union-box IoU>=0.3 or containment);
    # segmented teeth without a GT unit are UNANNOTATED, never "plaque"
    gt_units = gt_teeth(find_label(iid), W, H) if find_label(iid) else {}
    gt_of_seg: dict[int, tuple[int, int]] = {}   # seg keep-idx -> (tooth_id, y_any)
    if gt_units:
        tb_all = np.array([teeth[ti]["bbox"] for _, ti in keeps], float)
        for tid, g in gt_units.items():
            j, miou, mtype = match_gt_tooth(g, tb_all)
            if mtype != "none":
                gt_of_seg[j] = (tid, g["y_any"])

    # ---- step2: segmentation overlay ----
    ov = disp.copy()
    rng = np.random.RandomState(7)
    for rank, ti in keeps:
        color = rng.randint(60, 255, 3).tolist()
        m = teeth[ti]["mask"]
        msmall = cv2.resize(m, (DISPLAY_W, int(H * sc)), interpolation=cv2.INTER_NEAREST) > 0
        ov[msmall] = (0.45 * np.array(color) + 0.55 * ov[msmall]).astype(np.uint8)
        x0, y0, x1, y1 = [int(v * sc) for v in tb[ti]]
        cv2.rectangle(ov, (x0, y0), (x1, y1), color, 2)
        draw_text(ov, f"T{rank + 1:02d}", (x0 + 4, max(24, y0 + 24)))
    cv2.imwrite(str(OUT / "step2_tooth_segmentation.jpg"), ov)

    # ---- step3: tooth crops (classifier input) ----
    keep_probs = [probs[k] for k, _ in enumerate(crops)]
    idx_sorted = np.argsort(keep_probs)[::-1][:GRID_N]
    cells = []
    for k in idx_sorted:
        rank, ti = keeps[k]
        c = crops[k]
        ch, cw = c.shape[:2]
        s = (CELL - 8) / max(ch, cw)
        c = cv2.resize(c, (max(1, int(cw * s)), max(1, int(ch * s))), interpolation=cv2.INTER_AREA)
        canvas = np.full((CELL, CELL, 3), 24, np.uint8)
        y0 = (CELL - c.shape[0]) // 2; x0 = (CELL - c.shape[1]) // 2
        canvas[y0:y0 + c.shape[0], x0:x0 + c.shape[1]] = c
        p = float(probs[k])
        col = (60, 60, 220) if p >= THR else (80, 190, 80)
        cv2.rectangle(canvas, (0, 0), (CELL - 1, CELL - 1), col, 5)
        draw_text(canvas, f"T{rank + 1:02d} P={p:.2f}", (6, 24), scale=0.55)
        g = gt_of_seg.get(k)
        if g is not None:
            y = g[1]
            draw_text(canvas, f"GT:{'plaque' if y else 'clean'}", (6, CELL - 10),
                      scale=0.5, color=(200, 200, 255) if y else (180, 255, 180))
        else:
            draw_text(canvas, "GT:-", (6, CELL - 10), scale=0.5, color=(150, 150, 150))
        cells.append(canvas)
    rows = [np.hstack(cells[i:i + 4]) for i in range(0, len(cells), 4)]
    wmax = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.full((CELL, wmax - r.shape[1], 3), 24, np.uint8)]) for r in rows]
    grid = np.vstack(rows)
    header = np.full((52, grid.shape[1], 3), 24, np.uint8)
    draw_text(header, f"single-tooth crops -> plaque classifier (YOLO11s-cls @224)  "
                      f"top-{len(cells)} by P(plaque)   red=P>={THR}", (12, 34), scale=0.7)
    cv2.imwrite(str(OUT / "step3_tooth_crops.jpg"), np.vstack([header, grid]))
    for k in range(len(crops)):
        rank, _ = keeps[k]
        cv2.imwrite(str(OUT / f"crops/tooth_{rank + 1:02d}_P{probs[k]:.3f}.jpg"), crops[k])

    # ---- step4: per-tooth scores ----
    sc4 = disp.copy()
    for k in range(len(crops)):
        rank, ti = keeps[k]
        p = float(probs[k])
        x0, y0, x1, y1 = [int(v * sc) for v in tb[ti]]
        col = (60, 60, 220) if p >= THR else (80, 190, 80)
        cv2.rectangle(sc4, (x0, y0), (x1, y1), col, 4)
        draw_text(sc4, f"T{rank + 1:02d} {p:.2f}", (x0 + 3, max(22, y0 - 8)),
                  color=col, scale=0.55)
    cv2.imwrite(str(OUT / "step4_plaque_scores.jpg"), sc4)

    # ---- step5: final result + image-level stats ----
    fin = disp.copy()
    n_pos = sum(p >= THR for p in probs)
    bar = np.full((90, DISPLAY_W, 3), 24, np.uint8)
    draw_text(bar, f"plaque screening  |  teeth: {len(crops)}  positive @{THR}: {n_pos}",
              (14, 34), scale=0.75,
              color=(120, 120, 255) if n_pos else (140, 230, 140))
    draw_text(bar, f"image-level  max-P={max(probs):.3f}  mean-P={float(np.mean(probs)):.3f}"
                   f"  (screening: max-P rules, confirm: per-tooth thr)",
              (14, 72), scale=0.62)
    for k in range(len(crops)):
        rank, ti = keeps[k]
        p = float(probs[k])
        x0, y0, x1, y1 = [int(v * sc) for v in tb[ti]]
        col = (60, 60, 220) if p >= THR else (80, 190, 80)
        cv2.rectangle(fin, (x0, y0), (x1, y1), col, 3)
    cv2.imwrite(str(OUT / "step5_result.jpg"), np.vstack([bar, fin]))

    # GT comparison table (only matched teeth are judgeable)
    n_gt_pos = sum(1 for k in range(len(crops))
                   if (g := gt_of_seg.get(k)) and g[1] == 1)
    tp = sum(1 for k in range(len(crops))
             if (g := gt_of_seg.get(k)) and g[1] == 1 and probs[k] >= THR)
    fn = n_gt_pos - tp
    fp = sum(1 for k in range(len(crops))
             if (g := gt_of_seg.get(k)) and g[1] == 0 and probs[k] >= THR)
    tbl = ["| 牙 | P(plaque) | 判定@%.3f | GT | 一致性 |" % THR,
           "|---|---|---|---|---|"]
    for k in np.argsort(probs)[::-1]:
        rank, _ = keeps[k]
        g = gt_of_seg.get(k)
        p = float(probs[k]); dec = "plaque" if p >= THR else "normal"
        if g is None:
            tbl.append(f"| T{rank + 1:02d} | {p:.3f} | {dec} | 未标注 | — |")
        else:
            gt_s = "plaque" if g[1] else "clean"
            ok = "✓" if (p >= THR) == bool(g[1]) else "✗ MISS"
            tbl.append(f"| T{rank + 1:02d} | {p:.3f} | {dec} | {gt_s} | {ok} |")

    readme = f"""# 单牙分割 -> 牙菌斑检测 Demo 步骤

样本：Mendeley test base `{iid}`（匹配牙最多的 test 图）
阈值：部署校准刻度 thr={THR}（raw 0.70 等价）；龈缘窗口已证伪，整牙 crop 为最终输入。

## 流程

1. **step1_input.jpg** 原始口内照（26MP 缩放展示）
2. **step2_tooth_segmentation.jpg** `iop_toothseg`（YOLO11m-seg @896, conf0.30/iou0.70）
   逐牙实例分割 -> mask 彩色叠加 + 牙序号 T01..T{len(crops)}
3. **step3_tooth_crops.jpg** 每颗牙 bbox crop（即分类器输入，224 resize），
   逐张原始 crop 在 `crops/tooth_XX_P[Y].jpg`；按 P 降序排列，红框=P>={THR}
   GT 标注 = 与评测相同的 union-box 配对（IoU>=0.3 或质心包含）；`GT:-` = 该牙无标注
4. **step4_plaque_scores.jpg** `plaque_bin_v2_sano`（YOLO11s-cls @224）逐牙打分
   + Platt 校准（a=0.4695,b=0.6801），框色/文字 = 单牙 P(plaque)
5. **step5_result.jpg** 最终结果：单牙红/绿标注 + 图级统计
   （筛查口径 max-P；确认口径逐牙 thr；全阴性图判断用 mean-P 档）

## GT 对比（仅可评判已配对牙）

- 分割 {len(crops)} 颗，其中与 GT 配对 {len(gt_of_seg)} 颗（GT 全部为 plaque）
- @{THR}：TP {tp} / FN {fn} / FP {fp}；漏检牙见下表 ✗（托槽牙为已知盲区，
  筛查档 0.39 下大部分可召回）

{chr(10).join(tbl)}

## 数值

- 模型底数：牙位分割 mAP50 0.99 / 菌斑框级 AUROC 0.896（test）
- 单牙级双档阈值：筛查档 0.39（Sens 优先，阈值扫描冻结）/ 部署默认 {THR}（Spec 优先）
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(f"demo done: {iid}  teeth={len(crops)}  pos@{THR}={n_pos}  -> {OUT}")


if __name__ == "__main__":
    main()
