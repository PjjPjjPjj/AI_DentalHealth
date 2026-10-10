"""V5 inference pipeline walkthrough figure (real intermediates, Chinese labels).

Image: (1270).jpg (612x408, 2 GT instances) - S1 gives 7 candidates, fusion
passes exactly 1: ideal for explaining selectivity.

Output: outputs/demo/v5_report/v5_pipeline_walkthrough.jpg
"""
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
from PIL import Image, ImageDraw, ImageFont
from sklearn.linear_model import LogisticRegression

import scripts.v3_pipeline_lib as _lib
_lib.OUT = Path("outputs/calculus_v5_cls")
_lib.C1 = 0.005
from scripts.v3_pipeline_lib import (DATA, SIDE, TARGET, TILE, WIDE_MIN,
                                     collect_split, greedy_nms, make_verifier,
                                     rec_bin, s1_pass, s2_instances)
from ultralytics import YOLO

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

FONT = f"{_ROOT}/assets/fonts/NotoSansSC.otf"
OUT_DIR = Path("outputs/demo/v5_report")
IMG_ID = "(1270)"
YEL, RED, GREEN, BLUE, WHITE, BLACK = (0, 255, 255), (0, 0, 255), (0, 200, 0), (255, 80, 0), (255, 255, 255), (30, 30, 30)

man = pd.read_csv(DATA / "split_manifest_v3.csv")
row = man[(man.image_id.astype(str) == IMG_ID) & (man.split == "test")].iloc[0]
img = cv2.imread(row.image_path)
H, W = img.shape[:2]

s1m = YOLO("outputs/calculus_v5_stage1/weights/best.pt")
s2m = YOLO("outputs/calculus_v3b_stage2/weights/best.pt")
vb = make_verifier()

mc = man[man.image_id.astype(str) == IMG_ID].copy()
im = collect_split(s1m, s2m, mc, "test", with_verifier=True, vb=vb)[0]
f = im["feats"]
fz = np.load(_lib.OUT / "fusion_v3.npz")
lr = LogisticRegression()
lr.coef_, lr.intercept_, lr.classes_ = fz["coef"], fz["intercept"], np.array([0, 1])
X = np.array([[x[c] for c in _lib.FCOLS] for x in f], float)
fused = lr.predict_proba((X - fz["mu"]) / fz["sg"])[:, 1]
s1c = np.array([x["s1_conf"] for x in f])
ar = np.array([x["s2_area_ratio"] for x in f])
thr = np.where(ar >= 0.01565, 0.45, 0.6)
ok = (s1c >= 0.02) & (fused >= thr)

# panel 0: input
p0 = img.copy()
# panel 1: tiles + S1 boxes
boxes, confs, masks = s1_pass(s1m, row.image_path)
p1 = img.copy()
tw = TILE
if max(W, H) >= WIDE_MIN:
    offs = [(0, 0), (W - tw, 0), (0, H - tw), (W - tw, H - tw)]
    for ox, oy in offs:
        cv2.rectangle(p1, (ox, oy), (ox + tw, oy + tw), (200, 200, 200), 1)
for b, c in zip(boxes, confs):
    cv2.rectangle(p1, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), YEL, 2)
    cv2.putText(p1, f"S1 {c:.2f}", (int(b[0]), max(14, int(b[1]) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, YEL, 1)
# panel 2: S2 windows + one crop zoom
p2 = img.copy()
top = int(np.argmax(fused))
rect = im["rects"][top]
for i in range(len(f)):
    rc = im["rects"][i]
    cv2.rectangle(p2, (rc["x0"], rc["y0"]), (rc["x0"] + rc["cw"], rc["y0"] + rc["ch"]),
                  GREEN if ok[i] else (150, 150, 150), 1)
bx = None
rc_top = im["rects"][top]
x0, y0, cw_, ch_ = rc_top["x0"], rc_top["y0"], rc_top["cw"], rc_top["ch"]
crop = img[y0:y0 + ch_, x0:x0 + cw_]
crop640 = cv2.resize(crop, (TARGET, TARGET), interpolation=cv2.INTER_AREA)
insts = s2_instances(s2m, crop640)
crop_vis = crop640.copy()
bi = None
if insts:
    bi = max(insts, key=lambda d: d["s2_conf"])
    crop_vis[bi["bin640"] > 0] = (0.5 * np.array([0, 200, 0]) + 0.5 * crop_vis[bi["bin640"] > 0]).astype(np.uint8)
    cnts, _ = cv2.findContours(bi["bin640"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(crop_vis, cnts, -1, (0, 255, 0), 2)
    cv2.putText(crop_vis, f"S2 conf {bi['s2_conf']:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.putText(p2, "160x160", (x0, max(14, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)

# panel 3: fusion table (text, PIL)
# panel 4: final instances
preds, confs_f = [], []
if ok.any():
    recs = []
    for j in np.argwhere(ok).ravel():
        rc = im["rects"][j]
        recs.append({"x0": rc["x0"], "y0": rc["y0"], "cw": rc["cw"], "ch": rc["ch"],
                     "bin": im["bins"][j], "score": f[j]["s2_conf"]})
    for k in greedy_nms(recs, "mask_nms", 0.5, H, W):
        preds.append(rec_bin(k, H, W))
        confs_f.append(k["score"])
p4 = img.copy()
for p, c in zip(preds, confs_f):
    m = p > 0
    p4[m] = (0.5 * np.array([0, 200, 0]) + 0.5 * p4[m]).astype(np.uint8)
    cnts, _ = cv2.findContours(p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(p4, cnts, -1, (0, 255, 0), 2)
    ys, xs = np.nonzero(p)
    cv2.putText(p4, f"{c:.2f}", (int(xs.min()), max(14, int(ys.min()) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
# GT contour for reference
gt_bin = np.zeros((H, W), np.uint8)
for g in im["gts"]:
    gt_bin |= g["mask"]
for p in (p0, p1, p4):
    cnts, _ = cv2.findContours(gt_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(p, cnts, -1, RED, 1)

PW, PH = 560, int(H * 560 / W)
gap = 16
crop_small = cv2.resize(crop_vis, (PH, PH))

def to_pil(a):
    return Image.fromarray(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))

canvas = Image.new("RGB", (PW * 3 + gap * 2, PH * 2 + 60 + 240), (250, 250, 250))
d = ImageDraw.Draw(canvas)
f22 = ImageFont.truetype(FONT, 22)
f16 = ImageFont.truetype(FONT, 16)
f14 = ImageFont.truetype(FONT, 14)

panels = [(p0, "0 输入图像 612x408 (红=GT)"),
          (p1, f"1 S1 分割: 2x2切片(320px,2x放大) S1候选{len(boxes)} -> S2实例{len(f)}"),
          (p2, "2 S2 精化: 框中心160px窗口 4x放大至640")]
for i, (pa, ttl) in enumerate(panels):
    xx = i * (PW + gap)
    canvas.paste(to_pil(cv2.resize(pa, (PW, PH))), (xx, 44))
    d.text((xx, 12), ttl, font=f16, fill=(30, 30, 30))
canvas.paste(to_pil(crop_small), (0, PH + 60))
conf_txt = f"S2 conf {bi['s2_conf']:.2f}" if bi is not None else "no instance"
d.text((0, PH + 40), f"2b S2 输出: 640x640 mask ({conf_txt})", font=f16, fill=(30, 30, 30))

# fusion table
tx = PW + gap
d.text((tx, PH + 40), "3 Verifier + 融合LR: 每候选8特征 -> fused -> 阈值判决", font=f16, fill=(30, 30, 30))
ty = PH + 70
d.text((tx, ty), "#  s1c   s2c   verif  fused  thr  判定", font=f14, fill=(80, 80, 80))
for i in range(len(f)):
    yy = ty + 26 + i * 24
    mark = "PASS" if ok[i] else "rej"
    col = (0, 130, 0) if ok[i] else (160, 160, 160)
    d.text((tx, yy), f"{i:2d}  {s1c[i]:.2f}  {f[i]['s2_conf']:.2f}  "
                     f"{f[i]['verifier_prob']:.2f}  {fused[i]:.2f}  {thr[i]:.2f}",
           font=f14, fill=(60, 60, 60))
    d.text((tx + 320, yy), mark, font=f14, fill=col)
d.text((tx, ty + 26 + len(f) * 24 + 6),
       f"通过 {int(ok.sum())}/{len(f)} -> Mask NMS(0.5) -> 最终实例 {len(preds)} 个",
       font=f16, fill=(0, 130, 0))
canvas.paste(to_pil(cv2.resize(p4, (PW, PH))), (2 * (PW + gap), PH + 60))
d.text((2 * (PW + gap), PH + 40), "4 输出: 实例 mask + 置信度 (红=GT)", font=f16, fill=(30, 30, 30))

# image level strip
d.text((0, PH * 2 + 64), "图像级三流集成:", font=f22, fill=(30, 30, 30))
info = (f"A 检测流 max(fused of PASS) = {fused[ok].max() if ok.any() else 0:.2f}   "
        f"B EffNet-TTA = 见集成   C YOLO-cls = 见集成   ->  LR >= 0.38 判定")
d.text((0, PH * 2 + 96), "A: 检测流 | B: EffNet-V2-S @448 + 翻转TTA | C: YOLO11m-cls @448 -> LR 集成 -> p >= 0.38 判\"有牙结石\"",
       font=f16, fill=(60, 60, 60))
d.text((0, PH * 2 + 124), f"本图: {len(f)} 候选 -> {int(ok.sum())} 通过 -> 实例输出 {len(preds)} 个; 图像级 A 分数 = "
                          f"{float(fused[ok].max()) if ok.any() else 0:.2f}", font=f16, fill=(0, 130, 0))

out = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
cv2.imwrite(str(OUT_DIR / "v5_pipeline_walkthrough.jpg"), out)
print("table rows:", [(round(s1c[i],2), round(f[i]["s2_conf"],2), round(fused[i],2), round(thr[i],2), bool(ok[i])) for i in range(len(f))])
print("final confs:", [round(c,3) for c in confs_f])
print("saved", OUT_DIR / "v5_pipeline_walkthrough.jpg", canvas.size)
