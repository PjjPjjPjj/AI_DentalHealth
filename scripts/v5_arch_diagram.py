"""V5 overall architecture flow diagram with tensor/image sizes (Chinese).

Output: outputs/demo/v5_report/v5_architecture_diagram.jpg
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

FONT = f"{_ROOT}/assets/fonts/NotoSansSC.otf"
OUT = "outputs/demo/v5_report/v5_architecture_diagram.jpg"
W, H = 1560, 1740
BG, NAVY, BLUE, GREEN, ORANGE, GREY, RED = (250, 250, 250), (31, 58, 95), (46, 117, 182), (47, 125, 50), (191, 96, 0), (110, 110, 110), (192, 0, 0)
f36 = ImageFont.truetype(FONT, 36)
f26 = ImageFont.truetype(FONT, 26)
f22 = ImageFont.truetype(FONT, 22)
f20 = ImageFont.truetype(FONT, 20)
f18 = ImageFont.truetype(FONT, 18)

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)
d.text((30, 20), "V5 牙结石检测 —— 整体推理架构（含各阶段输入/输出尺寸）", font=f32 if False else f36, fill=(20, 20, 20))


def box(x, y, w, h, lines, fill=BLUE, title_fs=24, body_fs=19, dashed=False):
    r = 12
    d.rounded_rectangle([x, y, x + w, y + h], radius=r, fill=fill, outline=(60, 60, 60), width=1)
    n = len(lines)
    total = title_fs + (n - 1) * (body_fs + 6)
    ty = y + h / 2 - total / 2
    for i, ln in enumerate(lines):
        for sub in ln.split("\n"):
            fs = title_fs if i == 0 else body_fs
            font = ImageFont.truetype(FONT, fs)
            tw = d.textlength(sub, font=font)
            d.text((x + w / 2 - tw / 2, ty), sub, font=font,
                   fill=(255, 255, 255) if fill != (255, 255, 255) else (30, 30, 30))
            ty += fs + 4
        ty += 2


def arrow_down(x, y1, y2, label=None, color=GREY, dashed=False):
    d.line([x, y1, x, y2 - 12], fill=color, width=3)
    d.polygon([(x - 8, y2 - 14), (x + 8, y2 - 14), (x, y2)], fill=color)
    if label:
        font = ImageFont.truetype(FONT, 18)
        tw = d.textlength(label, font=font)
        d.text((x + 10, (y1 + y2) / 2 - 10), label, font=font, fill=(70, 70, 70))


def arrow_right(x1, x2, y, label=None, color=GREY):
    d.line([x1, y, x2 - 12, y], fill=color, width=3)
    d.polygon([(x2 - 14, y - 8), (x2 - 14, y + 8), (x2, y)], fill=color)
    if label:
        font = ImageFont.truetype(FONT, 18)
        d.text(((x1 + x2) / 2 - d.textlength(label, font=font) / 2, y - 30), label, font=font, fill=(70, 70, 70))


CX = W // 2 - 190
BW, BH = 560, 86

# input
box(CX, 80, BW, BH, ["输入口内照片", "612×408×3 (宽图) 或 <500px 特写"], fill=NAVY)
arrow_down(CX + BW / 2, 166, 216, "max(W,H)≥500 → 2×2 切片；特写 → 整图")
box(CX - 60, 216, BW + 120, 100, ["预处理", "宽图: 2×2 切片 320×320×3 (2×放大, 25%重叠)\n特写: resize 640×640×3"], fill=NAVY)
arrow_down(CX + BW / 2, 316, 366)
box(CX, 366, BW, BH, ["S1  YOLO11m-seg  @640", "2类: calculus / other_lesion"], fill=BLUE)
arrow_down(CX + BW / 2, 452, 502, f"框+mask原型, 置信度≥0.02 (本例 10 候选)")
box(CX, 502, BW, 78, ["S2 精化预处理", "每个S1框: 框中心裁 160×160×3 → 4×放大 640×640×3"], fill=NAVY)
arrow_down(CX + BW / 2, 580, 630)
box(CX, 630, BW, BH, ["S2  YOLO11m-seg  @640", "实例 mask 640×640 → 缩回160² → 贴回原图"], fill=BLUE)
arrow_down(CX + BW / 2, 716, 766, "每个候选: RGB裁剪 + S2 mask")
box(CX, 766, BW, BH, ["Verifier  EfficientNet-V2-S", "输入 4×640×640 (RGB+mask), 输出真伪概率"], fill=GREEN)
arrow_down(CX + BW / 2, 852, 902, "8 维特征: s1c / s2c / verifier / 面积×2 / s1s2-IoU / bbox面积 / 长宽比")
box(CX, 902, BW, BH, ["特征融合  LogisticRegression", "输出 fused ∈ [0,1]  (冻结权重 fusion_v3.npz)"], fill=GREEN)
arrow_down(CX + BW / 2, 988, 1038, "分尺寸判决: s2面积比≥1.565% → thr 0.45, 否则 0.60; 需 s1c≥0.02")
box(CX, 1038, BW, BH, ["Mask NMS (IoU 0.5)", "→ 实例输出: mask + 置信度  (救援级联见下)"], fill=ORANGE)

# rescue cascade (dashed, left side)
rx, ry = 60, 960
d.rounded_rectangle([rx, ry, rx + 400, ry + 120], radius=12, outline=RED, width=2)
d.text((rx + 14, ry + 10), "救援级联 (虚线)", font=f22, fill=RED)
d.text((rx + 14, ry + 42), "当 0 个候选过阈值 且 图像级 B≥0.40:", font=f18, fill=(60, 60, 60))
d.text((rx + 14, ry + 66), "阈值降为 0.25 重新录取候选", font=f18, fill=(60, 60, 60))
d.text((rx + 14, ry + 90), "(test 上救回 28/700 张, 中病灶召回+2.1pt)", font=f18, fill=(60, 60, 60))
for i in range(8):
    d.line([rx + 400 + i * 14, ry + 60, rx + 406 + i * 14, ry + 60], fill=RED, width=2)
arrow_right(rx + 400, CX - 10, ry + 60, None)

# image-level ensemble
ey = 1170
box(60, ey, 430, 110, ["A 检测流", "PASS 候选的最大 fused 分数"], fill=BLUE, title_fs=22)
box(560, ey, 430, 110, ["B 分类流", "EffNet-V2-S @448×448 + 水平翻转 TTA"], fill=GREEN, title_fs=22)
box(1060, ey, 430, 110, ["C 分类流", "YOLO11m-cls @448×448 (独立骨干)"], fill=GREEN, title_fs=22)
for xx in (275, 775, 1275):
    arrow_down(xx, ey + 110, ey + 170)
box(360, ey + 170, 830, 84, ["图像级 LR 集成", "prob = LR(A, B, C)  ≥ 0.38 → 判\"有牙结石\""], fill=ORANGE)
arrow_down(775, ey + 254, ey + 304)
box(560, ey + 304, 430, 104, ["输出", "图像级: 有/无牙结石\n实例级: mask + 置信度"], fill=NAVY, title_fs=22)

# side note
d.text((60, ey + 420), "注: 所有阈值/融合权重均在 val 集冻结, test 只推理一次; 跨片合并用框 NMS; S2 mask 用最近邻插值保持二值。",
       font=f18, fill=(90, 90, 90))

img.save(OUT, quality=92)
print("saved", OUT, img.size)
