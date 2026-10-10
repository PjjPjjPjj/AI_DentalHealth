"""Generate a data-interpretation three-line table (data | value | reading)
for the plaque dataset, PPT-ready (SimSun/Times, white bg), and insert it
into the PPT right after the plaque dataset slides.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

ROOT = Path(f"{_ROOT}")
OUT = ROOT / "outputs/demo/pipeline_walkthrough/plaque_figs"
OUT.mkdir(parents=True, exist_ok=True)
PPT = ROOT / "outputs/demo/pipeline_walkthrough/完整检测汇报.pptx"

ROWS = [
    ("数据来源", "Mendeley 学术数据集（2 卷 21.6GB）", "附 306 份伦理批件 + 148 份临床记录，合规链完整"),
    ("患者数", "148 人（每人 9 视角 × 8 官方增强）", "患者级隔离划分，全流程零泄漏（md5 重复 0）"),
    ("图像总量", "10,450 张 = base 1,319 + 增强变体 9,131", "增强变体仅入训练；val/test 只用 base 干净图"),
    ("分辨率", "中位 6240×4160（26MP 单反）", "0 损坏 / 0 无法解码，裁剪 crop 中位 391×338px"),
    ("标注粒度", "框级实例：0/1 + 坐标 + FDI 牙位", "非图像级/像素级；一颗牙可对应多个框（冠+龈缘条带）"),
    ("标注框总量", "133,824（阳性 131,106 / 阴性 2,718）", "每图均值 12.8 框"),
    ("类别比例", "阳性 98% : 阴性 2%", "数据集最大特征——决定训练过采样与评估口径"),
    ("官方划分", "train 7,538 / val 1,266 / test 1,646 张", "base 口径 950 / 162 / 207，直接采用不重划分"),
    ("牙位分布", "20 个 FDI；前牙 11/21/41/31 占 40%", "下前牙区（菌斑最高发部位）标注最密集"),
    ("阴性样本", "全集约 2,718 框；test 干净仅 56", "Spec 置信区间宽的根本原因（±13pt）"),
    ("全阴性牙", "0 颗", "牙位级聚合不成立 → 任务形式定为框级二分类"),
    ("转换训练集", "113,836 crops；阴性 ×8 过采样", "train 94,018:19,818；训练期 val 重平衡 1:1 防塌缩"),
    ("评测口径", "clean 2,528 / extended 2,892（test）", "阴性 56 / 420 双口径；结论在两口径下一致"),
    ("已知瑕疵", "空标签 21 张、1 张源图损坏、大小写 bug", "均已排除/兼容处理，影响 <0.2%"),
]
HEADERS = ["数据项", "数值", "解读"]
COL_W = [1.6, 4.6, 6.2]


def draw_table():
    fm.fontManager.addfont(str(ROOT / "assets/fonts/NotoSansSC.otf"))
    plt.rcParams["font.family"] = "Noto Sans CJK SC"
    n = len(ROWS) + 1
    row_h = 0.52
    fig_h = 0.9 + row_h * n
    fig, ax = plt.subplots(figsize=(14.2, fig_h), dpi=170)
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    xs = [0.25]
    for w in COL_W:
        xs.append(xs[-1] + w)
    y_top = fig_h - 0.55

    def rule(y, lw):
        ax.plot([xs[0], xs[-1]], [y, y], color="black", lw=lw, solid_capstyle="butt")

    rule(y_top + 0.06, 1.8)                      # top rule
    for j, h in enumerate(HEADERS):
        ax.text(xs[j] + 0.12, y_top - row_h / 2, h, fontsize=13.5, weight="bold",
                va="center")
    rule(y_top - row_h, 0.9)                     # header rule
    for i, row in enumerate(ROWS):
        y = y_top - row_h * (i + 2)
        for j, cell in enumerate(row):
            ax.text(xs[j] + 0.12, y + row_h / 2, cell, fontsize=11.5, va="center",
                    color="#1a1a1a")
        if i < len(ROWS) - 1:
            ax.plot([xs[0], xs[-1]], [y, y], color="#D9D9D9", lw=0.5)
    rule(y_top - row_h * n, 1.8)                 # bottom rule
    fig.savefig(OUT / "fig_plaque_dataset_table.jpg", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("saved:", OUT / "fig_plaque_dataset_table.jpg")


def insert_ppt():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from update_ppt_v2 import add_picture_slide, walk_runs, white_bg  # noqa: E402
    from pptx import Presentation

    prs = Presentation(str(PPT))
    slide = add_picture_slide(
        prs, "牙菌斑 · 数据集解读（数据 → 含义）",
        OUT / "fig_plaque_dataset_table.jpg",
        caption="全部数字来自真实审计与转换日志；数据文件：outputs/plaque_dataset/")
    # move new slide to right after slide 21 (plaque section head, index 20)
    xml = prs.slides._sldIdLst
    elems = list(xml)
    new = elems[-1]
    xml.remove(new)
    xml.insert(21, new)
    for s in prs.slides:
        white_bg(s)
        walk_runs(s.shapes, black_all=False)
    prs.save(str(PPT))
    print("PPT saved, slides:", len(prs.slides))


if __name__ == "__main__":
    draw_table()
    insert_ppt()
