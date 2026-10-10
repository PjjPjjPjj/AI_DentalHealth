"""Draw plaque model-structure and principle diagrams (matplotlib, CJK font)."""
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

ROOT = Path(f"{_ROOT}")
OUT = ROOT / "outputs/demo/pipeline_walkthrough/plaque_figs"
OUT.mkdir(parents=True, exist_ok=True)
fm.fontManager.addfont(str(ROOT / "assets/fonts/NotoSansSC.otf"))
plt.rcParams["font.family"] = "Noto Sans CJK SC"
plt.rcParams["axes.unicode_minus"] = False


def box(ax, x, y, w, h, text, fc="#EAF3FB", ec="#2E75B6", fs=11, lw=1.4, text_kw=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec, lw=lw))
    kw = dict(ha="center", va="center", fontsize=fs, color="#1a1a1a")
    if text_kw:
        kw.update(text_kw)
    ax.text(x + w / 2, y + h / 2, text, **kw)
    # note: text_kw may override ha via "x" position key usage in caller


def box_left(ax, x, y, w, h, text, fc="#F2F2F2", ec="#7F7F7F", fs=11.5, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + 0.25, y + h / 2, text, ha="left", va="center", fontsize=fs, color="#1a1a1a")


def arrow(ax, x0, y0, x1, y1, color="#555555", lw=1.6, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=14, color=color, lw=lw))


def fig_structure():
    fig, ax = plt.subplots(figsize=(15.5, 5.2), dpi=150)
    ax.set_xlim(0, 15.8)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    ax.text(7.9, 4.9, "牙菌斑分类器模型结构：YOLO11s-cls（5.44M 参数 / 12.1 GFLOPs / 输入 224×224）",
            ha="center", fontsize=15, weight="bold")

    box(ax, 0.1, 2.0, 1.3, 1.1, "输入\n牙位 crop\n224×224×3", fc="#FFF7E6", ec="#BF8F00", fs=11)
    xs = 1.9
    stages = [("Conv 3×3 ×2\n(stem)", 1.3), ("Stage 1\nC3k2", 1.0),
              ("Stage 2\nC3k2", 1.0), ("Stage 3\nC3k2", 1.0), ("Stage 4\nC3k2", 1.0)]
    y0 = 2.0
    prev_r = 0.1 + 1.3
    for i, (name, w) in enumerate(stages):
        box(ax, prev_r + 0.26, y0, w, 1.1, name)
        arrow(ax, prev_r + 0.02, y0 + 0.55, prev_r + 0.24, y0 + 0.55)
        prev_r = prev_r + 0.26 + w
    box(ax, prev_r + 0.26, y0, 1.2, 1.1, "Conv 1×1\n特征聚合", fc="#E2EFDA", ec="#548235", fs=10.5)
    arrow(ax, prev_r + 0.02, y0 + 0.55, prev_r + 0.24, y0 + 0.55)
    prev_r += 0.26 + 1.2
    box(ax, prev_r + 0.26, y0, 1.35, 1.1, "全局平均池化\nGAP", fc="#E2EFDA", ec="#548235", fs=10.5)
    arrow(ax, prev_r + 0.02, y0 + 0.55, prev_r + 0.24, y0 + 0.55)
    prev_r += 0.26 + 1.35
    box(ax, prev_r + 0.26, y0, 1.3, 1.1, "全连接\n2 类 Logits", fc="#E2EFDA", ec="#548235", fs=10.5)
    arrow(ax, prev_r + 0.02, y0 + 0.55, prev_r + 0.24, y0 + 0.55)
    prev_r += 0.26 + 1.3
    box(ax, prev_r + 0.26, 2.0, 1.75, 1.1, "Softmax\nP(normal) P(plaque)", fc="#FBE5D6", ec="#C55A11", fs=10.5)
    arrow(ax, prev_r + 0.02, y0 + 0.55, prev_r + 0.24, y0 + 0.55)

    ax.text(7.9, 1.45, "判读：P(plaque) ≥ 0.92 → 该牙判定“有菌斑”（阈值在 val 上按筛查优先冻结）",
            ha="center", fontsize=12.5, color="#C00000")
    ax.text(7.9, 0.8, "训练要点：Mendeley 133,824 个标注框（阳性 97.7%）→ 框级裁剪；负类 ×8 过采样；训练期验证集按 1:1 重平衡；保守颜色增强（HSV 小抖动）",
            ha="center", fontsize=11, color="#404040")
    fig.tight_layout()
    fig.savefig(OUT / "fig_plaque_model_structure.jpg", bbox_inches="tight")
    plt.close(fig)
    print("saved fig_plaque_model_structure.jpg")


def fig_principle():
    fig, ax = plt.subplots(figsize=(13, 6.2), dpi=150)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.text(6.5, 5.95, "牙菌斑检测原理：不做像素级分割，做“牙位级二分类”", ha="center",
            fontsize=15, weight="bold")

    # top: pipeline context
    box(ax, 0.2, 4.5, 1.9, 0.95, "全口内照\n(完整 pipeline 输入)", fc="#FFF7E6", ec="#BF8F00", fs=10.5)
    box(ax, 2.7, 4.5, 2.1, 0.95, "牙齿实例分割\n(iop_toothseg, mAP50 0.99)", fs=10)
    box(ax, 5.4, 4.5, 2.0, 0.95, "逐牙裁剪 crop\n(牙位框 = 分类窗口)", fs=10)
    box(ax, 8.0, 4.5, 2.1, 0.95, "YOLO11s-cls\nP(plaque) per tooth", fc="#E2EFDA", ec="#548235", fs=10)
    box(ax, 10.7, 4.5, 2.1, 0.95, "thr 0.92 判定\n牙位级 有/无菌斑 → JSON", fc="#FBE5D6", ec="#C55A11", fs=10)
    for a, b in [(2.1, 2.7), (4.8, 5.4), (7.4, 8.0), (10.1, 10.7)]:
        arrow(ax, a, 4.97, b, 4.97)

    # middle: why
    box_left(ax, 0.2, 2.6, 6.1, 1.35,
        "为什么是二分类而不是分割？\n"
        "① 菌斑沿龈缘弥散分布，无清晰像素边界，像素级 GT 本身不可靠\n"
        "② 数据集标注即“区域级 有/无 (0/1)”——标签语义决定任务形式\n"
        "③ 临床口径即“这颗牙有没有菌斑”，牙位级结论可直接用于宣教/评分",
        fc="#F2F2F2", ec="#7F7F7F", fs=11.5, )
    box_left(ax, 6.7, 2.6, 6.1, 1.35,
        "类别极度失衡的应对（关键工程点）\n"
        "① 数据：阳性框 97.7%，阴性框仅 ~2,700 个\n"
        "② 训练：阴性 crop ×8 过采样 → 94,018 : 19,818\n"
        "③ 验证：训练期 val 重平衡到 1:1，防早停选中“全判阳性”权重",
        fc="#F2F2F2", ec="#7F7F7F", fs=11.5, )

    # bottom: metrics
    box_left(ax, 0.2, 0.5, 6.1, 1.6,
        "评估协议（诚实口径）\n"
        "• 官方患者级 split，绝不泄漏；val/test 只用无增强干净图\n"
        "• AUROC：GT 框直推 0.868 / 接入牙齿分割的牙位 crop 0.9164\n"
        "• test 阴性仅 56 个 → Spec 置信区间宽，如实披露",
        fc="#EAF3FB", ec="#2E75B6", fs=11.5, )
    box_left(ax, 6.7, 0.5, 6.1, 1.6,
        "工作点选择（筛查优先）\n"
        "• 约束 Sens ≥ 0.95，在此前提下最大化 Spec\n"
        "• 冻结 thr = 0.92：test Sens 0.940 / Spec 0.554\n"
        "• 场景可调：宣教提醒→降阈值提召回；科研口径→报 AUROC",
        fc="#FBE5D6", ec="#C55A11", fs=11.5, )
    fig.tight_layout()
    fig.savefig(OUT / "fig_plaque_principle.jpg", bbox_inches="tight")
    plt.close(fig)
    print("saved fig_plaque_principle.jpg")


if __name__ == "__main__":
    fig_structure()
    fig_principle()
