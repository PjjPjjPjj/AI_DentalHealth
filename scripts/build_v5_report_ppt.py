"""Build V5 calculus-detection report PPT (<=30 slides, Chinese).

Source material:
  outputs/demo/v5_report/*.{jpg,json}   demo panels + montage
  outputs/calculus_v5_cls/final_v5_metrics.json   frozen V5 test metrics
  outputs/reports/v3_final_report.md    evolution story + design rationale

Output: outputs/demo/v5_report/V5牙结石检测汇报.pptx
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

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

BASE = Path("outputs/demo/v5_report")
OUT = BASE / "V5牙结石检测汇报.pptx"

NAVY = RGBColor(0x1F, 0x3A, 0x5F)
BLUE = RGBColor(0x2E, 0x75, 0xB6)
LIGHT = RGBColor(0xDE, 0xEA, 0xF6)
GREY = RGBColor(0x59, 0x59, 0x59)
GREEN = RGBColor(0x2F, 0x7D, 0x32)
RED = RGBColor(0xC0, 0x00, 0x00)
ORANGE = RGBColor(0xBF, 0x60, 0x00)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

SW, SH = Inches(13.333), Inches(7.5)
prs = Presentation()
prs.slide_width = SW
prs.slide_height = SH
BLANK = prs.slide_layouts[6]

M = json.load(open("outputs/calculus_v5_cls/final_v5_metrics.json"))
IC = M["instance_cascade"]
IN = M["instance_no_cascade"]
IM = M["image"]


def blank():
    return prs.slides.add_slide(BLANK)


def bar(slide, text, sub=None):
    box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, Inches(0.9))
    box.fill.solid()
    box.fill.fore_color.rgb = NAVY
    box.line.fill.background()
    tf = box.text_frame
    tf.margin_left = Inches(0.4)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = text
    r.font.size = Pt(26)
    r.font.bold = True
    r.font.color.rgb = WHITE
    if sub:
        p2 = tf.add_paragraph()
        r2 = p2.add_run()
        r2.text = sub
        r2.font.size = Pt(12)
        r2.font.color.rgb = LIGHT


def bullets(slide, items, x=0.55, y=1.15, w=12.2, h=5.9, size=16, leading=1.15):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    first = True
    for it in items:
        if isinstance(it, tuple) and len(it) == 5:
            text, lvl, color, bold, sz = it
        elif isinstance(it, tuple) and len(it) == 2:
            text, lvl = it
            color, bold, sz = None, False, size
        else:
            text, lvl, color, bold, sz = it, 0, None, False, size
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = lvl
        p.line_spacing = leading
        r = p.add_run()
        r.text = ("• " if lvl == 0 else "– ") + text if lvl <= 1 else text
        r.font.size = Pt(sz)
        r.font.bold = bold
        if color:
            r.font.color.rgb = color
    return tb


def table(slide, data, x=0.55, y=1.3, w=12.2, row_h=0.42, header_fill=BLUE,
          fs=14, hfs=14, col_widths=None, align_first_left=True):
    rows, cols = len(data), len(data[0])
    shp = slide.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w),
                                 Inches(row_h * rows))
    tbl = shp.table
    if col_widths:
        total = sum(col_widths)
        for i, cw in enumerate(col_widths):
            tbl.columns[i].width = Inches(w * cw / total)
    for ri, row in enumerate(data):
        for ci, val in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.margin_left = Inches(0.08)
            cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = header_fill if ri == 0 else (
                LIGHT if ri % 2 == 0 else WHITE)
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if (ci == 0 and align_first_left) else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = str(val)
            r.font.size = Pt(hfs if ri == 0 else fs)
            r.font.bold = ri == 0
            r.font.color.rgb = WHITE if ri == 0 else RGBColor(0x26, 0x26, 0x26)
    return tbl


def caption(slide, text, y=6.95, color=GREY, size=12, x=0.55, w=12.2):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(0.45))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.color.rgb = color
    r.font.italic = True


def pic(slide, path, x, y, w=None, h=None):
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y),
                                    Inches(w) if w else None,
                                    Inches(h) if h else None)


def flow_boxes(slide, steps, y=1.7, box_w=2.15, box_h=0.95, gap=0.28,
               x0=0.55, fs=12, fills=None):
    x = x0
    for i, s in enumerate(steps):
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x),
                                     Inches(y), Inches(box_w), Inches(box_h))
        fill = (fills or {}).get(i, BLUE if i % 2 == 0 else NAVY)
        box.fill.solid()
        box.fill.fore_color.rgb = fill
        box.line.fill.background()
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.05)
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        lines = s.split("\n")
        for li, ln in enumerate(lines):
            p = tf.paragraphs[0] if li == 0 else tf.add_paragraph()
            p.alignment = PP_ALIGN.CENTER
            r = p.add_run()
            r.text = ln
            r.font.size = Pt(fs)
            r.font.bold = li == 0
            r.font.color.rgb = WHITE
        if i < len(steps) - 1:
            ar = slide.shapes.add_shape(
                MSO_SHAPE.RIGHT_ARROW, Inches(x + box_w + 0.02), Inches(y + box_h / 2 - 0.11),
                Inches(gap - 0.06), Inches(0.22))
            ar.fill.solid()
            ar.fill.fore_color.rgb = GREY
            ar.line.fill.background()
        x += box_w + gap


# ---------------- P1 封面 ----------------
s = blank()
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, SH)
bg.fill.solid()
bg.fill.fore_color.rgb = NAVY
bg.line.fill.background()
tb = s.shapes.add_textbox(Inches(1), Inches(2.3), Inches(11.3), Inches(2.6))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.alignment = PP_ALIGN.CENTER
r = p.add_run()
r.text = "口腔病灶智能检测系统"
r.font.size = Pt(44)
r.font.bold = True
r.font.color.rgb = WHITE
p2 = tf.add_paragraph()
p2.alignment = PP_ALIGN.CENTER
r = p2.add_run()
r.text = "V5 牙结石两阶段级联分割 —— 实验过程 · 指标 · Demo"
r.font.size = Pt(24)
r.font.color.rgb = LIGHT
p3 = tf.add_paragraph()
p3.alignment = PP_ALIGN.CENTER
r = p3.add_run()
r.text = "测试集 700 张（冻结阈值 · 一次性评测） | 2026-08"
r.font.size = Pt(14)
r.font.color.rgb = RGBColor(0x9D, 0xC3, 0xE6)

# ---------------- P2 目录 ----------------
s = blank()
bar(s, "汇报结构")
bullets(s, [
    "任务定义与挑战",
    "数据集：来源 · 清洗治理 · 实例分档（附示例图）",
    "实验迭代路线：从 v1 单模型到 v5 级联（思路与证据）",
    "V5 方法：两阶段分割 + 验证融合 + 图像级集成",
    "评测指标定义（公式 + 含义）",
    "实验结果（实例级 / 图像级）",
    "Demo：小 / 中 / 大 病灶 + 失败案例",
    "龋齿检测：数据集 · 结果 · Demo",
    "结论与下一步",
], size=20, y=1.4)

# ---------------- P3 任务定义 ----------------
s = blank()
bar(s, "任务定义与挑战", "输入一张口内照片 → 输出牙结石病灶的像素级位置与置信度")
flow_boxes(s, ["口内照片", "病灶检测分割", "病灶位置 + 置信度\n(像素级 mask)", "图像级结论\n有无牙结石"], y=1.5)
bullets(s, [
    ("三大挑战", 0, BLUE, True, 18),
    ("小目标：病灶普遍只占全图 0.2%~2%，宽视野图中仅数十像素", 1),
    ("边界模糊：牙结石与釉质/菌斑渐进过渡，标注边界本身有主观性", 1),
    ("域异构：全视野照 / 龋齿特写 / 牙龈炎特写 三种来源图像风格差异大", 1),
    ("输出要求", 0, BLUE, True, 18),
    ("实例级：mask 轮廓 + 置信度（可定位到牙面区域）", 1),
    ("图像级：该图是否存在牙结石（筛查结论）", 1),
], y=2.9)

# ---------------- P4 数据集总览 ----------------
s = blank()
bar(s, "数据集总览", "Oral-Mamba 多类分割标注 + 龋齿域图像，value 3 = 牙结石")
table(s, [
    ["项目", "内容"],
    ["标注格式", "单通道 PNG 多类 mask：0 背景 / 1 牙龈炎 / 2 菌斑 / 3 牙结石 / 4 龋齿"],
    ["图像规模", "原始 3375 张 → 清洗后 2702 张唯一图像"],
    ["数据划分", "train 1957 / val 708 / test 700（组级分层，跨组泄漏=0）"],
    ["测试集构成", "含牙结石图 148 张；GT 实例 281 个（小 158 / 中 95 / 大 28）"],
    ["图像来源域", "全视野 plain(613) / 龋齿特写 caries(1540) / 牙龈炎特写 gingivitis(1212)"],
    ["分辨率", "172 种；主体 612×408（39%），大量 200~400px 特写小图"],
], row_h=0.62, fs=14, col_widths=[2.2, 10])
caption(s, "划分规则：md5+dHash 并查集分组成 2476 组 → 组级分层划分，杜绝同图/近重复跨 split", y=6.5)

# ---------------- P5 数据集示例图 ----------------
s = blank()
bar(s, "数据集示例（红色 = 人工标注 GT）")
pic(s, BASE / "dataset_montage.jpg", 1.35, 1.15, w=10.6)
caption(s, "上：全视野域，病灶沿龈缘呈带状；下：特写域，病灶面积占比更大 —— 两域尺度差异是核心难点之一")

# ---------------- P6 数据治理 ----------------
s = blank()
bar(s, "数据治理（实验前最重要的一步）", "脏数据会让所有指标失真 —— 先审计、后训练")
table(s, [
    ["问题", "发现", "处理"],
    ["跨 split 泄漏", "346 组逐字节重复 + ~23% 近重复同时出现在 train/test", "md5+dHash 并查集分组\n组级分层划分，泄漏=0"],
    ["重复副本标签冲突", "同一照片多副本掩码互不一致（662/663 对）", "每内容组仅保留 1 份最优标注"],
    ["无标注图", "10 张图缺对应 mask", "确认为重复内容，随组去除"],
    ["清洗结果", "3375 → 2702 张唯一图像；calculus 正例 686 组", "raw 只读，清洗写入 processed/"],
], row_h=0.78, fs=13.5, col_widths=[2.6, 5.2, 4.4])
caption(s, "v1 旧口径在泄漏划分下 Acc=0.870；治理后同能力模型 0.884 且指标可信 —— 数据治理优先于调模型", y=6.35, color=RED)

# ---------------- P7 实例分档 ----------------
s = blank()
bar(s, "病灶分档规则（按单个实例面积占全图比 r）", "阈值取自 val 集三分位，v3-v5 全程冻结")
table(s, [
    ["档位", "规则", "test 实例数", "占比", "典型形态"],
    ["小 small", "r < 0.552%", "158", "56%", "点状/短线状沉积"],
    ["中 medium", "0.552% ≤ r < 1.565%", "95", "34%", "龈缘局部带状"],
    ["大 large", "r ≥ 1.565%", "28", "10%", "大片龈上结石"],
], row_h=0.6, fs=15, col_widths=[1.8, 3.4, 2.2, 1.6, 3.2])
bullets(s, [
    ("为什么分档：全图一个阈值会偏科 —— 大病灶显眼易检出，小病灶易被大病灶阈值挡掉", 0, None, False, 15),
    ("分档召回是本次实验的核心观测指标（防“平均数掩盖短板”）", 0, None, False, 15),
], y=4.4)

# ---------------- P8 迭代路线 ----------------
s = blank()
bar(s, "实验迭代路线：每一版解决一个明确问题")
table(s, [
    ["版本", "做法", "解决问题", "图像级 Acc", "实例 F1", "FP/图"],
    ["v1", "YOLO11s-seg 全图单模型", "跑通基线", "0.870*", "0.414", "0.412"],
    ["v3", "泄漏修复 + 两阶段(S1切片/S2精化)\n+ Verifier + 特征融合 + 图像级集成", "数据可信 & 小目标", "0.884", "0.445", "0.157"],
    ["v5", "zoom 数据 S1 + 分尺寸阈值\n+ 救援级联 + 三流集成重拟合", "漏检（尤其中病灶）", "0.881", "0.456", "0.243"],
], row_h=1.05, fs=13, col_widths=[0.9, 4.6, 2.4, 1.5, 1.3, 1.2])
bullets(s, [
    ("*v1 为泄漏口径，仅作历史对照，不可与治理后数字直接比较", 0, GREY, False, 12),
    ("v3→v5 取向变化：用少量 Precision 换 Recall —— F1 0.445→0.456，中病灶召回 0.453→0.558", 0, None, False, 15),
    ("v5 额外收益：图像级 PR-AUC 0.742→0.732 基本持平，AUROC 0.915 持平 —— 召回提升几乎无代价", 0, None, False, 15),
], y=5.0)

# ---------------- P9 整体架构图 ----------------
s = blank()
bar(s, "V5 整体推理架构（含各阶段输入/输出尺寸）")
pic(s, BASE / "v5_architecture_diagram.jpg", 3.85, 1.02, h=6.35)
caption(s, "要点: S1 找全 / S2 抠准 / Verifier+LR 融合判决 / 分尺寸阈值 / 三流图像级集成; 所有阈值在 val 冻结", y=7.12)

# ---------------- P9b 推理过程示例 ----------------
s = blank()
bar(s, "V5 推理过程逐步示例（真实中间产物）", "测试图 (1270): S1 候选 10 → S2 实例 7 → 融合判决通过 1 → 输出 1 实例")
pic(s, BASE / "v5_pipeline_walkthrough.jpg", 0.75, 1.1, w=11.8)
caption(s, "上排: 输入 → S1 切片检测(黄框+置信度) → S2 精化窗口; 下排: S2 640×640 mask → 融合判决表(PASS/rej) → 最终输出(红=GT)", y=7.12)

# ---------------- P10 思路1 ----------------
s = blank()
bar(s, "设计思路 ①：为什么必须两阶段放大", "小病灶是分辨率问题，不是模型容量问题")
bullets(s, [
    ("证据（v3 错误分析）", 0, BLUE, True, 18),
    ("残余 FN 的病灶面积占比中位仅 0.18%（全部阳性中位 0.94%，差 5 倍）", 1),
    ("0.18% 的病灶在 640 输入下只有 ~7px；2× 切片放大后 ~15px，仍接近输入极限", 1),
    ("26/42 漏检图：检测器与图像分类器同时无响应 —— 信息在输入端就丢了", 1),
    ("对策", 0, BLUE, True, 18),
    ("宽图(≥500px) 2×2 切片(320px 窗)推 S1，跨窗 NMS 合并；特写图整图推理", 1),
    ("S2 用 bbox 中心 160px 窗 4× 放大做精分割 —— mask 边界质量主要由 S2 决定", 1),
    ("效果", 0, BLUE, True, 18),
    ("TP Dice 0.82：检出的实例形状高度贴合 GT；FP/图 0.412→0.157", 1),
], y=1.2)

# ---------------- P11 思路2 ----------------
s = blank()
bar(s, "设计思路 ②：验证器融合 + 分尺寸阈值", "单一置信度无法同时服务大小病灶")
table(s, [
    ["组件", "做法", "为什么"],
    ["Verifier", "EffNet-V2-S，输入 RGB+S2mask 4 通道，输出实例真伪分", "mask 通道让分类器“看着分割结果”判真伪"],
    ["融合 LR", "8 特征: s1/s2 置信度、verifier 分、两阶段面积、\n两阶段 IoU、bbox 面积/长宽比 → fused 分数", "不同证据互补；LR 防过拟合（样本少）"],
    ["分尺寸阈值", "面积比 ≥1.565%（大）→ 0.45；其余 → 0.60", "大病灶证据强可用低阈值保召回；\n小病灶提阈值压 FP"],
], row_h=1.0, fs=13.5, col_widths=[2.0, 5.6, 4.6])
bullets(s, [
    ("frozen: thr_std=0.60, thr_large=0.45 —— 均为 val 集 F1/召回网格搜索后冻结", 0, None, False, 14),
], y=5.0)

# ---------------- P12 思路3 ----------------
s = blank()
bar(s, "设计思路 ③：图像级三流集成 + 救援级联", "检测盲区用分类流补 —— 两类模型失败模式不同")
table(s, [
    ["流", "来源", "作用"],
    ["A 检测流", "管线上通过阈值的最大 fused 分数", "有检出时最可信"],
    ["B 分类流", "EffNet-V2-S @448 + 水平翻转 TTA", "特写图整体判断，不看定位"],
    ["C 分类流", "YOLO11m-cls 图像分类", "不同骨干 → 误差不相关，集成才有增益"],
], row_h=0.62, fs=14, col_widths=[1.8, 5.6, 4.8])
bullets(s, [
    ("救援级联：三流都无正常检出 且 B≥0.4 → 放宽阈值至 0.25 重新录取实例", 0, None, False, 15),
    ("test 上 28 张图被救援通道找回 —— 直接对应中病灶召回 0.537→0.558", 0, None, False, 15),
    ("图像级概率 = LR(A,B,C) ≥ 0.38 判阳性（0.38 为 val 最大准确率阈值）", 0, None, False, 15),
], y=4.2)

# ---------------- P13 否决的方案 ----------------
s = blank()
bar(s, "被实验证据否决的方案（同样重要）", "每条都有 val/test 数据支撑，避免“感觉有用”式优化")
table(s, [
    ["方案", "预期", "实际结果", "结论"],
    ["纯推理端切片", "小目标更清晰", "召回不升反降", "训练/推理分布不一致"],
    ["P2 高分辨率检测头", "小目标召回↑", "FP 暴涨", "放弃"],
    ["宽图 960 输入", "细节更多", "图像级 AUROC 下降", "放弃"],
    ("ConvNeXt+域路由集成", "多模型互补", "val 0.907 → test 0.853（过拟合 val）", "回滚到 ABC 集成"),
], row_h=0.75, fs=13.5, col_widths=[3.0, 2.6, 4.2, 2.4])
caption(s, "方法论：每个新组件必须在 val 有明确增益且 test 不回退，否则回滚并记录原因", y=5.6, color=RED)

# ---------------- P14 指标定义1 ----------------
s = blank()
bar(s, "指标定义 ①：实例级", "匹配规则：预测与 GT 实例 IoU ≥ 0.5 记 TP（贪心一对一）")
table(s, [
    ["指标", "计算公式", "含义"],
    ["Precision 查准率", "TP / (TP + FP)", "报出来的病灶里有多少是真的（低=误报多）"],
    ["Recall 查全率", "TP / (TP + FN)", "真实病灶里有多少被报出（低=漏检多；FN 的镜像）"],
    ["F1", "2·P·R / (P + R)", "P 与 R 的调和平均，综合分"],
    ["IoU", "∩ / ∪", "单个实例预测 mask 与 GT 的重叠度，≥0.5 才算检出"],
    ["TP Dice", "2·∩ / (|A|+|B|)，仅对 TP 统计", "检到的实例“抠得准不准”（形状质量）"],
    ["FP/图", "FP 总数 / 图片数", "平均每张图的假阳性个数（医生负担）"],
], row_h=0.72, fs=13, col_widths=[2.2, 3.4, 6.6])
caption(s, "分档召回 Recall-small/medium/large：把 GT 按面积分档后分别算 Recall —— 防“平均分掩盖短板”", y=6.15)

# ---------------- P15 指标定义2 ----------------
s = blank()
bar(s, "指标定义 ②：图像级（筛查口径）", "整图二分类：该图“有 / 无”牙结石")
table(s, [
    ["指标", "计算公式", "含义"],
    ["Accuracy", "(TP+TN) / 总数", "整体判对比例（类别均衡时才直观）"],
    ["Sensitivity 灵敏度", "TP / (TP+FN)", "同 Recall：有病图里多少被报出（漏诊率=1−它）"],
    ["Specificity 特异度", "TN / (TN+FP)", "无病图里多少被正确放行（误报率=1−它）"],
    ["Precision", "TP / (TP+FP)", "报阳性的可信度"],
    ["F1", "2·P·R/(P+R)", "P 与 R 调和平均"],
    ["ROC-AUC", "ROC 曲线下面积", "阈值无关的排序能力；0.5=随机，1=完美"],
    ["PR-AUC", "PR 曲线下面积", "正样本少时比 ROC-AUC 更严格、更反映实用价值"],
], row_h=0.62, fs=13, col_widths=[2.4, 3.2, 6.6])

# ---------------- P16 实例级结果 ----------------
s = blank()
bar(s, "实验结果 ①：实例级（test=700）")
table(s, [
    ["指标", "无级联", "级联（V5 最终）", "解读"],
    ["Precision", "0.456", "0.439", "报 100 处约 44 处为真"],
    ["Recall", "0.466", "0.473", "仍有约一半病灶漏检（FN 主导）"],
    ["F1", "0.461", "0.456", "级联召回+0.7pt、精度−1.7pt"],
    ["Recall-小", "0.430", "0.430", "小病灶最难（像素少）"],
    ["Recall-中", "0.537", "0.558", "级联收益集中在中病灶"],
    ["Recall-大", "0.429", "0.429", "大病灶绝对数少(28)方差大"],
    ["TP Dice", "0.820", "0.818", "检出实例形状质量高"],
    ["FP/图", "0.223", "0.243", "每 4~5 张图多 1 个误报"],
], row_h=0.5, fs=13, col_widths=[2.0, 1.7, 2.3, 6.2])
caption(s, "结论：模型“检到就检得准”（Dice 0.82），瓶颈在“有没有检到”（Recall 0.47）", y=6.6, color=RED, size=13)

# ---------------- P17 图像级结果 ----------------
s = blank()
bar(s, "实验结果 ②：图像级（test=700，阈值 0.38）")
table(s, [
    ["指标", "V5 数值", "解读"],
    ["Accuracy", "0.881", "整体 88% 判对"],
    ["Sensitivity", "0.662", "有结石图 66% 被报出（筛查漏诊率 34%）"],
    ["Specificity", "0.940", "无结石图 94% 正确放行"],
    ["Precision", "0.748", "报阳性的图 3/4 真有结石"],
    ["F1", "0.703", "P-R 平衡点"],
    ["ROC-AUC", "0.915", "排序能力强，阈值可按场景调"],
    ["PR-AUC", "0.732", "阳性率 21% 下的实用价值指标"],
], row_h=0.52, fs=13, col_widths=[2.4, 2.0, 7.8])
bullets(s, [
    ("使用建议：筛查场景可降阈值(提 Sensitivity)；定量测量场景需先解决 Recall", 0, ORANGE, True, 15),
], y=6.0)

# ---------------- 龋齿数据集 ----------------
s = blank()
bar(s, "龋齿检测 · 数据集", "6313 Intraoral Caries Dataset —— 患者级划分，bbox 标注")
pic(s, Path("outputs/demo/caries_report/dataset_montage.jpg"), 0.7, 1.1, w=7.3)
table(s, [
    ["项目", "内容"],
    ["图像", "6266 张（train 5011 / val 627 / test 628）"],
    ["标注", "bbox 单类 caries；train 5386 框 / val 652 框 / test 690 框"],
    ["划分", "患者级不重叠：train 889 人 / val 360 人 / test 346 人"],
    ["视角", "正面 / 上颌合面 / 下颌合面 / 左侧 / 右侧 5 视角"],
    ["格式", "YOLO detect（images + labels + data.yaml）"],
], x=8.2, y=1.2, w=4.7, row_h=0.62, fs=11.5, hfs=12, col_widths=[1.3, 3.4])
caption(s, "红框 = 人工标注龋损 bbox", y=6.9, x=0.7)

# ---------------- 龋齿结果 ----------------
s = blank()
bar(s, "龋齿检测 · 实验结果", "test=628 张 / 690 框，阈值在 val 冻结后跑一次 test")
table(s, [
    ["模型", "输入", "conf", "Precision", "Recall", "F1", "mAP50", "mAP50-95"],
    ["基线 YOLO11s", "640", "0.45", "0.851", "0.902", "0.876", "0.928", "0.719"],
    ["C1 YOLO11m（最终）", "960", "0.45", "0.889", "0.874", "0.881", "0.934", "0.714"],
], row_h=0.55, fs=13.5, col_widths=[2.6, 1.0, 1.0, 1.5, 1.3, 1.2, 1.2, 1.6])
table(s, [
    ["补充指标", "定义", "C1 数值"],
    ["mAP50", "IoU≥0.5 下的 AP（PR 曲线面积），单类即 AP50", "0.934"],
    ["mAP50-95", "IoU 从 0.5 到 0.95（步长 0.05）的 AP 平均，更严格", "0.714"],
], x=0.55, y=3.35, w=12.2, row_h=0.5, fs=13, col_widths=[1.8, 8.0, 2.4])
bullets(s, [
    ("C1 用精度换召回：P +3.8pt / R −2.8pt，F1 0.876→0.881、mAP50 +0.6pt —— 容量+分辨率有效", 0, None, False, 15),
    ("对比牙结石（实例 F1 0.456）：龋齿 F1 0.881 —— bbox 任务粒度更粗、数据量 10×、原图分辨率更高", 0, None, False, 15),
    ("mAP50-95 仅 0.714：框的贴合度（定位精度）仍有提升空间", 0, ORANGE, False, 15),
], y=5.3)

# ---------------- 龋齿 Demo ----------------
s = blank()
bar(s, "龋齿检测 · Demo", "C1 模型 @conf 0.45，左=GT 人工标注，右=预测（绿=TP / 蓝=FP / 红=FN）")
pic(s, Path("outputs/demo/caries_report/demo_caries_anonymous-leftLateralView-1727089857497_compare.jpg"), 2.1, 1.15, w=9.1)
pic(s, Path("outputs/demo/caries_report/demo_caries_anonymous-maxillaryView-1727066058439_compare.jpg"), 2.1, 3.6, w=9.1)
bullets(s, [
    ("上：单病灶 case，1 检出 1 中（conf 0.69）", 0, None, False, 13),
    ("下：多病灶 case，3 GT 全部检出（conf 0.85 / 0.84 / 0.60），无 FP", 0, None, False, 13),
], y=6.35, h=0.9, size=13)

# ---------------- P18-20 Demos ----------------
def demo_slide(key, title, note_lines, img_name):
    s = blank()
    bar(s, title)
    pic(s, BASE / img_name, 0.75, 1.15, w=11.8)
    y = 1.15 + 11.8 * 408 / 1228 * (1228 / 1134)
    tb = s.shapes.add_textbox(Inches(0.55), Inches(4.35), Inches(12.2), Inches(2.4))
    tf = tb.text_frame
    tf.word_wrap = True
    first = True
    for ln, color, bold in note_lines:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        r = p.add_run()
        r.text = ln
        r.font.size = Pt(15)
        r.font.bold = bold
        r.font.color.rgb = color or RGBColor(0x26, 0x26, 0x26)
    caption(s, "左：GT 人工标注(红) ｜ 右：绿=TP 正确检出 · 深红=FN 漏检 · 蓝=FP 误检 · 黄框=实例置信度", y=7.02)


demo_slide("small", "Demo ① 小病灶（r<0.552%）", [
    ("图像 caries_0_2066（612×408，1 个 GT 实例）", None, True),
    ("检出 2 实例（conf 0.68 / 0.26）；实例级 IoU 0.824，TP Dice 0.904", None, False),
    ("像素级 P=0.45 / R=0.92：预测略大于 GT（边界外扩）—— 小病灶“宁多勿漏”取向的体现", ORANGE, False),
], "demo_small_caries_0_2066_compare.jpg")

demo_slide("medium", "Demo ② 中病灶（0.552%≤r<1.565%）", [
    ("图像 caries_0_7767（612×408，2 个 GT 实例）", None, True),
    ("检出 1 实例（conf 0.51）；实例级 IoU 0.852，TP Dice 0.920", None, False),
    ("全图 Dice 0.853，像素级 P=0.93 / R=0.79 —— 定位与形状质量都很高", GREEN, False),
], "demo_medium_caries_0_7767_compare.jpg")

demo_slide("large", "Demo ③ 大病灶（r≥1.565%）", [
    ("图像 caries_0_7671（612×408，2 个 GT 实例）", None, True),
    ("检出 2 实例（conf 0.47 / 0.34）；实例级 IoU 0.797，TP Dice 0.887", None, False),
    ("全图 Dice 0.875，像素级 P=0.90 / R=0.85 —— 大病灶整体覆盖良好", GREEN, False),
], "demo_large_caries_0_7671_compare.jpg")

# ---------------- P21 失败案例 ----------------
s = blank()
bar(s, "失败案例分析（漏检长什么样）")
pic(s, BASE / "demo_miss_caries_0_3129_compare.jpg", 1.7, 1.1, w=9.9)
bullets(s, [
    ("GT 大病灶 1869px（面积比 1.8%），管线零候选 —— 融合分未过阈值", 0, None, False, 15),
    ("该类“检测器+分类器同时盲”的图贡献了主要 FN —— 与 v3 错误分析结论一致", 0, None, False, 15),
    ("缓解方向：救援级联（已使中病灶召回 +2.1pt）、降融合阈值、难例挖掘、提高输入分辨率", 0, None, False, 15),
], y=4.15)
caption(s, "左：GT(红) ｜ 右：深红=漏检区域", y=7.02)

# ---------------- P22 结论 ----------------
s = blank()
bar(s, "结论与下一步")
bullets(s, [
    ("结论", 0, BLUE, True, 18),
    ("数据治理是本项目最大收益点：泄漏修复 + 重复/冲突标签清洗让所有后续指标可信", 1),
    ("V5 级联管线：实例级 F1 0.456 / TP Dice 0.818；图像级 Acc 0.881 / Spec 0.940 / AUROC 0.915", 1),
    ("模型当前定位：可靠的“定位提示 + 筛查”工具；不能作为完整定量测量使用", 1),
    ("下一步（按预期收益排序）", 0, BLUE, True, 18),
    ("补数据：宽视野域 +1000 张、拍摄分辨率 ≥1280px（小病灶信息瓶颈在输入端）", 1),
    ("牙齿分割约束（IO150K）：非牙面候选直接过滤 → 预期大幅砍 FP", 1),
    ("半监督自训练：对无标注口内图打伪标（阈值>0.9）扩充训练集", 1),
    ("患者级 ID 标注与划分，杜绝残余同患者泄漏", 1),
], y=1.2, size=15)

prs.save(str(OUT))
print(f"saved {OUT}  slides={len(prs.slides.__iter__.__self__._sldIdLst)}")
