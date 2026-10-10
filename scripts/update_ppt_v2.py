"""Update 完整检测汇报.pptx:
  - fonts: latin = Times New Roman, eastAsia = 宋体 (all runs, incl. tables)
  - text: white/light runs -> black (they sat on removed colored fills)
  - autoshapes: remove solid fills, thin gray outline, black text
  - slide backgrounds: explicit white
  - tables: three-line style (top/header/bottom rules, no fills, no banding)
  - slide 21 picture -> real plaque step image; slide 22 picture -> real
    6-panel pipeline demo figure
  - append: pipeline I/O three-line table slide + calculus demo slide
Backup: 完整检测汇报_backup.pptx
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import copy
import shutil
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

ROOT = Path(f"{_ROOT}")
PPT = ROOT / "outputs/demo/pipeline_walkthrough/完整检测汇报.pptx"
BAK = PPT.with_name("完整检测汇报_backup.pptx")
OUT_DIR = PPT.parent
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"

LIGHT_COLORS = {"FFFFFF", "DEEA F6", "DEEAF6", "9DC3E6", "FFF2CC"}


def ea_set(rPr, typeface="宋体"):
    for tag in ("a:ea",):
        el = rPr.find(qn(tag))
        if el is not None:
            rPr.remove(el)
    latin = rPr.find(qn("a:latin"))
    ea = rPr.makeelement(qn("a:ea"), {"typeface": typeface})
    if latin is not None:
        latin.addnext(ea)
    else:
        rPr.append(ea)


def style_run(r, black=False):
    rPr = r._r.get_or_add_rPr()
    r.font.name = "Times New Roman"          # sets a:latin
    ea_set(rPr, "宋体")
    if black:
        # strip explicit fill color and set black
        for f in rPr.findall(qn("a:solidFill")):
            rPr.remove(f)
        r.font.color.rgb = RGBColor(0, 0, 0)
    else:
        try:
            rgb = r.font.color.rgb
            if rgb is not None and str(rgb).upper() in LIGHT_COLORS:
                r.font.color.rgb = RGBColor(0, 0, 0)
        except Exception:
            pass


def walk_runs(shapes, black_all=False):
    for sh in shapes:
        if sh.shape_type == MSO_SHAPE_TYPE.GROUP:
            walk_runs(sh.shapes, black_all)
            continue
        frames = []
        if sh.has_text_frame:
            frames.append(sh.text_frame)
        if sh.has_table:
            for row in sh.table.rows:
                for c in row.cells:
                    frames.append(c.text_frame)
        for tf in frames:
            for para in tf.paragraphs:
                for r in para.runs:
                    style_run(r, black=black_all)


def strip_shape_fills(shapes):
    for sh in shapes:
        if sh.shape_type == MSO_SHAPE_TYPE.GROUP:
            strip_shape_fills(sh.shapes)
            continue
        if sh.shape_type == MSO_SHAPE_TYPE.AUTO_SHAPE:
            try:
                sp = sh._element
                spPr = sp.find(qn("p:spPr"))
                if spPr is None:
                    continue
                for tag in ("a:solidFill", "a:noFill", "a:gradFill", "a:blipFill",
                            "a:pattFill", "a:grpFill"):
                    for el in spPr.findall(qn(tag)):
                        spPr.remove(el)
                nofill = spPr.makeelement(qn("a:noFill"), {})
                # insert after a:prstGeom / a:custGeom (fill position)
                geom = spPr.find(qn("a:prstGeom")) or spPr.find(qn("a:custGeom"))
                if geom is not None:
                    geom.addnext(nofill)
                else:
                    spPr.append(nofill)
                # thin gray outline
                for tag in ("a:ln",):
                    for el in spPr.findall(qn(tag)):
                        spPr.remove(el)
                ln = spPr.makeelement(qn("a:ln"), {"w": "9525"})
                sf = ln.makeelement(qn("a:solidFill"), {})
                clr = sf.makeelement(qn("a:srgbClr"), {"val": "A6A6A6"})
                sf.append(clr)
                ln.append(sf)
                # ln goes after fill in spPr order
                nf = spPr.find(qn("a:noFill"))
                if nf is not None:
                    nf.addnext(ln)
                else:
                    spPr.append(ln)
            except Exception as exc:
                print("shape fill strip failed:", exc)


def white_bg(slide):
    bg = slide.background
    bg.fill.solid()
    bg.fill.fore_color.rgb = RGBColor(255, 255, 255)


def three_line_table(tbl):
    tblPr = tbl._tbl.find(qn("a:tblPr"))
    if tblPr is not None:
        tblPr.set("bandRow", "0")
        tblPr.set("firstRow", "1")
    n_rows = len(tbl.rows)

    def tc_borders(cell, top=None, bottom=None):
        tc = cell._tc
        tcPr = tc.find(qn("a:tcPr"))
        if tcPr is None:
            tcPr = tc.makeelement(qn("a:tcPr"), {})
            tx = tc.find(qn("a:txBody"))
            if tx is not None:
                tx.addnext(tcPr)
            else:
                tc.insert(0, tcPr)
        # remove existing borders and fills
        remove_tags = ("a:lnL", "a:lnR", "a:lnT", "a:lnB", "a:lnTlToBr", "a:lnBlToTr",
                       "a:solidFill", "a:noFill", "a:gradFill", "a:pattFill", "a:blipFill",
                       "a:grpFill", "a:cell3D")
        rest = []
        for el in list(tcPr):
            if el.tag in {qn(t) for t in remove_tags}:
                tcPr.remove(el)
            else:
                rest.append(el)
                tcPr.remove(el)
        # rebuild in schema order: lnL, lnR, lnT, lnB, [ln diag], fill
        def ln(tag, w):
            el = tcPr.makeelement(qn(tag), {"w": str(w), "cap": "flat"})
            sf = el.makeelement(qn("a:solidFill"), {})
            clr = sf.makeelement(qn("a:srgbClr"), {"val": "000000"})
            sf.append(clr)
            el.append(sf)
            tcPr.append(el)
        if top is not None:
            ln("a:lnT", top)
        if bottom is not None:
            ln("a:lnB", bottom)
        tcPr.append(tcPr.makeelement(qn("a:noFill"), {}))
        for el in rest:
            tcPr.append(el)

    for ri, row in enumerate(tbl.rows):
        for cell in row.cells:
            top = 19050 if ri == 0 else None            # 1.5pt top rule
            bottom = None
            if ri == 0:
                bottom = 9525                            # 0.75pt header rule
            if ri == n_rows - 1:
                bottom = 19050                           # 1.5pt bottom rule
            tc_borders(cell, top=top, bottom=bottom)
            for para in cell.text_frame.paragraphs:
                for r in para.runs:
                    rPr = r._r.get_or_add_rPr()
                    for f in rPr.findall(qn("a:solidFill")):
                        rPr.remove(f)
                    if ri == 0:
                        r.font.bold = True
                    r.font.color.rgb = RGBColor(0, 0, 0)
                    r.font.name = "Times New Roman"
                    ea_set(rPr, "宋体")


def replace_picture(slide, img_path: Path):
    """Replace the (first) picture shape on a slide, keep its bbox center."""
    pics = [sh for sh in slide.shapes if sh.shape_type == MSO_SHAPE_TYPE.PICTURE]
    if not pics:
        return False
    old = pics[0]
    cx, cy = old.left + old.width // 2, old.top + old.height // 2
    max_w, max_h = old.width, old.height
    old._element.getparent().remove(old._element)
    from PIL import Image
    w, h = Image.open(img_path).size
    scale = min(max_w / w, max_h / h)
    nw, nh = int(w * scale), int(h * scale)
    slide.shapes.add_picture(str(img_path), cx - nw // 2, cy - nh // 2, nw, nh)
    return True


def add_three_line_table_slide(prs, title, headers, rows, col_widths=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(0.8))
    p = tb.text_frame.paragraphs[0]
    r = p.add_run()
    r.text = title
    r.font.size = Pt(28)
    r.font.bold = True
    style_run(r)
    n_r, n_c = len(rows) + 1, len(headers)
    gt = slide.shapes.add_table(n_r, n_c, Inches(0.5), Inches(1.3),
                                Inches(12.3), Inches(0.4 * n_r))
    table = gt.table
    if col_widths:
        for i, wd in enumerate(col_widths):
            table.columns[i].width = Inches(wd)
    for j, h in enumerate(headers):
        table.cell(0, j).text = h
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            table.cell(i + 1, j).text = v
    three_line_table(table)
    return slide


def add_picture_slide(prs, title, img_path, caption=""):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.25), Inches(12.3), Inches(0.7))
    p = tb.text_frame.paragraphs[0]
    r = p.add_run()
    r.text = title
    r.font.size = Pt(28)
    r.font.bold = True
    style_run(r)
    from PIL import Image
    w, h = Image.open(img_path).size
    max_w, max_h = Inches(11.5), Inches(5.4)
    scale = min(max_w / w, max_h / h)
    nw, nh = int(w * scale), int(h * scale)
    slide.shapes.add_picture(str(img_path), int((Inches(13.33) - nw) / 2),
                             Inches(1.15), nw, nh)
    if caption:
        cb = slide.shapes.add_textbox(Inches(0.5), Inches(6.7), Inches(12.3), Inches(0.5))
        cp = cb.text_frame.paragraphs[0]
        cr = cp.add_run()
        cr.text = caption
        cr.font.size = Pt(14)
        style_run(cr)
    return slide


def main() -> None:
    shutil.copy2(PPT, BAK)
    prs = Presentation(str(PPT))
    for slide in prs.slides:
        white_bg(slide)
        walk_runs(slide.shapes, black_all=False)
        # white text -> black (was sitting on removed fills)
        def flip_white(shapes):
            for sh in shapes:
                if sh.shape_type == MSO_SHAPE_TYPE.GROUP:
                    flip_white(sh.shapes)
                    continue
                frames = []
                if sh.has_text_frame:
                    frames.append(sh.text_frame)
                if sh.has_table:
                    frames += [c.text_frame for row in sh.table.rows for c in row.cells]
                for tf in frames:
                    for para in tf.paragraphs:
                        for r in para.runs:
                            try:
                                if r.font.color.rgb is not None and \
                                        str(r.font.color.rgb).upper() in LIGHT_COLORS:
                                    r.font.color.rgb = RGBColor(0, 0, 0)
                            except Exception:
                                pass
        flip_white(slide.shapes)
        strip_shape_fills(slide.shapes)
        for sh in slide.shapes:
            if sh.has_table:
                three_line_table(sh.table)

    # slide 21 (plaque) & 22 (pipeline) pictures -> real demo images
    replace_picture(prs.slides[20], OUT_DIR / "step4_plaque.jpg")
    replace_picture(prs.slides[21], OUT_DIR / "pipeline_panels.jpg")

    # new slide: pipeline I/O three-line table
    rows = [
        ["0 输入", "-", "口内照片 (JPG/PNG)", "BGR 图像阵列"],
        ["1 质量检查", "Laplacian/亮度/高光阈值", "图像", "usable/blur/brightness JSON"],
        ["2 牙齿实例分割", "IOP-Compass YOLO11m-seg@896", "图像", "14-44 颗牙 mask+bbox+conf"],
        ["3 龋齿检测", "C2 YOLO11m@960 整图一次", "图像", "龋损 bbox+conf (F1 0.887)"],
        ["4 牙结石级联", "V5 S1分块→S2自适应窗口→Verifier→M3融合", "图像", "结石 mask+融合置信度+牙位归属"],
        ["5 牙菌斑分级", "Mendeley YOLO11s-cls@224 (thr 0.92)", "逐牙 crop", "每牙 P(plaque) + 二值判定"],
        ["6 病灶-牙位关联", "中心入 mask / 最大重叠规则", "病灶+牙 masks", "tooth_index / unassigned"],
        ["7 汇总输出", "V1 JSON schema + 可视化", "以上全部", "统一 JSON + 叠加可视化"],
    ]
    add_three_line_table_slide(
        prs, "完整检测流程 · 各步骤输入输出（真实运行）",
        ["步骤", "模型/方法", "输入", "输出"],
        rows, col_widths=[1.9, 4.2, 2.6, 3.6])

    # new slide: calculus demo image
    add_picture_slide(
        prs, "牙结石级联 · 完整流程 Demo",
        OUT_DIR / "step35_calculus.jpg",
        caption="来源：pipeline 实跑（caries_0_1508.jpg，Oral-Mamba test）— 检出 1 处牙结石，conf 0.45，已归属牙位")

    # re-style the two appended slides too (fonts/black)
    for slide in prs.slides:
        walk_runs(slide.shapes, black_all=False)
        white_bg(slide)

    prs.save(str(PPT))
    print("saved:", PPT, "slides:", len(prs.slides))


if __name__ == "__main__":
    main()
