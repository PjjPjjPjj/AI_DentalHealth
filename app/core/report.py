"""Detection report generation: HTML + PDF (reportlab, embedded NotoSansSC)."""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import base64
import html
import time
from pathlib import Path

FONT_PATH = Path(__file__).resolve().parents[1] / "fonts" / "NotoSansSC.ttf"


def _b64img(path: str | Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def _risk_table(assess: dict) -> list:
    rows = []
    for key, name in (("plaque", "牙菌斑"), ("calculus", "牙结石"), ("caries", "龋齿")):
        it = assess[key]
        count = f"{it['count']} 牙位" if key == "plaque" else f"{it['count']} 处"
        rows.append((name, count))
    return rows


def make_html(image_path: str | Path, annotated_path: str | Path,
              result: dict, assess: dict, out_path: str | Path) -> Path:
    rows = _risk_table(assess)
    tr = "".join(
        f"<tr><td>{n}</td><td>{c}</td></tr>"
        for n, c in rows)
    advice_rows = "".join(
        "<li><b>" + html.escape(it["lesion"]) + "</b><ul>" +
        "".join(f"<li>{html.escape(a)}</li>" for a in it["advice"]) + "</ul></li>"
        for it in assess["items"])
    warn = ""
    if result.get("quality", {}).get("usable") is False:
        warn = ("<p style='color:#f44336'>⚠ 图片质量提示："
                + html.escape("; ".join(result.get("warnings", [])))
                + "</p>")
    doc = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>AI 口腔健康检测报告</title><style>
body {{ font-family: "Noto Sans SC", "Microsoft YaHei", sans-serif; margin: 32px auto;
       max-width: 860px; color: #1a1a1a; }}
h1 {{ border-bottom: 3px solid #1976d2; padding-bottom: 8px; }}
.meta {{ color: #666; font-size: 13px; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
td, th {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
th {{ background: #f2f6fb; }}
.imgs img {{ max-width: 100%; border: 1px solid #ccc; margin: 6px 0; }}
ul {{ line-height: 1.7; }}
</style></head><body>
<h1>AI 口腔健康检测报告</h1>
<p class="meta">Dental AI Assistant · 生成时间 {time.strftime('%Y-%m-%d %H:%M:%S')} ·
文件 {html.escape(Path(image_path).name)} ·
分辨率 {result['image_size']['width']}×{result['image_size']['height']} ·
耗时 {result.get('timing', {}).get('total', '-')}s</p>
{warn}
<h2>检测结果</h2>
<table><tr><th>病灶</th><th>数量</th></tr>{tr}</table>
<h2>检测图片</h2>
<div class="imgs"><img src="data:image/jpeg;base64,{_b64img(annotated_path)}"></div>
<h2>干预建议</h2><ul>{advice_rows}</ul>
<p class="meta">本报告由 AI 辅助生成，仅供筛查参考，不构成临床诊断依据。</p>
</body></html>"""
    out_path = Path(out_path)
    out_path.write_text(doc, encoding="utf-8")
    return out_path


def make_pdf(image_path: str | Path, annotated_path: str | Path,
             result: dict, assess: dict, out_path: str | Path) -> Path:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (Image as RLImage, Paragraph, SimpleDocTemplate,
                                    Spacer, Table, TableStyle)
    from reportlab.lib import colors as C

    pdfmetrics.registerFont(TTFont("NotoSC", str(FONT_PATH)))
    doc = SimpleDocTemplate(str(out_path), pagesize=A4,
                            leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=14 * mm, bottomMargin=14 * mm)
    h1 = ParagraphStyle("h1", fontName="NotoSC", fontSize=18, leading=24,
                        spaceAfter=6)
    h2 = ParagraphStyle("h2", fontName="NotoSC", fontSize=13, leading=18,
                        spaceBefore=10, spaceAfter=4,
                        textColor=C.HexColor("#1976d2"))
    body = ParagraphStyle("body", fontName="NotoSC", fontSize=9.5, leading=14)
    small = ParagraphStyle("small", fontName="NotoSC", fontSize=8, leading=11,
                           textColor=C.HexColor("#666666"))
    cell = ParagraphStyle("cell", fontName="NotoSC", fontSize=9, leading=12)

    story = [Paragraph("AI 口腔健康检测报告", h1),
             Paragraph(f"Dental AI Assistant · {time.strftime('%Y-%m-%d %H:%M:%S')} · "
                       f"文件 {Path(image_path).name} · "
                       f"{result['image_size']['width']}×{result['image_size']['height']}"
                       f" · 推理耗时 {result.get('timing', {}).get('total', '-')}s", small),
             Spacer(1, 4)]
    if result.get("quality", {}).get("usable") is False:
        story.append(Paragraph("⚠ 图片质量提示：" + "; ".join(result.get("warnings", [])),
                               ParagraphStyle("w", parent=small,
                                              textColor=C.HexColor("#f44336"))))
    header = [Paragraph("病灶", cell), Paragraph("数量", cell)]
    data = [header]
    for n, c in _risk_table(assess):
        data.append([Paragraph(n, cell), Paragraph(c, cell)])
    tbl = Table(data, colWidths=[120, 80])
    tbl.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, C.HexColor("#dddddd")),
        ("BACKGROUND", (0, 0), (-1, 0), C.HexColor("#f2f6fb")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story += [tbl, Paragraph("检测图片", h2),
              RLImage(str(annotated_path), width=178 * mm,
                      height=178 * mm * (result["image_size"]["height"]
                                         / max(result["image_size"]["width"], 1))),
              Paragraph("干预建议", h2)]
    for it in assess["items"]:
        story.append(Paragraph(f"• <b>{it['lesion']}</b>", body))
        for a in it["advice"]:
            story.append(Paragraph(f"　- {a}", body))
    story.append(Spacer(1, 8))
    story.append(Paragraph("本报告由 AI 辅助生成，仅供筛查参考，不构成临床诊断依据。", small))
    doc.build(story)
    return Path(out_path)


def save_results(image_path, annotated_path, result, assess, out_dir: Path,
                 stem: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}_result.json").write_text(
        __import__("json").dumps(result, ensure_ascii=False, indent=1),
        encoding="utf-8")
    html_p = make_html(image_path, annotated_path, result, assess,
                       out_dir / f"{stem}_report.html")
    pdf_p = make_pdf(image_path, annotated_path, result, assess,
                     out_dir / f"{stem}_report.pdf")
    return {"json": out_dir / f"{stem}_result.json",
            "html": html_p, "pdf": pdf_p}
