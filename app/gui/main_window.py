"""Dental AI Assistant - main window (SRS modules 1/3/4/5/7)."""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
from app.qt_compat import QAction, QtCore, QtGui, QtWidgets

from app.gui.worker import BatchDemoWorker, InferenceWorker


def _outputs_root() -> Path:
    import sys

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


DEMO_DIR = Path(__file__).resolve().parents[1] / "demo"
DEMO_CASES = [
    ("Case 1 牙菌斑检测", DEMO_DIR / "plaque_demo.jpg"),
    ("Case 2 牙结石检测", DEMO_DIR / "calculus_demo.jpg"),
    ("Case 3 龋齿检测", DEMO_DIR / "caries_demo.jpg"),
    ("Case 4 三病灶联合", DEMO_DIR / "mixed_demo.jpg"),
]

_BLUE = "#1976d2"
_ORANGE = "#ef6c00"
_RED = "#c62828"


def _read_settings():
    return QtCore.QSettings("DentalAI", "DentalAIAssistant")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dental AI Assistant · 口腔AI多病灶智能检测系统")
        self.resize(1280, 800)
        self.image_path: str | None = None
        self.current: dict | None = None
        self.worker: InferenceWorker | None = None
        self._build_ui()

    # ---------------------------------------------------------------- UI
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # left: input
        left = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel("口腔图片输入")
        title.setStyleSheet(f"font-size:16px;font-weight:600;color:{_BLUE};")
        left.addWidget(title)
        self.image_label = QtWidgets.QLabel("请上传口腔照片\n(JPG / JPEG / PNG / BMP)")
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(560, 420)
        self.image_label.setStyleSheet(
            "border:2px dashed #b0bec5;border-radius:8px;background:#fafafa;color:#90a4ae;")
        left.addWidget(self.image_label, stretch=1)

        btns = QtWidgets.QGridLayout()
        self.btn_upload = QtWidgets.QPushButton("上传图片")
        self.btn_start = QtWidgets.QPushButton("开始检测")
        self.btn_start.setEnabled(False)
        self.btn_start.setStyleSheet(
            f"background:{_BLUE};color:white;font-weight:600;padding:6px;")
        self.btn_clear = QtWidgets.QPushButton("清空")
        self.btn_save = QtWidgets.QPushButton("保存结果")
        self.btn_save.setEnabled(False)
        self.btn_report = QtWidgets.QPushButton("生成报告")
        self.btn_report.setEnabled(False)
        for i, b in enumerate((self.btn_upload, self.btn_start,
                               self.btn_clear, self.btn_save, self.btn_report)):
            btns.addWidget(b, i // 3, i % 3)
        left.addLayout(btns)
        self.progress = QtWidgets.QLabel("")
        self.progress.setStyleSheet("color:#666;")
        left.addWidget(self.progress)
        layout.addLayout(left, stretch=11)

        # right: results
        right = QtWidgets.QVBoxLayout()
        rtitle = QtWidgets.QLabel("检测结果")
        rtitle.setStyleSheet(f"font-size:16px;font-weight:600;color:{_BLUE};")
        right.addWidget(rtitle)
        self.annotated_label = QtWidgets.QLabel("AI 标注图")
        self.annotated_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.annotated_label.setMinimumSize(560, 380)
        self.annotated_label.setStyleSheet(
            "border:1px solid #e0e0e0;border-radius:8px;background:#111;color:#777;")
        right.addWidget(self.annotated_label, stretch=3)

        self.summary_box = QtWidgets.QTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setStyleSheet("border:none;background:#f7f9fc;")
        right.addWidget(self.summary_box, stretch=2)
        layout.addLayout(right, stretch=13)

        # menu / toolbar actions
        menu = self.menuBar()
        m_demo = menu.addMenu("Demo")
        act_demo = QAction("Demo 展示模式", self)
        act_demo.triggered.connect(self.open_demo)
        m_demo.addAction(act_demo)
        m_set = menu.addMenu("系统设置")
        act_set = QAction("推理设置", self)
        act_set.triggered.connect(self.open_settings)
        m_set.addAction(act_set)

        self.btn_upload.clicked.connect(self.pick_image)
        self.btn_start.clicked.connect(self.run_detection)
        self.btn_clear.clicked.connect(self.clear_all)
        self.btn_save.clicked.connect(self.save_result)
        self.btn_report.clicked.connect(self.open_report)

    # ------------------------------------------------------------- actions
    def pick_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择口腔照片", "", "Images (*.jpg *.jpeg *.png *.bmp)")
        if not path:
            return
        self.load_image(path)

    def load_image(self, path: str):
        pm = QtGui.QPixmap(path)
        if pm.isNull():
            QtWidgets.QMessageBox.warning(self, "错误", "无法读取该图片文件")
            return
        self.image_path = path
        self.image_label.setPixmap(pm.scaled(
            self.image_label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation))
        self.btn_start.setEnabled(True)
        self.progress.setText(f"已加载: {Path(path).name}")

    def run_detection(self):
        if not self.image_path:
            return
        st = _read_settings()
        fast = st.value("fast_mode", "false") in ("true", True, 1)
        self.btn_start.setEnabled(False)
        self.progress.setText("检测中…（CPU 约 5-25 秒，取决于图片与模式）")
        QtWidgets.QApplication.processEvents()
        self.worker = InferenceWorker(self.image_path, fast_mode=fast)
        self.worker.progress.connect(self.progress.setText)
        self.worker.finished_ok.connect(self.on_done)
        self.worker.failed.connect(self.on_fail)
        self.worker.start()

    def on_done(self, payload: dict):
        self.btn_start.setEnabled(True)
        self.current = payload
        result, risk = payload["result"], payload["risk"]
        pm = QtGui.QPixmap(payload["artifacts"]["image"])
        self.annotated_label.setPixmap(pm.scaled(
            self.annotated_label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation))
        s = result["summary"]
        t = result.get("timing", {}).get("total", "-")
        html = f"""
        <div style='font-size:15px;'>
        <b style='font-size:19px;color:{_BLUE};'>检测结果</b>
        <span style='color:#555;'>· 检出牙位 {s['tooth_count']} 颗 · 推理耗时 {t}s</span>
        <table cellpadding='6' style='margin-top:8px;'>
        <tr><th>病灶</th><th>数量</th></tr>
        <tr><td style='color:{_BLUE};font-weight:600;'>牙菌斑</td>
            <td>{risk['plaque']['count']} 牙位</td></tr>
        <tr><td style='color:{_ORANGE};font-weight:600;'>牙结石</td>
            <td>{risk['calculus']['count']} 处</td></tr>
        <tr><td style='color:{_RED};font-weight:600;'>龋齿</td>
            <td>{risk['caries']['count']} 处</td></tr>
        </table>
        <div style='margin-top:6px;'><b>干预建议</b></div>
        <ol style='margin-top:2px;'>
        {''.join('<li><b>' + it['lesion'] + '</b>：' + it['advice'][0] + '；' +
                 '；'.join(it['advice'][1:2]) + '。</li>' for it in risk['items'])}
        </ol></div>"""
        self.summary_box.setHtml(html)
        self.btn_save.setEnabled(True)
        self.btn_report.setEnabled(True)
        self.progress.setText("检测完成")

    def on_fail(self, msg: str):
        self.btn_start.setEnabled(True)
        self.progress.setText("")
        QtWidgets.QMessageBox.critical(self, "检测失败", msg)

    def clear_all(self):
        self.image_path = None
        self.current = None
        self.image_label.clear()
        self.image_label.setText("请上传口腔照片\n(JPG / JPEG / PNG / BMP)")
        self.annotated_label.clear()
        self.annotated_label.setText("AI 标注图")
        self.summary_box.clear()
        self.progress.setText("")
        self.btn_start.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.btn_report.setEnabled(False)

    def save_result(self):
        if not self.current:
            return
        src = Path(self.current["artifacts"]["image"])
        dst, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "保存检测结果", str(Path.home() / src.name), "Images (*.jpg)")
        if dst:
            shutil.copyfile(src, dst)
            self.progress.setText(f"已保存: {dst}")

    def open_report(self):
        if not self.current:
            return
        rep_dir = _outputs_root() / "results" / "reports"
        stem = Path(self.current["result"]["image_id"]).stem
        pdf = rep_dir / f"{stem}_report.pdf"
        htmlp = rep_dir / f"{stem}_report.html"
        if not pdf.exists():
            QtWidgets.QMessageBox.information(
                self, "提示", "报告尚未生成，请先运行检测（已自动生成）。")
            return
        import webbrowser

        webbrowser.open(htmlp.as_uri())
        QtWidgets.QMessageBox.information(
            self, "报告已生成",
            f"PDF: {pdf}\nHTML: {htmlp}")

    # ----------------------------------------------------------- demo mode
    def open_demo(self):
        dlg = DemoDialog(self)
        dlg.exec()

    def open_settings(self):
        st = _read_settings()
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("系统设置")
        v = QtWidgets.QVBoxLayout(dlg)
        cb = QtWidgets.QCheckBox("快速模式（跳过牙结石级联的 TTA/切片，速度约 2 倍，灵敏度略降）")
        cb.setChecked(st.value("fast_mode", "false") in ("true", True, 1))
        v.addWidget(cb)
        info = QtWidgets.QLabel(
            "标准模式与训练侧评测管线逐位一致；快速模式适合低配 CPU。\n"
            "如机器有 NVIDIA GPU，可设置环境变量 DENTALAI_EP=cuda 启用加速。")
        info.setStyleSheet("color:#666;")
        v.addWidget(info)
        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        v.addWidget(bb)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            st.setValue("fast_mode", "true" if cb.isChecked() else "false")


class DemoDialog(QtWidgets.QDialog):
    """SRS module 7: one-click demo of the four preset cases."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Demo 展示 · 预设案例自动检测")
        self.resize(1180, 760)
        self.worker: BatchDemoWorker | None = None
        v = QtWidgets.QVBoxLayout(self)

        top = QtWidgets.QHBoxLayout()
        self.case_list = QtWidgets.QListWidget()
        self.case_list.setMaximumWidth(280)
        for name, path in DEMO_CASES:
            self.case_list.addItem(
                QtWidgets.QListWidgetItem(f"{name}\n      {Path(path).name}"))
        top.addWidget(self.case_list)

        right = QtWidgets.QVBoxLayout()
        self.before = QtWidgets.QLabel("原图")
        self.before.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.before.setMinimumSize(420, 300)
        self.before.setStyleSheet(
            "border:1px solid #ccc;background:#111;color:#888;")
        self.after = QtWidgets.QLabel("AI 检测结果")
        self.after.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.after.setMinimumSize(420, 300)
        self.after.setStyleSheet(
            "border:1px solid #ccc;background:#111;color:#888;")
        pair = QtWidgets.QHBoxLayout()
        pair.addWidget(self.before)
        pair.addWidget(self.after)
        right.addLayout(pair, stretch=3)
        self.verdict = QtWidgets.QLabel("点击「自动运行全部案例」开始 Demo")
        self.verdict.setStyleSheet("font-size:15px;")
        self.verdict.setWordWrap(True)
        right.addWidget(self.verdict, stretch=2)
        top.addLayout(right, stretch=1)
        v.addLayout(top, stretch=1)

        bottom = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton("自动运行全部案例")
        self.btn_run.setStyleSheet(
            f"background:{_BLUE};color:white;font-weight:600;padding:6px 18px;")
        self.btn_run.clicked.connect(self.run_all)
        self.status = QtWidgets.QLabel("")
        bottom.addWidget(self.btn_run)
        bottom.addWidget(self.status, stretch=1)
        v.addLayout(bottom)

    def run_all(self):
        self.btn_run.setEnabled(False)
        self.status.setText("模型加载中…")
        self.worker = BatchDemoWorker(DEMO_CASES)
        self.worker.progress.connect(self.status.setText)
        self.worker.step.connect(self.on_step)
        self.worker.failed.connect(self.on_fail)
        self.worker.all_done.connect(self.on_all_done)
        self.worker.start()

    def _scaled(self, label: QtWidgets.QLabel, path: str):
        pm = QtGui.QPixmap(path)
        label.setPixmap(pm.scaled(
            label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation))

    def on_step(self, name: str, payload: dict):
        self._scaled(self.before, payload["image"])
        self._scaled(self.after, payload["annotated"])
        risk = payload["risk"]
        self.verdict.setText(
            f"<b>{name}</b><br>"
            f"牙菌斑 {risk['plaque']['count']} 牙位 · "
            f"牙结石 {risk['calculus']['count']} 处 · "
            f"龋齿 {risk['caries']['count']} 处<br>"
            + "；".join(f"{it['lesion']}：{it['advice'][0]}" for it in risk["items"]))
        self.status.setText(f"完成 {name}")

    def on_all_done(self):
        self.btn_run.setEnabled(True)
        self.status.setText("全部案例演示完成（标注图保存在 results/demo/）")

    def on_fail(self, msg: str):
        self.btn_run.setEnabled(True)
        self.status.setText("")
        QtWidgets.QMessageBox.critical(self, "Demo 运行失败", msg)
