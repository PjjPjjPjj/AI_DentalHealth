"""Startup launcher (SRS: [开始检测] [Demo展示] [系统设置])."""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

from app.qt_compat import QtCore, QtGui, QtWidgets

_BLUE = "#1976d2"
_BG = "#10243e"


class Launcher(QtWidgets.QDialog):
    """Startup screen with the three SRS entry points."""

    CHOICE_RUN, CHOICE_DEMO = 1, 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self.choice = 0
        self.setWindowTitle("Dental AI Assistant")
        self.setFixedSize(520, 420)
        self.setStyleSheet(f"background-color:{_BG};")

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(40, 36, 40, 36)
        v.setSpacing(14)

        title = QtWidgets.QLabel("Dental AI Assistant")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color:white;font-size:26px;font-weight:700;")
        sub = QtWidgets.QLabel("AI-powered Oral Health Screening System\n口腔AI多病灶智能检测系统")
        sub.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        sub.setStyleSheet("color:#9fc3ea;font-size:13px;")
        v.addWidget(title)
        v.addWidget(sub)
        v.addSpacing(18)

        btn_run = QtWidgets.QPushButton("开始检测")
        btn_run.setMinimumHeight(46)
        btn_run.setStyleSheet(
            f"QPushButton{{background:{_BLUE};color:white;font-size:15px;"
            f"font-weight:600;border:none;border-radius:6px;}}"
            "QPushButton:hover{background:#1565c0;}")
        btn_run.clicked.connect(lambda: self._done(self.CHOICE_RUN))

        btn_demo = QtWidgets.QPushButton("Demo 展示")
        btn_demo.setMinimumHeight(46)
        btn_demo.setStyleSheet(
            "QPushButton{background:#ffffff;color:#10243e;font-size:15px;"
            "font-weight:600;border:none;border-radius:6px;}"
            "QPushButton:hover{background:#e3f2fd;}")
        btn_demo.clicked.connect(lambda: self._done(self.CHOICE_DEMO))

        btn_set = QtWidgets.QPushButton("系统设置")
        btn_set.setMinimumHeight(40)
        btn_set.setStyleSheet(
            "QPushButton{background:transparent;color:#9fc3ea;font-size:13px;"
            "border:1px solid #3a5a80;border-radius:6px;}"
            "QPushButton:hover{color:white;}")
        btn_set.clicked.connect(self.open_settings)

        v.addWidget(btn_run)
        v.addWidget(btn_demo)
        v.addSpacing(6)
        v.addWidget(btn_set)
        v.addStretch(1)
        tip = QtWidgets.QLabel("检测内容：牙菌斑 · 牙结石 · 龋齿\n仅供筛查参考，不构成临床诊断")
        tip.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        tip.setStyleSheet("color:#5f7d9c;font-size:11px;")
        v.addWidget(tip)

    def _done(self, choice: int):
        self.choice = choice
        self.accept()

    def open_settings(self):
        from app.gui.main_window import MainWindow

        MainWindow.open_settings(self)
