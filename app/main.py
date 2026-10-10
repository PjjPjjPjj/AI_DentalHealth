"""Dental AI Assistant entry point.

Dev run:    PYTHONPATH=<repo> python -m app.main
Frozen exe: DentalAI.exe (PyInstaller, see app/DentalAI.spec)
Headless:   python -m app.main --headless <image...>
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import sys
from pathlib import Path


def _icon() -> str | None:
    p = Path(__file__).parent / "assets" / "icon.png"
    return str(p) if p.exists() else None


def main() -> int:
    if "--headless" in sys.argv:
        sys.argv.remove("--headless")
        from app.headless import main as hl

        return hl()

    from app.qt_compat import QtCore, QT_BINDING, QtGui, QtWidgets

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Dental AI Assistant")
    app.setStyle("Fusion")
    icon = _icon()

    from app.gui.main_window import MainWindow

    icon = _icon()
    if icon:
        win_icon = QtGui.QIcon(icon)

    # SRS startup screen: [开始检测] [Demo展示] [系统设置]
    choice = 0
    try:
        from app.gui.launcher import Launcher

        launcher = Launcher()
        if icon:
            launcher.setWindowIcon(QtGui.QIcon(icon))
        launcher.exec()
        choice = launcher.choice
    except Exception:
        pass
    if not choice:  # closed without choosing -> default to main window
        choice = 1

    win = MainWindow()
    if icon:
        win.setWindowIcon(QtGui.QIcon(icon))
    win.show()
    if choice == 2:
        win.open_demo()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
