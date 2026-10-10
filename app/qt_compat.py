"""Qt binding compatibility: prefer PySide6 (SRS recommendation), fall back to
PyQt5 (same API). The app code imports from here."""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

try:
    from PySide6 import QtCore, QtGui, QtWidgets

    Signal = QtCore.Signal
    QAction = QtGui.QAction          # Qt6: moved out of QtWidgets
    QT_BINDING = "PySide6"
except Exception:  # pragma: no cover - dev machines without PySide6
    from PyQt5 import QtCore, QtGui, QtWidgets

    Signal = QtCore.pyqtSignal
    QAction = QtWidgets.QAction      # Qt5: lives in QtWidgets
    QT_BINDING = "PyQt5"

__all__ = ["QtCore", "QtGui", "QtWidgets", "Signal", "QAction", "QT_BINDING"]
