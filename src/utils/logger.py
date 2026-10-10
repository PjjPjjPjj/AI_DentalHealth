"""Logging: console + rotating file handler.

Every module should use `get_logger(name)` so all log lines are captured
into outputs/logs/ as well as stdout.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .paths import LOGS_DIR

_configured = False
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "oralai") -> logging.Logger:
    global _configured
    logger = logging.getLogger(name)
    if not _configured:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        logger.setLevel(logging.INFO)
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(_LOG_FORMAT))
        file_handler = RotatingFileHandler(
            LOGS_DIR / "oralai.log", maxBytes=20 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(console)
        logger.addHandler(file_handler)
        logger.propagate = False
        _configured = True
    return logger
