"""Generic config loading with deep merge (YAML -> dict)."""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay into base (overlay wins); returns new dict."""
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve_config(config_path: str | Path, overrides: dict | None = None) -> dict:
    cfg = load_yaml(config_path)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg
