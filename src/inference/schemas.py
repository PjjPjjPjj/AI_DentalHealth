"""JSON output schema for OralAI-V1 (plan 20).

Validated by tests/test_schema.py. All fields present; tooth_id is a LOCAL
index (1..N) and must never be claimed as FDI numbering.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

from typing import Any

SCHEMA_KEYS = {
    "image_id", "quality", "teeth", "unassigned_lesions",
}


def validate_result(result: dict) -> list[str]:
    """Return list of violations; empty list = valid."""
    errs = []
    for k in SCHEMA_KEYS:
        if k not in result:
            errs.append(f"missing key: {k}")
    for tooth in result.get("teeth", []):
        for k in ["tooth_id", "bbox", "plaque", "calculus", "caries", "mask"]:
            if k not in tooth:
                errs.append(f"tooth missing key: {k}")
        if not all(k in tooth for k in ["plaque", "calculus", "caries"]):
            continue
        p, cal, c = tooth["plaque"], tooth["calculus"], tooth["caries"]
        for k in ["gingival", "incisal", "mesial", "distal", "score", "confidence"]:
            if k not in p:
                errs.append(f"plaque missing key: {k}")
        for k in ["present", "confidence", "area_ratio"]:
            if k not in cal:
                errs.append(f"calculus missing key: {k}")
        for k in ["present", "confidence"]:
            if k not in c:
                errs.append(f"caries missing key: {k}")
    return errs


def empty_teeth_result(image_id: str, quality: dict | None = None) -> dict[str, Any]:
    return {
        "image_id": image_id,
        "quality": quality or {
            "usable": True, "blur_score": 0.0, "brightness": 0.0,
            "highlight_ratio": 0.0,
        },
        "teeth": [],
        "unassigned_lesions": [],
    }
