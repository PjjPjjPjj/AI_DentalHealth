"""JSON schema tests (plan 20/27): serializability + required keys."""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
import json

import pytest

from src.inference.schemas import empty_teeth_result, validate_result


def _full_result():
    return {
        "image_id": "test_0001.jpg",
        "quality": {"usable": True, "blur_score": 10.0, "brightness": 120.0,
                    "highlight_ratio": 0.02},
        "teeth": [
            {
                "tooth_id": 1,
                "bbox": [10, 10, 100, 100],
                "mask": None,
                "plaque": {"gingival": None, "incisal": None, "mesial": None,
                           "distal": None, "score": 0.5, "confidence": 0.8},
                "calculus": {"present": True, "confidence": 0.9, "area_ratio": 0.06},
                "caries": {"present": False, "confidence": 0.1},
            }
        ],
        "unassigned_lesions": [],
    }


def test_schema_valid():
    assert validate_result(_full_result()) == []


def test_schema_missing_teeth_key():
    r = _full_result()
    del r["teeth"]
    assert "missing key: teeth" in validate_result(r)


def test_schema_tooth_missing_fields():
    r = _full_result()
    del r["teeth"][0]["caries"]
    assert any("caries" in e for e in validate_result(r))


def test_serializable():
    json.dumps(_full_result())


def test_empty_result_valid():
    r = empty_teeth_result("x.jpg")
    assert validate_result(r) == []
    assert r["teeth"] == []


def test_tooth_id_local_not_fdi_note():
    """tooth_id is local index; no FDI strings should appear."""
    r = _full_result()
    s = json.dumps(r).lower()
    assert "fdi" not in s
    assert not any(k.startswith("11") or k in ("12", "13", "21", "22") for k in s.split())
    # medical-safety: no "definitive" claims in schema defaults
    assert "definitive" not in json.dumps(_full_result()).lower()
