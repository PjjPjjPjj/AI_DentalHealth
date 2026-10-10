"""Pipeline smoke tests (plan 27): quality gate, empty/multi lesion cases,
unreadable image, no weights behavior, JSON output."""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.inference.image_quality import check_quality
from src.inference.pipeline import OralAIPipeline
from src.utils.paths import CONFIGS_DIR

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))


def _synthetic_img() -> np.ndarray:
    img = np.full((480, 640, 3), 120, dtype=np.uint8)
    cv2.rectangle(img, (200, 150), (440, 330), (90, 90, 90), -1)
    cv2.rectangle(img, (260, 200), (380, 280), (200, 200, 200), -1)
    return img


class _StubTooth:
    def predict(self, img):
        m = np.zeros(img.shape[:2], dtype=np.uint8)
        m[150:330, 200:440] = 1
        return [{"mask": m, "bbox": [200.0, 150.0, 440.0, 330.0], "conf": 0.9, "cls": 0}]


class _StubCalculus:
    def predict(self, img):
        m = np.zeros(img.shape[:2], dtype=np.uint8)
        m[250:300, 250:400] = 1
        return [{"mask": m, "conf": 0.8, "cls": 0}]


class _StubCaries:
    def predict(self, img):
        return [{"bbox": [260.0, 200.0, 380.0, 280.0], "conf": 0.6, "cls": 0},
                {"bbox": [5.0, 5.0, 40.0, 40.0], "conf": 0.5, "cls": 0}]


class _StubPlaque:
    def predict_proba(self, crop):
        return 0.7


def test_quality_blurry_flat():
    img = np.full((400, 400, 3), 0, dtype=np.uint8)
    q = check_quality(img, {"blur_laplacian_threshold": 100.0, "brightness_underexposed": 60.0,
                            "brightness_overexposed": 200.0, "dark_pixel_ratio_limit": 0.35,
                            "bright_pixel_ratio_limit": 0.35, "highlight_ratio_limit": 0.12,
                            "min_size": 320})
    assert q["usable"] is False  # pure black -> underexposed + blur


def test_quality_good():
    img = _synthetic_img()
    rng = np.random.default_rng(0)
    img = np.clip(img.astype(np.float32) + rng.normal(0, 12, img.shape), 0, 255).astype(np.uint8)
    q = check_quality(img, {"blur_laplacian_threshold": 50.0, "brightness_underexposed": 60.0,
                            "brightness_overexposed": 200.0, "dark_pixel_ratio_limit": 0.35,
                            "bright_pixel_ratio_limit": 0.35, "highlight_ratio_limit": 0.12,
                            "min_size": 320})
    assert q["usable"] is True


def test_quality_unreadable():
    q = check_quality(Path("/no/such.jpg"), {})
    assert q["usable"] is False
    assert "error" in q


def test_pipeline_smoke_stub_models():
    pipe = OralAIPipeline.__new__(OralAIPipeline)  # instance w/o heavy init
    pipe.quality_cfg = {"min_size": 320}
    pipe.assoc_cfg = {"calculus_min_overlap": 0.2, "caries_center_rule": True, "caries_min_iou": 0.01}
    pipe.tooth = _StubTooth()
    pipe.calculus = _StubCalculus()
    pipe.caries = _StubCaries()
    pipe.plaque = _StubPlaque()
    result = pipe.run_array(_synthetic_img(), "smoke.jpg")
    assert len(result["teeth"]) == 1
    t = result["teeth"][0]
    assert t["plaque"]["score"] == 0.7
    assert t["calculus"]["present"] is True
    assert t["caries"]["present"] is True
    # second caries box falls outside tooth -> unassigned, not forced
    assert any(l["type"] == "caries" for l in result["unassigned_lesions"])


def test_pipeline_no_teeth():
    pipe = OralAIPipeline.__new__(OralAIPipeline)
    pipe.quality_cfg = {"min_size": 320}
    pipe.assoc_cfg = {"calculus_min_overlap": 0.2, "caries_center_rule": True, "caries_min_iou": 0.01}
    pipe.tooth = None
    pipe.calculus = _StubCalculus()
    pipe.caries = _StubCaries()
    pipe.plaque = _StubPlaque()
    result = pipe.run_array(_synthetic_img(), "none.jpg")
    assert result["teeth"] == []
    assert result["unassigned_lesions"]
    import json
    json.dumps(result)  # serializable


def test_pipeline_missing_weights_raises():
    with pytest.raises(FileNotFoundError):
        OralAIPipeline(weights={"tooth": "no/such.pt", "calculus": "no/such.pt",
                                "caries": "no/such.pt", "plaque": "no/such.pt"})


def test_pipeline_skip_missing_models():
    pipe = OralAIPipeline(skip_missing_models=True,
                          weights={"tooth": "no/such.pt", "calculus": "no/such.pt",
                                   "caries": "no/such.pt", "plaque": "no/such.pt"})
    result = pipe.run_array(_synthetic_img(), "skip.jpg")
    assert result["teeth"] == []
    import json
    json.dumps(result)


def test_pipeline_real_weights_e2e():
    """End-to-end with the real v2 weights (tooth+caries+plaque, calculus off).

    Skips automatically when any weight file is absent (CI / fresh clones).
    """
    import json
    from pathlib import Path

    root = Path(f"{_ROOT}")
    weights = {
        "tooth": root / "outputs/iop_toothseg/weights/best.pt",
        "caries": root / "outputs/caries_c2/weights/best.pt",
        "plaque": root / "outputs/plaque_bin_v1/weights/best.pt",
    }
    if not all(p.exists() for p in weights.values()):
        pytest.skip("real weights not available")
    img_path = root / "demo_images/intraoral_real.jpg"
    if not img_path.exists():
        pytest.skip("demo image not available")
    pipe = OralAIPipeline()
    result = pipe.run(img_path)
    assert not validate_result_json(result)
    assert len(result["teeth"]) > 0
    json.dumps(result)  # serializable


def validate_result_json(result):
    from src.inference.schemas import validate_result
    return validate_result(result)
