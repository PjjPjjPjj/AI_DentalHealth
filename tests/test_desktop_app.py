"""Desktop app tests (run with appenv python).

    PYTHONPATH=<repo> <appenv>/bin/pytest tests/test_desktop_app.py -v
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import importlib.util

def _desktop_env_ok() -> bool:
    try:
        import numpy
        import onnxruntime

        onnxruntime.get_available_providers()
        # onnxruntime 1.x wheels are incompatible with numpy>=2 (segfaults)
        if int(numpy.__version__.split(".")[0]) >= 2:
            return False
    except Exception:
        return False
    return bool(importlib.util.find_spec("PyQt5")
                or importlib.util.find_spec("PySide6"))


if not _desktop_env_ok():
    pytest.skip("desktop app environment (working onnxruntime + Qt) not available",
                allow_module_level=True)

ROOT = Path(__file__).resolve().parents[1]
DEMO_IMG = ROOT / "demo_images" / "intraoral_163.jpg"

MODELS = ROOT / "app" / "models"
MODELS_OK = all(
    (MODELS / n).exists()
    for n in ("tooth_seg.onnx", "calculus_s1.onnx", "calculus_s2.onnx",
              "calculus_verifier.onnx", "calculus_fusion.npz", "caries_det.onnx",
              "plaque_cls.onnx"))


def _fake_result() -> dict:
    teeth = []
    for i in range(6):
        teeth.append({
            "tooth_id": i + 1,
            "bbox": [50 + 60 * i, 100, 100 + 60 * i, 200],
            "mask": None,
            "plaque": {"score": 0.97 if i < 3 else 0.10,
                       "present": i < 3, "confidence": 0.97 if i < 3 else 0.10},
            "calculus": {"present": i == 0, "confidence": 0.8,
                         "area_ratio": 0.05 if i == 0 else 0.0},
            "caries": {"present": i == 1, "confidence": 0.9},
        })
    return {
        "image_id": "fake.jpg",
        "image_size": {"width": 500, "height": 300},
        "quality": {"usable": True, "blur_score": 150.0, "brightness": 120.0,
                    "underexposed": False, "overexposed": False,
                    "highlight_ratio": 0.0, "highlight_warning": False,
                    "blurry": False},
        "teeth": teeth,
        "calculus_lesions": [{"type": "calculus", "present": True, "confidence": 0.8,
                              "area_px": 500,
                              "bbox": [50, 100, 100, 200], "polygon": [[[0.1, 0.3], [0.2, 0.3], [0.15, 0.5]]],
                              "tooth_index": 0, "overlap_ratio": 0.8, "area_ratio": 0.05}],
        "caries_lesions": [{"type": "caries", "present": True, "confidence": 0.9,
                            "bbox": [110, 100, 160, 200], "tooth_index": 1}],
        "unassigned_lesions": [],
        "warnings": [],
        "timing": {"total": 1.0},
    }


# ---------------------------------------------------------------- risk
def test_risk_assess_counts_and_advice():
    from app.core.risk import assess

    r = _fake_result()
    a = assess(r)
    assert a["plaque"]["count"] == 3
    assert a["calculus"]["count"] == 1
    assert a["caries"]["count"] == 1
    assert "score" not in a and "overall_level" not in a
    assert len(a["items"]) == 3
    for it in a["items"]:
        assert it["advice"] and it["lesion"]


def test_risk_empty_is_clean():
    from app.core.risk import assess

    r = _fake_result()
    r["teeth"] = []
    r["calculus_lesions"] = []
    r["caries_lesions"] = []
    a = assess(r)
    assert a["plaque"]["count"] == 0
    assert a["items"][0]["lesion"] == "综合"
    assert a["items"][0]["advice"]


# ---------------------------------------------------------- visualize/report
def _font_available() -> bool:
    """True when the (non-distributed) Noto Sans SC font is present."""
    for candidate in (
        Path(__file__).resolve().parents[1] / "app" / "fonts" / "NotoSansSC.ttf",
        Path(__file__).resolve().parents[1] / "assets" / "fonts" / "NotoSansSC.otf",
    ):
        if candidate.exists():
            return True
    return False


def test_visualize_and_report(tmp_path):
    from app.core.risk import assess
    from app.core.report import make_html, make_pdf
    from app.core.visualize import annotate, collect_lesion_masks

    # The CJK font is not distributed with the repo (SIL OFL-1.1).
    # See THIRD_PARTY_NOTICES.md section 2 for how to obtain it.
    if not _font_available():
        pytest.skip("CJK font missing — download Noto Sans SC (see THIRD_PARTY_NOTICES.md)")

    r = _fake_result()
    risk = assess(r)
    img = np.full((300, 500, 3), 90, np.uint8)
    masks = collect_lesion_masks(r, img.shape[:2])
    vis = annotate(img, r, show_masks=True, masks=masks)
    assert vis.shape == img.shape
    assert not np.array_equal(vis, img)

    p_img = tmp_path / "vis.jpg"
    cv2.imwrite(str(p_img), vis)
    hp = make_html("fake.jpg", p_img, r, risk, tmp_path / "r.html")
    pp = make_pdf("fake.jpg", p_img, r, risk, tmp_path / "r.pdf")
    assert hp.exists() and hp.stat().st_size > 5000
    assert pp.exists() and pp.stat().st_size > 8000
    html_text = hp.read_text(encoding="utf-8")
    assert "牙菌斑" in html_text and "干预建议" in html_text
    assert "综合评分" not in html_text and "风险" not in html_text


# ------------------------------------------------------------- pipeline e2e
@pytest.mark.skipif(not (DEMO_IMG.exists() and MODELS_OK),
                    reason="demo image or ONNX models missing")
def test_pipeline_e2e_real_weights():
    from app.core.pipeline import DentalPipeline

    pipe = DentalPipeline(fast_mode=True)
    result = pipe.run(DEMO_IMG)
    assert result["summary"]["tooth_count"] >= 3
    from app.core.risk import assess

    risk = assess(result)
    assert risk["plaque"]["count"] <= result["summary"]["tooth_count"]
    assert all(it["advice"] for it in risk["items"])


@pytest.mark.skipif(not MODELS_OK, reason="ONNX models missing")
def test_backend_letterbox_and_nms():
    from app.core import backend as B

    img = np.random.randint(0, 255, (459, 612, 3), np.uint8)
    blob, r, px, py = B.letterbox(img, 640)
    assert blob.shape == (1, 3, 640, 608) or blob.shape[2] % 32 == 0
    assert blob.shape[3] % 32 == 0
    boxes = np.array([[10, 10, 50, 50], [12, 12, 52, 52], [100, 100, 150, 150]],
                     np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    keep = B.nms_xyxy(boxes, scores, 0.5)
    assert keep.tolist() == [0, 2]


def test_quality_gate():
    from app.core.quality import check_quality

    small = np.full((100, 100, 3), 128, np.uint8)
    assert check_quality(small, {"min_size": 320})["usable"] is False
    rng = np.random.default_rng(0)
    textured = (rng.integers(0, 255, (480, 640, 3))).astype(np.uint8)
    q = check_quality(textured, {})
    assert q["blur_score"] > 100 and q["usable"] is True


# ---------------------------------------------------------------- GUI smoke
def _qt_env():
    import os
    import sys

    if not (sys.platform in ("win32", "darwin")
            or os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        # No display and no offscreen platform yet: Qt would block forever.
        pytest.skip("no display — set QT_QPA_PLATFORM=offscreen to run GUI tests")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_gui_main_window_smoke():
    _qt_env()
    pytest.importorskip("PyQt5")
    from app.qt_compat import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from app.gui.main_window import MainWindow

    w = MainWindow()
    assert w.windowTitle()
    assert not w.btn_start.isEnabled()
    w.load_image(str(DEMO_IMG))
    assert w.btn_start.isEnabled()


def test_gui_demo_dialog_smoke():
    _qt_env()
    pytest.importorskip("PyQt5")
    from app.qt_compat import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from app.gui.main_window import DEMO_CASES, DemoDialog

    d = DemoDialog()
    assert d.case_list.count() == 4
    assert all(Path(p).exists() for _, p in DEMO_CASES)
