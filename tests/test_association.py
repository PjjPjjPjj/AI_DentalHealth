"""Lesion-tooth association tests (plan 27 cases)."""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
import numpy as np
import pytest

from src.inference.associate import (
    associate_calculus, associate_caries, compute_plaque_regions, _bbox_iou,
)


def _mask(h=100, w=100, y1=0, y2=100, x1=0, x2=100):
    m = np.zeros((h, w), dtype=np.uint8)
    m[y1:y2, x1:x2] = 1
    return m


def test_calculus_assign_to_max_overlap():
    tooth_a = _mask(y1=10, y2=60)
    tooth_b = _mask(y1=60, y2=100)
    lesion = tooth_b.copy()
    res = associate_calculus(lesion, [tooth_a, tooth_b], min_overlap=0.2)
    assert res["assigned"] is True
    assert res["tooth"] == 1
    assert res["overlap_ratio"] == pytest.approx(1.0)


def test_calculus_unassigned_below_threshold():
    tooth = _mask(y1=30, y2=80)
    lesion = _mask(y1=10, y2=30)  # small contact strip
    res = associate_calculus(lesion, [tooth], min_overlap=0.2)
    assert res["assigned"] is False or res["overlap_ratio"] < 0.2


def test_calculus_empty_lesion():
    res = associate_calculus(np.zeros((100, 100), dtype=np.uint8),
                             [_mask()], min_overlap=0.2)
    assert res is None


def test_caries_center_in_mask():
    tooth = _mask(y1=30, y2=60, x1=0, x2=100)  # center band
    res = associate_caries([40, 50, 60, 70], [tooth], center_rule=True)
    assert res["assigned"] is True
    assert res["tooth"] == 0


def test_caries_no_tooth_masks():
    res = associate_caries([10, 10, 20, 20], [], center_rule=True)
    assert res["assigned"] is False
    assert res["tooth"] is None


def test_caries_far_away_unassigned():
    tooth = _mask(y1=30, y2=60, x1=0, x2=100)
    res = associate_caries([0, 0, 10, 10], [tooth], center_rule=True, min_iou=0.01)
    assert res["assigned"] is False


def test_bbox_iou():
    m = _mask(y1=0, y2=100, x1=0, x2=100)
    iou = _bbox_iou([25, 25, 75, 75], m)
    assert iou > 0.0


def test_plaque_regions_reserved_none():
    regions = compute_plaque_regions(_mask())
    assert regions == {"gingival": None, "incisal": None, "mesial": None, "distal": None}


def test_mask_shape_mismatch_raises():
    small = np.ones((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError):
        associate_calculus(small, [_mask(h=20, w=20)], 0.2)
