"""Data-layer tests: audit, conversion guards, split integrity."""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
import json
from pathlib import Path

import numpy as np
import pytest

from src.data.audit import audit_label_file, audit_static_image, find_images
from src.data.split import assert_no_patient_leak, patient_split


def test_patient_split_no_leak():
    sample_ids = [f"img{i}" for i in range(30)]
    patient_ids = [f"p{i % 6}" for i in range(30)]
    tr, va, te = patient_split(sample_ids, patient_ids, ratios=(0.7, 0.15, 0.15), seed=42)
    overlap_tr_va = tr & va
    overlap_tr_te = tr & te
    overlap_va_te = va & te
    assert not overlap_tr_va and not overlap_tr_te and not overlap_va_te
    assert len(tr) + len(va) + len(te) == len(sample_ids)


def test_patient_split_disjoint_patients():
    """Patients (not just samples) must be disjoint."""
    sample_ids = [f"img{i}" for i in range(30)]
    patient_ids = [f"p{i % 6}" for i in range(30)]
    tr, va, te = patient_split(sample_ids, patient_ids, ratios=(0.7, 0.15, 0.15), seed=7)
    # reconstruct patient sets from a mapping
    p_of = dict(zip(sample_ids, patient_ids))
    tr_p = {p_of[s] for s in tr}
    va_p = {p_of[s] for s in va}
    te_p = {p_of[s] for s in te}
    assert not (tr_p & va_p)
    assert not (tr_p & te_p)
    assert not (va_p & te_p)


def test_audit_label_oob_detection(tmp_path):
    lbl = tmp_path / "a.txt"
    lbl.write_text("0 1.5 1.5 0.2 0.2")  # center outside
    rec = audit_label_file(lbl, 640, 480)
    assert rec["bbox_out_of_bounds"] >= 1


def test_audit_label_valid(tmp_path):
    lbl = tmp_path / "a.txt"
    lbl.write_text("0 0.5 0.5 0.1 0.1")
    rec = audit_label_file(lbl, 640, 480)
    assert rec["objects"] == 1
    assert rec["bbox_out_of_bounds"] == 0


def test_audit_label_empty(tmp_path):
    lbl = tmp_path / "a.txt"
    lbl.write_text("")
    rec = audit_label_file(lbl, 640, 480)
    assert rec["empty"] is True


def test_audit_unreadable_image():
    rec = audit_static_image(Path("/nonexistent/x.jpg"))
    assert rec["readable"] is False


def test_find_images(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.txt").write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.png").write_bytes(b"x")
    found = find_images(tmp_path)
    assert len(found) == 2
