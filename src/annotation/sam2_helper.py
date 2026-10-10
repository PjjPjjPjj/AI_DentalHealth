"""SAM2-assisted annotation (plan 21).

STRICT scope: SAM2 proposes geometry only (mask candidates). It NEVER decides
medical class. Class assignment comes from: existing expert labels, human
review, or confirmed lesion-model candidates.

sam2_helper.py is import-safe without SAM2 installed: check_sam2_available().

Outputs to outputs/annotation_queue/<image_stem>/:
  queue.json   - candidates with box/points + geometry
  proposal_*.png / overlay image
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
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

from src.utils.paths import ANNOTATION_QUEUE_DIR


def check_sam2_available() -> bool:
    try:
        import sam2  # noqa: F401
        return True
    except ImportError:
        return False


def sam2_predictor_bbox(model, bbox: list[float]):
    """Predict mask candidates for a bbox prompt."""
    from sam2.automatic_mask_generator import Sam2AutomaticMaskGenerator, Sam2Predictor  # noqa: F401
    import numpy as np  # noqa: F401


class SAM2Helper:
    """Thin wrapper. sam2 package must be installed separately for runtime use."""

    def __init__(self, model_cfg: str = "configs/sam2/sam2.1_hiera_b+.yaml",
                 checkpoint: str = "checkpoints/sam2.1_hiera_b+.pt"):
        if not check_sam2_available():
            raise RuntimeError("sam2 not installed. pip install sam2 (per Meta AI repo) first.")
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

    def propose_mask_from_bbox(self, image: np.ndarray, bbox: list[float]) -> np.ndarray:
        """Single candidate mask (geometry only)."""
        self.predictor.set_image(image)
        out_masks, scores, _ = self.predictor.predict(box=np.asarray([[bbox[0], bbox[1], bbox[2], bbox[3]]]))
        if len(out_masks) == 0:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        best = out_masks[int(np.argmax(scores))]
        return (best.astype(np.uint8) * 255).astype(np.uint8)


def prepare_review_entries(image: np.ndarray, candidates: list[dict], image_id: str) -> Path:
    """Write queue.json + proposal images for human review.

    candidates: [{"bbox": [x1,y1,x2,y2], "class_suggestion": "tooth"|None, "source": "detector"|"lesion_model", "conf": 0.9}]
    class_suggestion MUST come from a medical-class source (existing labels or
    lesion model + confirm), never SAM2.
    """
    out = ANNOTATION_QUEUE_DIR / image_id
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    overlay = image.copy()
    for i, cand in enumerate(candidates):
        entry = {
            "candidate_id": i,
            "bbox": [float(v) for v in cand["bbox"]],
            "class_suggestion": cand.get("class_suggestion"),
            "class_source_approved": cand.get("class_suggestion") is not None,
            "confidence": cand.get("conf"),
            "approved_by": None,
            "medical_class_from": "sam2? NO - sam2 returns geometry only",
            "status": "pending_review",
        }
        entries.append(entry)
        x1, y1, x2, y2 = [int(v) for v in cand["bbox"]]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    (out / "queue.json").write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    cv2.imwrite(str(out / "proposals_overlay.png"), overlay)
    return out
