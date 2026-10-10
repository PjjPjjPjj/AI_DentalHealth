"""ONNX Runtime backend: ultralytics-compatible pre/post processing.

Reproduces the exact behavior of Ultralytics 8.4 predict for the exported
models (letterbox -> inference -> conf filter -> NMS -> scale_boxes ->
process_mask_native for retina seg), so ONNX outputs match the production
torch pipeline. Verification: scripts/verify_onnx_consistency.py
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

try:
    from PIL import Image
    _HAS_PIL = True
except Exception:  # pragma: no cover
    _HAS_PIL = False

_IMGNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMGNET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _select_providers() -> list:
    avail = []
    try:
        import onnxruntime as ort
        avail = ort.get_available_providers()
    except Exception:
        return ["CPUExecutionProvider"]
    requested = os.environ.get("DENTALAI_EP", "auto").lower()
    if requested == "cpu":
        return ["CPUExecutionProvider"]
    if requested == "cuda" or (requested == "auto" and "CUDAExecutionProvider" in avail):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class OnnxSession:
    """Thin onnxruntime session wrapper (single static input)."""

    def __init__(self, path: str | Path, intra_threads: int | None = None):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = intra_threads or min(8, max(1, os.cpu_count() or 4))
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(path), so, providers=_select_providers())
        self.input_name = self.sess.get_inputs()[0].name

    def run(self, blob: np.ndarray) -> list:
        return self.sess.run(None, {self.input_name: blob})


def letterbox(img: np.ndarray, size: int, stride: int = 32):
    """Ultralytics LetterBox(auto=True): scale to fit `size`, pad only to the
    next stride multiple (minimum rectangle). Returns (blob, r, pad_x, pad_y)."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    dw, dh = size - new_w, size - new_h
    dw, dh = int(np.mod(dw, stride)), int(np.mod(dh, stride))
    left, right = dw // 2, dw - dw // 2
    top, bottom = dh // 2, dh - dh // 2
    if (new_w, new_h) != (w, h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    out = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                             value=(114, 114, 114))
    blob = np.ascontiguousarray(out[..., ::-1].transpose(2, 0, 1),
                                np.float32)[None] / 255.0
    return blob, r, float(left), float(top)


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Standard greedy NMS; returns kept indices sorted by score desc."""
    if len(boxes) == 0:
        return np.zeros(0, np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(-scores)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-9)
        order = order[1:][iou < iou_thr]
    return np.array(keep, np.int64)


def decode_det(out0: np.ndarray, nc: int, conf: float, iou: float,
               max_det: int) -> tuple[np.ndarray, np.ndarray]:
    """out0: (4+nc+extra, N) -> (boxes_xyxy_letterbox (M,4), cls (M,))."""
    pred = out0.T
    scores = pred[:, 4:4 + nc]
    cls_ids = scores.argmax(1)
    confs = scores.max(1)
    m = confs > conf
    if not m.any():
        return np.zeros((0, 4), np.float32), np.zeros(0, np.int64)
    pred = pred[m]
    cls_ids, confs = cls_ids[m], confs[m]
    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    keep = nms_xyxy(boxes, confs, iou)[:max_det]
    return boxes[keep].astype(np.float32), cls_ids[keep].astype(np.int64)


def scale_boxes(boxes_lb: np.ndarray, r: float, pad_x: float, pad_y: float,
                orig_hw: tuple) -> np.ndarray:
    boxes = boxes_lb.copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / r
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / r
    h, w = orig_hw
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    return boxes


def process_mask_native(coeffs: np.ndarray, proto: np.ndarray, boxes_orig: np.ndarray,
                        orig_hw: tuple, thresh: float = 0.0) -> np.ndarray:
    """Ultralytics ops.process_mask_native (retina): proto-space logits ->
    bilinear upsample to original size -> threshold -> crop to bbox.

    coeffs: (N, 32) mask coefficients; proto: (32, mh, mw);
    boxes_orig: (N, 4) xyxy in ORIGINAL image coords.
    Returns uint8 masks (N, H, W), cropped to their bbox (zero elsewhere).
    """
    h, w = orig_hw
    n = coeffs.shape[0]
    if n == 0:
        return np.zeros((0, h, w), np.uint8)
    mh, mw = proto.shape[1], proto.shape[2]
    logits = (coeffs @ proto.reshape(proto.shape[0], -1)).reshape(n, mh, mw)
    out = np.zeros((n, h, w), np.uint8)
    for i in range(n):
        m = cv2.resize(logits[i], (w, h), interpolation=cv2.INTER_LINEAR)
        binm = (m > thresh).astype(np.uint8)
        x1, y1, x2, y2 = boxes_orig[i]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 > x1 and y2 > y1:
            out[i, y1:y2, x1:x2] = binm[y1:y2, x1:x2]
    return out


def cls_blob_pil(img_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    """Ultralytics classify transforms (torchvision F_pil semantics):
    RGB, Resize(short side = size, PIL bilinear, long side int()-truncated),
    CenterCrop, /255, no normalization. -> (1,3,size,size)."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    if _HAS_PIL:
        if (w <= h and w == size) or (h <= w and h == size):
            arr = np.asarray(rgb, np.float32)
        else:
            if w < h:
                ow, oh = size, int(size * h / w)
            else:
                oh, ow = size, int(size * w / h)
            im = Image.fromarray(rgb).resize((ow, oh), Image.BILINEAR)
            arr = np.asarray(im, np.float32)
    else:
        if w < h:
            ow, oh = size, int(size * h / w)
        else:
            oh, ow = size, int(size * w / h)
        arr = cv2.resize(rgb, (ow, oh), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    top = max(0, (arr.shape[0] - size) // 2)
    left = max(0, (arr.shape[1] - size) // 2)
    crop = arr[top:top + size, left:left + size]
    if crop.shape[0] < size or crop.shape[1] < size:
        pad = np.zeros((size, size, 3), np.float32)
        pad[:crop.shape[0], :crop.shape[1]] = crop
        crop = pad
    blob = crop.transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(blob, np.float32)


def effnet_blob(img_rgb: np.ndarray, size: tuple, normalize: bool = True,
                extra_mask: np.ndarray | None = None) -> np.ndarray:
    """TorchVision-TF-style blob: direct bilinear resize (no antialias) to
    (size, size), /255, optional ImageNet normalize; optional 4th mask channel
    (nearest-resized 0/1, never normalized)."""
    h, w = img_rgb.shape[:2]
    if (w, h) != (size[1], size[0]):
        img = cv2.resize(img_rgb, size, interpolation=cv2.INTER_LINEAR)
    else:
        img = img_rgb
    x = img.astype(np.float32) / 255.0
    if normalize:
        x = (x - _IMGNET_MEAN) / _IMGNET_STD
    if extra_mask is not None:
        mm = cv2.resize(extra_mask.astype(np.float32), size,
                        interpolation=cv2.INTER_NEAREST)
        mm = (mm > 0.5).astype(np.float32)
        x = np.concatenate([x, mm[..., None]], 2)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
