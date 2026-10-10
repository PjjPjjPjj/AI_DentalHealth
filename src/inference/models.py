"""Model wrappers: thin adapters over Ultralytics YOLO models.

Each wrapper loads weights lazily and exposes a stable dict interface so the
pipeline can swap backends without touching its internals.
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

CALCULUS_CLS = 0
CARIES_CLS = 0
TOOTH_CLS = 0

_MISSING = object()


def _check_weights(weights: str | Path) -> None:
    if weights is None or not Path(weights).exists():
        raise FileNotFoundError(f"Model weights not found: {weights}")


def _safe_prob(res) -> float:
    """Torch tensor -> float."""
    try:
        return float(res)
    except Exception:
        return 0.0


def _morph_close(mask, k: int):
    """Binary morphological close (elliptical kernel), used by the calculus
    stage1-mask fallback (v5-fix7/FB)."""
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx((mask > 0).astype(np.uint8), cv2.MORPH_CLOSE, ker)


class ToothModel:
    """Tooth instance segmentation (YOLO11m-seg, IOP-Compass, single class)."""

    def __init__(self, weights: str | Path | None = None, conf: float = 0.25,
                 iou: float = 0.5, imgsz: int = 896, max_det: int = 60):
        self.weights = weights
        self.conf, self.iou = conf, iou
        self.imgsz, self.max_det = imgsz, max_det
        self._model = None
        if weights:
            _check_weights(weights)
            self._load()

    def _load(self) -> None:
        from ultralytics import YOLO
        self._model = YOLO(str(self.weights))

    def predict(self, img) -> list[dict]:
        if self._model is None:
            raise RuntimeError("No tooth weights loaded")
        res = self._model.predict(img, conf=self.conf, iou=self.iou,
                                  imgsz=self.imgsz, device=0, retina_masks=True,
                                  verbose=False, max_det=self.max_det)[0]
        out = []
        if res.masks is None:
            return out
        for i in range(len(res.boxes)):
            m = res.masks.data[i].cpu().numpy().astype(np.uint8)
            m = cv2.resize(m, (int(res.orig_shape[1]), int(res.orig_shape[0])), interpolation=cv2.INTER_NEAREST)
            b = res.boxes.xyxy[i].cpu().numpy().tolist()
            out.append({
                "mask": m, "bbox": [float(v) for v in b],
                "conf": float(res.boxes.conf[i]),
                "cls": int(res.boxes.cls[i]),
            })
        return out


class CalculusModel:
    """YOLO11s-seg calculus lesion segmentation."""

    def __init__(self, weights: str | Path | None = None, conf: float = 0.25, iou: float = 0.5):
        self.weights = weights
        self.conf, self.iou = conf, iou
        self._model = None
        if weights:
            _check_weights(weights)
            self._load()

    def _load(self) -> None:
        from ultralytics import YOLO
        self._model = YOLO(str(self.weights))

    def predict(self, img) -> list[dict]:
        if self._model is None:
            raise RuntimeError("No calculus weights loaded")
        res = self._model.predict(img, conf=self.conf, iou=self.iou, verbose=False, max_det=100)[0]
        out = []
        if res.masks is None:
            return out
        for i in range(len(res.boxes)):
            m = res.masks.data[i].cpu().numpy().astype(np.uint8)
            m = cv2.resize(m, (int(res.orig_shape[1]), int(res.orig_shape[0])), interpolation=cv2.INTER_NEAREST)
            out.append({"mask": m, "conf": float(res.boxes.conf[i]), "cls": int(res.boxes.cls[i])})
        return out


class CariesModel:
    """Caries object detection (bbox, YOLO11m @960 for C2)."""

    def __init__(self, weights: str | Path | None = None, conf: float = 0.25,
                 iou: float = 0.6, imgsz: int = 960, max_det: int = 100):
        self.weights = weights
        self.conf, self.iou = conf, iou
        self.imgsz, self.max_det = imgsz, max_det
        self._model = None
        if weights:
            _check_weights(weights)
            self._load()

    def _load(self) -> None:
        from ultralytics import YOLO
        self._model = YOLO(str(self.weights))

    def predict(self, img) -> list[dict]:
        if self._model is None:
            raise RuntimeError("No caries weights loaded")
        res = self._model.predict(img, conf=self.conf, iou=self.iou,
                                  imgsz=self.imgsz, device=0,
                                  verbose=False, max_det=self.max_det)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        b = res.boxes.xyxy.cpu().numpy()
        c = res.boxes.conf.cpu().numpy()
        return [{"bbox": [float(v) for v in b[i]], "conf": float(c[i]), "cls": int(res.boxes.cls[i])}
                for i in range(len(b))]


class PlaqueModel:
    """Tooth-crop binary plaque classifier (YOLO11s-cls, 0=normal, 1=plaque).

    predict_proba returns P(plaque) for the full crop; predict_proba_windows
    takes the max over configured fractional windows (localization tolerance
    for gingival-margin plaque).

    calibration: optional Platt mapping fitted on Mendeley extended val +
    SANO val negatives (npz with key "platt" = [a, b] or a raw (a, b) tuple).
    When set, returned probabilities are calibrated:
        P' = sigmoid(a * logit(P) + b)
    so the score is a domain-robust probability and the operating threshold
    transfers across imaging domains. Raw scores remain available via
    predict_proba(..., calibrated=False) / _proba_array_raw.
    """

    def __init__(self, weights: str | Path | None = None, conf_threshold: float = 0.5,
                 imgsz: int = 224, windows: list | None = None,
                 calibration: str | Path | tuple | None = None):
        self.weights = weights
        self.threshold = conf_threshold
        self.imgsz = imgsz
        self.windows = windows or [[0.0, 0.0, 1.0, 1.0]]
        self.calibration = calibration
        self._calib = None
        if isinstance(calibration, (str, Path)):
            import numpy as np
            z = np.load(calibration)
            self._calib = tuple(float(v) for v in z["platt"])
        elif isinstance(calibration, (tuple, list)) and len(calibration) == 2:
            self._calib = (float(calibration[0]), float(calibration[1]))
        self._model = None
        if weights:
            _check_weights(weights)
            self._load()

    def _load(self) -> None:
        from ultralytics import YOLO
        self._model = YOLO(str(self.weights))

    @staticmethod
    def _apply_calib(probs: list[float], calib) -> list[float]:
        import math
        if not calib:
            return probs
        a, b = calib
        out = []
        for p in probs:
            p = min(max(p, 1e-6), 1 - 1e-6)
            lg = math.log(p / (1 - p))
            out.append(1.0 / (1.0 + math.exp(-(a * lg + b))))
        return out

    def _proba_array_raw(self, crops: list) -> list[float]:
        if self._model is None:
            raise RuntimeError("No plaque weights loaded")
        res = self._model.predict(crops, imgsz=self.imgsz, device=0, verbose=False)
        out = []
        for r in res:
            probs = r.probs.data.cpu().numpy()
            out.append(float(probs[1]) if probs.shape[0] >= 2 else float(probs[0]))
        return out

    def _proba_array(self, crops: list) -> list[float]:
        return self._apply_calib(self._proba_array_raw(crops), self._calib)

    def predict_proba(self, img) -> float:
        """Return P(plaque) for a crop (BGR np array or path)."""
        return self._proba_array([img])[0]

    def predict_proba_windows(self, img) -> float:
        """Max P(plaque) over configured fractional windows of the crop."""
        h, w = img.shape[:2]
        crops = []
        for fx0, fy0, fx1, fy1 in self.windows:
            x0, y0 = int(fx0 * w), int(fy0 * h)
            x1, y1 = max(x0 + 8, int(fx1 * w)), max(y0 + 8, int(fy1 * h))
            x1, y1 = min(w, x1), min(h, y1)
            c = img[y0:y1, x0:x1]
            if c.size:
                crops.append(c)
        if not crops:
            crops = [img]
        return max(self._proba_array(crops))


class CalculusCascadeModel:
    """V5 two-stage calculus cascade packaged for the unified pipeline.

    stage1 (tiled seg, c1=0.005) -> stage2 refine (adaptive window) ->
    verifier (EffNet 4ch) -> M3 logistic fusion -> size-conditioned
    thresholds + conditional flip-TTA union + image-level rescue.

    Weights/params all come from configs/pipeline.yaml (inference.calculus +
    weights.calculus). Instance selection happens inside; returns only the
    selected instances in original-image coordinates.
    """

    def __init__(self, s1: str, s2: str, artifacts: str,
                 thr_std: float = 0.6, thr_large: float = 0.45,
                 large_edge: float = 0.01565, tau: float = 0.4,
                 thr_rescue: float = 0.25, s1_conf_min: float = 0.02,
                 c1: float = 0.005, fb: bool = True, fb_min_px: int = 30,
                 close_k: int = 5, device: str = "0"):
        import sys
        root = Path(__file__).resolve().parents[2]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import scripts.v3_pipeline_lib as _lib
        _lib.C1 = c1
        _lib.S2_ADAPTIVE = True
        _lib.S2_TTA = True
        _lib.OUT = Path(artifacts)
        self._lib = _lib
        self.thr_std, self.thr_large = thr_std, thr_large
        self.large_edge, self.tau = large_edge, tau
        self.thr_rescue, self.s1_conf_min = thr_rescue, s1_conf_min
        self.fb, self.fb_min_px, self.close_k = fb, fb_min_px, close_k
        self.device = device if str(device).startswith("cuda") else f"cuda:{device}"
        from ultralytics import YOLO
        self.s1m = YOLO(str(s1))
        self.s2m = YOLO(str(s2))
        self.vb = _lib.make_verifier()
        try:
            self.clf = _lib.make_imgclf()
        except Exception:
            self.clf = None
        import numpy as np
        fz = np.load(_lib.OUT / "fusion_v3.npz")
        self.fz = fz

    def predict(self, img) -> list[dict]:
        """img: image path (str/Path) or BGR ndarray. Returns selected
        calculus instances: {mask(H,W) uint8, conf (fused), s1_conf,
        area_ratio}."""
        import tempfile

        import cv2
        import numpy as np
        from sklearn.linear_model import LogisticRegression

        if isinstance(img, (str, Path)):
            path = str(img)
            image = cv2.imread(path)
        else:
            image = img
            fd, path = tempfile.mkstemp(suffix=".jpg", dir="/tmp/opencode")
            cv2.imwrite(path, image)
        if image is None:
            return []
        H, W = image.shape[:2]
        _lib = self._lib

        boxes, confs, masks = _lib.s1_pass(self.s1m, path)
        feats, bins, rects, crop_local = [], [], [], []
        for (b, sc1, msk) in zip(boxes, confs, masks):
            bx = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
            side = _lib.SIDE
            bw, bh = bx[2] - bx[0], bx[3] - bx[1]
            if max(bw, bh) >= _lib.S2_ADAPT_TRIG * _lib.SIDE:
                side = int(min(max(W, H), max(_lib.SIDE, _lib.S2_ADAPT_MULT * max(bw, bh))))
            x0 = int(round((bx[0] + bx[2]) / 2 - side / 2))
            y0 = int(round((bx[1] + bx[3]) / 2 - side / 2))
            x0 = max(0, min(x0, W - side)); y0 = max(0, min(y0, H - side))
            x1, y1 = min(x0 + side, W), min(y0 + side, H)
            crop_img = image[y0:y1, x0:x1]
            crop = cv2.resize(crop_img, (_lib.TARGET, _lib.TARGET),
                              interpolation=cv2.INTER_AREA)
            insts = _lib.s2_instances(self.s2m, crop)
            # conditional flip-TTA (fix4): only for relatively large blobs
            bw_, bh_ = bx[2] - bx[0], bx[3] - bx[1]
            saturated2 = (x1 - x0) * (y1 - y0) >= 0.9 * W * H
            rel2 = max(bw_, bh_) / max(x1 - x0, y1 - y0)
            do_tta = saturated2 or rel2 >= _lib.S2_TTA_TRIG
            if do_tta:
                flipped = _lib.s2_instances(self.s2m, crop[:, ::-1])
                for it in flipped:
                    it["bin640"] = np.ascontiguousarray(it["bin640"][:, ::-1])
                insts = insts + flipped
            sel = [{"x0": 0, "y0": 0, "cw": _lib.TARGET, "ch": _lib.TARGET,
                    "bin": i["bin640"], "s2_conf": i["s2_conf"],
                    "score": i["s2_conf"]} for i in insts]
            # v5-fix7/FB: stage1-mask fallback for LARGE boxes. The s1 retina
            # mask (never routed through the tiny s2 crop) is added as an extra
            # candidate so mask-off/near-miss large lesions can still be emitted;
            # mask-NMS dedups it where s2 already claimed the region. Scored by
            # the SAME verifier + M3 (FB-refit) so precision is still gated.
            if self.fb and float(msk.sum()) / float(H * W) >= self.large_edge:
                s1c640 = cv2.resize(_morph_close(msk[y0:y1, x0:x1], self.close_k),
                                    (_lib.TARGET, _lib.TARGET),
                                    interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                if int(s1c640.sum()) >= self.fb_min_px:
                    sel.append({"x0": 0, "y0": 0, "cw": _lib.TARGET, "ch": _lib.TARGET,
                                "bin": s1c640, "s2_conf": float(sc1),
                                "score": float(sc1)})
            keep = _lib.greedy_nms(sel, "mask_nms", _lib.NMS_IOU, _lib.TARGET, _lib.TARGET)
            if do_tta:
                keep = _lib.tta_fuse(keep, sel, 0.6)
            rectl = {"x0": x0, "y0": y0, "cw": x1 - x0, "ch": y1 - y0}
            for k in keep:
                P = k["bin"]
                m160 = (cv2.resize(P, (_lib.SIDE, _lib.SIDE),
                                   interpolation=cv2.INTER_NEAREST) > 0)
                s1m_ = (cv2.resize(msk[y0:y1, x0:x1], (_lib.SIDE, _lib.SIDE),
                                   interpolation=cv2.INTER_AREA) > 0.5)
                gx, gy, gw, gh = cv2.boundingRect(P.astype(np.uint8))
                feats.append({
                    "s1_conf": float(sc1), "s2_conf": float(k["s2_conf"]),
                    "s1_area_ratio": float(s1m_.sum()) / (_lib.SIDE * _lib.SIDE),
                    "s2_area_ratio": float(m160.sum()) / (_lib.SIDE * _lib.SIDE),
                    "s1s2_iou": float((s1m_ & m160).sum() / max((s1m_ | m160).sum(), 1)),
                    "bbox_area_ratio": (gw * gh) / float(_lib.TARGET * _lib.TARGET),
                    "bbox_aspect": float(min(gw, gh) / max(gw, gh)),
                    "verifier_prob": None})
                bins.append(cv2.resize(P.astype(np.uint8), (rectl["cw"], rectl["ch"]),
                                       interpolation=cv2.INTER_NEAREST))
                rects.append(rectl)
                crop_local.append({"crop_img": crop_img, "mask160": m160})
        if not feats:
            return []
        if self.vb is not None:
            vp = _lib.verifier_probs(self.vb, crop_local)
            for i in range(len(feats)):
                feats[i]["verifier_prob"] = float(vp[i])
        m3 = LogisticRegression(max_iter=1000, class_weight="balanced")
        m3.coef_, m3.intercept_, m3.classes_ = self.fz["coef"], self.fz["intercept"], np.array([0, 1])
        X = np.array([[f[c] for c in _lib.FCOLS] for f in feats], float)
        fused = m3.predict_proba((X - self.fz["mu"]) / self.fz["sg"])[:, 1]
        s1c = np.array([f["s1_conf"] for f in feats])
        ar = np.array([f["s2_area_ratio"] for f in feats])
        thr = np.where(ar >= self.large_edge, self.thr_large, self.thr_std)
        ok = (s1c >= self.s1_conf_min) & (fused >= thr)
        if not ok.any() and self.clf is not None:
            # rescue gate: imgclf EffNet @448 with hflip TTA (matches stream B)
            import torch
            from torchvision.transforms import functional as TF
            probs = []
            for im in (image, np.ascontiguousarray(image[:, ::-1])):
                rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                t = TF.resize(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255,
                              (448, 448), TF.InterpolationMode.BILINEAR)
                t = TF.normalize(t, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                with torch.no_grad():
                    probs.append(float(self.clf(t[None].to(self.device)).sigmoid()))
            if float(np.mean(probs)) >= self.tau:
                ok = fused >= self.thr_rescue
        recs = []
        for i in np.argwhere(ok).ravel():
            r = rects[i]
            recs.append({"x0": r["x0"], "y0": r["y0"], "cw": r["cw"], "ch": r["ch"],
                         "bin": bins[i], "score": feats[i]["s2_conf"],
                         "fused": float(fused[i]), "s1_conf": float(s1c[i]),
                         "area_ratio": float(ar[i])})
        if not recs:
            return []
        kept = _lib.greedy_nms(recs, "mask_nms", _lib.NMS_IOU, H, W)
        out = []
        for k in kept:
            out.append({"mask": _lib.rec_bin(k, H, W), "conf": k["fused"],
                        "s1_conf": k["s1_conf"], "area_ratio": k["area_ratio"],
                        "cls": 0})
        return out
