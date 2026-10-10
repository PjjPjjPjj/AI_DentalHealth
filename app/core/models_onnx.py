"""ONNX model wrappers for the four production models + calculus cascade.

Faithful re-implementation of src/inference/models.py + scripts/v3_pipeline_lib
on onnxruntime (CPU/CUDA). Numbers match the promoted v5-fix7 pipeline.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.core import backend as B

SIDE = 160
TARGET = 640
NMS_IOU = 0.5
WIDE_MIN = 500
TILE = 320
S2_ADAPT_TRIG = 0.55
S2_ADAPT_MULT = 2.4
S2_TTA_TRIG = 0.35
FCOLS = ["s1_conf", "s2_conf", "verifier_prob", "s1_area_ratio",
         "s2_area_ratio", "s1s2_iou", "bbox_area_ratio", "bbox_aspect"]


def _morph_close(mask: np.ndarray, k: int) -> np.ndarray:
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx((mask > 0).astype(np.uint8), cv2.MORPH_CLOSE, ker)


def _decode(out0: np.ndarray, nc: int, conf: float, iou: float,
            max_det: int, keep_classes: list | None = None):
    """Shared det decode. Returns (boxes_lb, cls_ids, confs, anchor_idx)."""
    pred = out0.T  # (N, 4+nc+extra)
    scores = pred[:, 4:4 + nc]
    cls_ids = scores.argmax(1)
    confs = scores.max(1)
    m = confs > conf
    if keep_classes is not None:
        m &= np.isin(cls_ids, keep_classes)
    if not m.any():
        z = np.zeros(0, np.float32)
        return z.reshape(0, 4), z.astype(np.int64), z, np.zeros(0, np.int64)
    idx = np.nonzero(m)[0]
    pred, cls_ids, confs = pred[idx], cls_ids[idx], confs[idx]
    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    keep = B.nms_xyxy(boxes, confs, iou)[:max_det]
    return (boxes[keep].astype(np.float32), cls_ids[keep].astype(np.int64),
            confs[keep].astype(np.float32), idx[keep])


class YoloSegOnnx:
    """Retina-masks instance segmentation (ultralytics-compatible)."""

    def __init__(self, weights: str | Path, nc: int, conf: float, iou: float,
                 imgsz: int, max_det: int, keep_classes: list | None = None,
                 threads: int | None = None):
        self.sess = B.OnnxSession(weights, threads)
        self.nc, self.conf, self.iou = nc, conf, iou
        self.imgsz, self.max_det = imgsz, max_det
        self.keep = keep_classes

    def _forward(self, img: np.ndarray):
        h0, w0 = img.shape[:2]
        blob, r, px, py = B.letterbox(img, self.imgsz)
        out0, proto = self.sess.run(blob)
        out0 = np.asarray(out0)[0]
        proto = np.asarray(proto)[0].astype(np.float32)
        boxes_lb, cls_ids, confs, aidx = _decode(
            out0, self.nc, self.conf, self.iou, self.max_det, self.keep)
        if len(boxes_lb) == 0:
            return [], None
        boxes_orig = B.scale_boxes(boxes_lb, r, px, py, (h0, w0))
        coeffs = out0[4 + self.nc: 4 + self.nc + 32, aidx].T.astype(np.float32)
        masks = B.process_mask_native(coeffs, proto, boxes_orig, (h0, w0))
        out = []
        for i in range(len(boxes_lb)):
            m = masks[i]
            if int(m.sum()) == 0:
                continue
            out.append({"mask": m,
                        "bbox": [float(v) for v in boxes_orig[i]],
                        "conf": float(confs[i]),
                        "cls": int(cls_ids[i])})
        return out, None

    def predict(self, img: np.ndarray) -> list[dict]:
        out, _ = self._forward(img)
        return out


class YoloDetOnnx:
    """Object detection (bbox)."""

    def __init__(self, weights: str | Path, nc: int, conf: float, iou: float,
                 imgsz: int, max_det: int, keep_classes: list | None = None,
                 threads: int | None = None):
        self.sess = B.OnnxSession(weights, threads)
        self.nc, self.conf, self.iou = nc, conf, iou
        self.imgsz, self.max_det = imgsz, max_det
        self.keep = keep_classes

    def predict(self, img: np.ndarray) -> list[dict]:
        h0, w0 = img.shape[:2]
        blob, r, px, py = B.letterbox(img, self.imgsz)
        (out0,) = self.sess.run(blob)
        boxes_lb, cls_ids, confs, _ = _decode(
            np.asarray(out0)[0], self.nc, self.conf, self.iou, self.max_det,
            self.keep)
        boxes = B.scale_boxes(boxes_lb, r, px, py, (h0, w0))
        return [{"bbox": [float(v) for v in boxes[i]],
                 "conf": float(confs[i]),
                 "cls": int(cls_ids[i])} for i in range(len(boxes))]


class YoloClsOnnx:
    """Binary tooth-crop classifier -> P(class index 1).

    Optional Platt calibration (a, b): P' = sigmoid(a*logit(P)+b), fitted on
    Mendeley ext val + SANO val negatives. Makes scores cross-domain comparable
    so the box-level threshold (0.746) and image-level threshold (0.523) transfer
    across imaging domains.
    """

    def __init__(self, weights: str | Path, imgsz: int = 224,
                 threads: int | None = None, calibration=None):
        self.sess = B.OnnxSession(weights, threads)
        self.imgsz = imgsz
        self._calib = None
        if isinstance(calibration, (tuple, list)) and len(calibration) == 2:
            self._calib = (float(calibration[0]), float(calibration[1]))

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

    def predict_proba_batch(self, crops: list[np.ndarray]) -> list[float]:
        out = []
        for crop in crops:
            blob = B.cls_blob_pil(crop, self.imgsz)
            (probs,) = self.sess.run(blob)
            p = np.asarray(probs)[0].astype(np.float64)
            if p.max() > 1.0 + 1e-6 or abs(p.sum() - 1.0) > 1e-3:
                p = B.sigmoid(p - p.max())
                p = np.exp(p) / np.exp(p).sum()
            out.append(float(p[1]) if p.shape[0] >= 2 else float(p[0]))
        return self._apply_calib(out, self._calib)


class ToothModelOnnx(YoloSegOnnx):
    def __init__(self, weights: str | Path, conf: float = 0.30, iou: float = 0.70,
                 imgsz: int = 896, max_det: int = 60, threads: int | None = None):
        super().__init__(weights, nc=1, conf=conf, iou=iou, imgsz=imgsz,
                         max_det=max_det, threads=threads)


class CariesModelOnnx(YoloDetOnnx):
    def __init__(self, weights: str | Path, conf: float = 0.55, iou: float = 0.60,
                 imgsz: int = 960, max_det: int = 100, threads: int | None = None):
        super().__init__(weights, nc=1, conf=conf, iou=iou, imgsz=imgsz,
                         max_det=max_det, threads=threads)


class PlaqueModelOnnx(YoloClsOnnx):
    def __init__(self, weights: str | Path, imgsz: int = 224,
                 threads: int | None = None, calibration=None):
        super().__init__(weights, imgsz, threads, calibration)


# ---------------- calculus two-stage cascade (v5-fix7/FB) ----------------

def _tile_grid(W: int, H: int) -> list:
    side = min(TILE, W, H)
    xs = sorted({0, max(0, W - side)}) if W > side else [0]
    ys = sorted({0, max(0, H - side)}) if H > side else [0]
    return [(x, y, min(x + side, W), min(y + side, H)) for y in ys for x in xs]


def _nms_xyxy_idx(boxes: np.ndarray, confs: np.ndarray, thr: float = 0.5):
    if len(boxes) == 0:
        return np.zeros(0, int)
    order = np.argsort(-confs)
    keep = []
    for i in order:
        b = boxes[i]
        ok = True
        for j in keep:
            bj = boxes[j]
            ix0, iy0 = max(b[0], bj[0]), max(b[1], bj[1])
            ix1, iy1 = min(b[2], bj[2]), min(b[3], bj[3])
            inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            ua = ((b[2] - b[0]) * (b[3] - b[1])
                  + (bj[2] - bj[0]) * (bj[3] - bj[1]) - inter)
            if ua > 0 and inter / ua >= thr:
                ok = False
                break
        keep.append(i)
    return np.array(sorted(keep), int)


def _greedy_mask_nms(cands: list[dict], H: int, W: int, iou_thr: float = NMS_IOU):
    order = sorted(range(len(cands)), key=lambda i: cands[i].get("score", 0.0),
                   reverse=True)
    accepted = []
    for idx in order:
        c = cands[idx]
        cm = _rec_bin(c, H, W)
        keep = True
        for a in accepted:
            am = _rec_bin(a, H, W)
            inter = int(((cm > 0) & (am > 0)).sum())
            uni = int(((cm > 0) | (am > 0)).sum())
            if uni and inter / uni >= iou_thr:
                keep = False
                break
        if keep:
            accepted.append(c)
    return accepted


def _rec_bin(rec: dict, H: int, W: int) -> np.ndarray:
    buf = np.zeros((H, W), np.uint8)
    b = rec.get("bin")
    if b is not None:
        buf[rec["y0"]:rec["y0"] + rec["ch"], rec["x0"]:rec["x0"] + rec["cw"]] = b
    return buf


def _tta_fuse(keep: list[dict], sel: list[dict], iou_thr: float = 0.6):
    if not keep:
        return keep
    kept_ids = {id(k) for k in keep}
    for k in keep:
        for s in sel:
            if id(s) in kept_ids:
                continue
            inter = int((k["bin"] & s["bin"]).sum())
            uni = int((k["bin"] | s["bin"]).sum())
            if uni and inter / uni >= iou_thr:
                k["bin"] = ((k["bin"] | s["bin"]) > 0).astype(np.uint8)
    return keep


class CalculusCascadeOnnx:
    """S1 tiled seg -> adaptive S2 refine (+flip TTA, FB fallback) ->
    verifier -> M3 logistic fusion -> size-conditioned thresholds ->
    image-level rescue (imgclf). Mirrors CalculusCascadeModel.predict."""

    def __init__(self, s1: str | Path, s2: str | Path, verifier: str | Path,
                 imgclf: str | Path | None, fusion_npz: str | Path,
                 c1: float = 0.005, s2_conf: float = 0.05,
                 thr_std: float = 0.60, thr_large: float = 0.45,
                 large_edge: float = 0.01565, tau: float = 0.40,
                 thr_rescue: float = 0.25, s1_conf_min: float = 0.02,
                 fb: bool = True, fb_min_px: int = 30, close_k: int = 5,
                 threads: int | None = None, fast_mode: bool = False):
        self.s1 = YoloSegOnnx(s1, nc=2, conf=c1, iou=0.6, imgsz=TARGET,
                              max_det=150, keep_classes=[0], threads=threads)
        self.s2 = YoloSegOnnx(s2, nc=2, conf=s2_conf, iou=0.6, imgsz=TARGET,
                              max_det=100, keep_classes=[0], threads=threads)
        self.vb = B.OnnxSession(verifier, threads)
        self.clf = B.OnnxSession(imgclf, threads) if imgclf else None
        fz = np.load(fusion_npz)
        self.coef = fz["coef"].astype(np.float64).ravel()
        self.intercept = float(np.asarray(fz["intercept"]).ravel()[0])
        self.mu = fz["mu"].astype(np.float64).ravel()
        self.sg = fz["sg"].astype(np.float64).ravel()
        self.thr_std, self.thr_large, self.large_edge = thr_std, thr_large, large_edge
        self.tau, self.thr_rescue, self.s1_conf_min = tau, thr_rescue, s1_conf_min
        self.fb, self.fb_min_px, self.close_k = fb, fb_min_px, close_k
        # fast_mode: skips stage-2 flip-TTA and stage-1 tiling (~2x faster on
        # CPU, slightly lower recall on large lesions); standard mode keeps
        # bit-equivalence with the production torch pipeline.
        self.fast_mode = fast_mode

    # -- stage 1 -------------------------------------------------------
    def _s1_pass(self, img: np.ndarray):
        H, W = img.shape[:2]
        if self.fast_mode or max(W, H) < WIDE_MIN:
            return self.s1.predict(img)

        all_b, all_c, all_m, all_meta = [], [], [], []
        for (x0, y0, x1, y1) in _tile_grid(W, H):
            insts = self.s1.predict(img[y0:y1, x0:x1])
            if not insts:
                continue
            for it in insts:
                b = np.array(it["bbox"], np.float32)
                b[[0, 2]] += x0
                b[[1, 3]] += y0
                all_b.append(b)
                all_c.append(it["conf"])
                all_m.append(it["mask"])
                all_meta.append((x0, y0, x1 - x0, y1 - y0))
        if not all_b:
            return []
        boxes = np.stack(all_b, 0)
        confs = np.array(all_c, np.float32)
        keep = _nms_xyxy_idx(boxes, confs, 0.5)
        out = []
        for i in keep:
            tx, ty, tw, th = all_meta[i]
            m = cv2.resize(all_m[i], (tw, th), interpolation=cv2.INTER_NEAREST)
            canvas = np.zeros((H, W), np.uint8)
            canvas[ty:ty + th, tx:tx + tw] = m
            out.append({"mask": canvas, "bbox": boxes[i].tolist(),
                        "conf": float(confs[i])})
        return out

    # -- stage 2 helpers -------------------------------------------------
    def _s2_instances(self, crop: np.ndarray) -> list[dict]:
        out = []
        for it in self.s2.predict(crop):
            mi = it["mask"]
            if int(mi.sum()) == 0:
                continue
            out.append({"bin640": mi, "s2_conf": it["conf"]})
        return out

    def predict(self, img: np.ndarray) -> list[dict]:
        H, W = img.shape[:2]
        boxes, confs, masks = [], [], []
        for it in self._s1_pass(img):
            boxes.append(np.array(it["bbox"], np.float32))
            confs.append(it["conf"])
            masks.append(it["mask"])
        # Group candidates: only s1_conf >= s1_conf_min can pass the fusion
        # gate on the normal path; the rest matter ONLY when the image-level
        # rescue fires (ok empty + imgclf >= tau). Processing group A first
        # keeps outputs bit-equivalent to the torch pipeline while skipping
        # most s2 calls on CPU.
        idx_a = [i for i, c in enumerate(confs) if c >= self.s1_conf_min]
        idx_b = [i for i, c in enumerate(confs) if c < self.s1_conf_min]
        state = {"rescue": False}
        recs_a, feats_a = self._process_group(img, idx_a, boxes, confs, masks, state)
        ok_a = self._gate(recs_a, feats_a, state)
        recs = list(recs_a)
        if len(ok_a) and ok_a.any():
            sel = [recs_a[i] for i in np.nonzero(ok_a)[0]]
        else:
            if not self._rescue_ok(img):
                return []
            recs_b, feats_b = self._process_group(img, idx_b, boxes, confs, masks, state)
            recs += recs_b
            all_recs = recs_a + recs_b
            all_feats = feats_a + feats_b
            fused = np.array([r["fused"] for r in all_recs])
            keep = fused >= self.thr_rescue
            sel = [all_recs[i] for i in np.nonzero(keep)[0]]
        if not sel:
            return []
        kept = _greedy_mask_nms(sel, H, W)
        return [{"mask": _rec_bin(k, H, W), "conf": k["fused"],
                 "s1_conf": k["s1_conf"], "area_ratio": k["area_ratio"], "cls": 0}
                for k in kept]

    def _rescue_ok(self, img: np.ndarray) -> bool:
        if self.clf is None:
            return False
        probs = []
        for im in (img, np.ascontiguousarray(img[:, ::-1])):
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            (lg,) = self.clf.run(B.effnet_blob(rgb, (448, 448), normalize=True))
            probs.append(float(B.sigmoid(np.asarray(lg).reshape(-1))[0]))
        return float(np.mean(probs)) >= self.tau

    def _gate(self, recs: list, feats: list, state: dict):
        if not recs:
            return np.zeros(0, bool)
        fused = np.array([r["fused"] for r in recs])
        if state["rescue"]:
            return fused >= self.thr_rescue
        s1c = np.array([r["s1_conf"] for r in recs])
        ar = np.array([r["area_ratio"] for r in recs])
        thr = np.where(ar >= self.large_edge, self.thr_large, self.thr_std)
        return (s1c >= self.s1_conf_min) & (fused >= thr)

    def _process_group(self, img, idx_list, boxes, confs, masks, state):
        H, W = img.shape[:2]
        feats, bins, rects, crop_local = [], [], [], []
        for i in idx_list:
            b, sc1, msk = boxes[i], confs[i], masks[i]
            bx = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
            side = SIDE
            bw, bh = bx[2] - bx[0], bx[3] - bx[1]
            if max(bw, bh) >= S2_ADAPT_TRIG * SIDE:
                side = int(min(max(W, H), max(SIDE, S2_ADAPT_MULT * max(bw, bh))))
            x0 = int(round((bx[0] + bx[2]) / 2 - side / 2))
            y0 = int(round((bx[1] + bx[3]) / 2 - side / 2))
            x0 = max(0, min(x0, W - side))
            y0 = max(0, min(y0, H - side))
            x1, y1 = min(x0 + side, W), min(y0 + side, H)
            crop_img = img[y0:y1, x0:x1]
            crop = cv2.resize(crop_img, (TARGET, TARGET), interpolation=cv2.INTER_AREA)
            insts = self._s2_instances(crop)
            bw_, bh_ = bx[2] - bx[0], bx[3] - bx[1]
            saturated2 = (x1 - x0) * (y1 - y0) >= 0.9 * W * H
            rel2 = max(bw_, bh_) / max(x1 - x0, y1 - y0)
            do_tta = (not self.fast_mode) and (saturated2 or rel2 >= S2_TTA_TRIG)
            if do_tta:
                flipped = self._s2_instances(crop[:, ::-1])
                for it in flipped:
                    it["bin640"] = np.ascontiguousarray(it["bin640"][:, ::-1])
                insts = insts + flipped
            sel = [{"x0": 0, "y0": 0, "cw": TARGET, "ch": TARGET,
                    "bin": it["bin640"], "s2_conf": it["s2_conf"],
                    "score": it["s2_conf"]} for it in insts]
            if self.fb and float(msk.sum()) / float(H * W) >= self.large_edge:
                s1c640 = cv2.resize(
                    _morph_close(msk[y0:y1, x0:x1], self.close_k),
                    (TARGET, TARGET), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                if int(s1c640.sum()) >= self.fb_min_px:
                    sel.append({"x0": 0, "y0": 0, "cw": TARGET, "ch": TARGET,
                                "bin": s1c640, "s2_conf": float(sc1),
                                "score": float(sc1)})
            keep = _greedy_mask_nms(sel, TARGET, TARGET)
            if do_tta:
                keep = _tta_fuse(keep, sel, 0.6)
            rectl = {"x0": x0, "y0": y0, "cw": x1 - x0, "ch": y1 - y0}
            for k in keep:
                P = k["bin"]
                m160 = (cv2.resize(P, (SIDE, SIDE),
                                   interpolation=cv2.INTER_NEAREST) > 0)
                s1m_ = (cv2.resize(msk[y0:y1, x0:x1], (SIDE, SIDE),
                                   interpolation=cv2.INTER_AREA) > 0.5)
                gx, gy, gw, gh = cv2.boundingRect(P.astype(np.uint8))
                feats.append({
                    "s1_conf": float(sc1), "s2_conf": float(k["s2_conf"]),
                    "s1_area_ratio": float(s1m_.sum()) / (SIDE * SIDE),
                    "s2_area_ratio": float(m160.sum()) / (SIDE * SIDE),
                    "s1s2_iou": float((s1m_ & m160).sum() / max((s1m_ | m160).sum(), 1)),
                    "bbox_area_ratio": (gw * gh) / float(TARGET * TARGET),
                    "bbox_aspect": float(min(gw, gh) / max(gw, gh)),
                    "verifier_prob": 0.0})
                bins.append(cv2.resize(P.astype(np.uint8), (rectl["cw"], rectl["ch"]),
                                       interpolation=cv2.INTER_NEAREST))
                rects.append(rectl)
                crop_local.append({"crop_img": crop_img, "mask160": m160})
        if not feats:
            return [], []
        vb_in = []
        for it in crop_local:
            rgb = cv2.cvtColor(it["crop_img"], cv2.COLOR_BGR2RGB)
            vb_in.append(B.effnet_blob(rgb, (224, 224), normalize=True,
                                       extra_mask=it["mask160"]))
        (logits,) = self.vb.run(np.concatenate(vb_in, 0))
        vp = B.sigmoid(np.asarray(logits).reshape(-1))
        for i in range(len(feats)):
            feats[i]["verifier_prob"] = float(vp[i])
        X = np.array([[f[c] for c in FCOLS] for f in feats], float)
        z = ((X - self.mu) / self.sg) @ self.coef + self.intercept
        fused = 1.0 / (1.0 + np.exp(-z))
        recs = []
        for j, (k, r) in enumerate(zip(bins, rects)):
            recs.append({"x0": r["x0"], "y0": r["y0"], "cw": r["cw"], "ch": r["ch"],
                         "bin": k, "score": feats[j]["s2_conf"],
                         "fused": float(fused[j]),
                         "s1_conf": float(feats[j]["s1_conf"]),
                         "area_ratio": float(feats[j]["s2_area_ratio"])})
        return recs, feats
