"""Shared V3 two-stage pipeline helpers (group-honest split, 3-class models).

All detection/mask extraction filters to class 0 (calculus); class 1 is
other_lesion and must never become a calculus candidate.
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
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models as tvmodels
from torchvision.transforms import functional as TF

from src.inference.fusion import greedy_nms, rec_bin  # noqa: F401

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

DATA = Path("datasets/processed/calculus_seg_v3")
OUT = Path("outputs/calculus_v3_cls")
RAW = Path(f"{_ROOT}/datasets/raw/oral_mamba/lables")
C1 = 0.02
C2 = 0.05
SIDE = 160
TARGET = 640
NMS_IOU = 0.5
IOU_MATCH = 0.5
WIDE_MIN = 500
TILE = 320
FCOLS = ["s1_conf", "s2_conf", "verifier_prob", "s1_area_ratio",
         "s2_area_ratio", "s1s2_iou", "bbox_area_ratio", "bbox_aspect"]
BUCKET_EDGES = (0.00552, 0.01565)
BUCKETS = ("small", "medium", "large")
# v5-fix1: adaptive stage2 window — enlarge the s2 crop when the s1 box is
# large relative to the fixed SIDE window (large-blob closeup OOD fix).
# Weights of s1/s2/verifier unchanged; M3 fusion is refit on val each run.
S2_ADAPTIVE = False
S2_ADAPT_TRIG = 0.55   # trigger: max(bw,bh) >= trig * SIDE
S2_ADAPT_MULT = 2.4    # side = clamp(2.4 * max(bw,bh), SIDE, max(W,H))
# v5-fix2: gray-canvas padding for saturated s2 crops (closeup large blobs).
# When the crop covers ~the whole image (closeups) the calculus blob fills the
# 640 canvas and is OOD for s2; embedding the crop on a larger gray canvas
# restores a normal object/canvas ratio. Masks are remapped back to crop space
# before NMS, so downstream placement/verifier logic is unchanged.
S2_PAD = False
S2_PAD_TRIG = 0.35     # trigger: max(bw,bh) >= trig * crop max side (or saturated crop)
S2_PAD_FACTOR = 2.5
# v5-fix3: horizontal-flip TTA at stage2. The flipped variant is merged by
# unioning suppressed near-duplicate masks (IoU >= 0.6) into the kept one.
S2_TTA = False
S2_TTA_TRIG = 0.35     # only TTA when blob is relatively large in the crop
                       # (max(bw,bh) >= trig * crop max side) or crop saturated
# v5-fix7/FB: stage1-mask fallback candidate for large boxes (dir 1). OFF by
# default so archived runs reproduce bit-exact; the packaged CalculusCascadeModel
# (src/inference/models.py) enables it for the promoted production path. M3 must
# be refit with FB on (see scripts/final_test_v5_fix7.py fitfb) before enabling.
FB_ENABLE = False
FB_MIN_PX = 30
CLOSE_K = 5
LARGE_EDGE_FB = BUCKET_EDGES[1]   # 0.01565 == "large" relative-area edge


def morph_close(mask: np.ndarray, k: int = CLOSE_K) -> np.ndarray:
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx((mask > 0).astype(np.uint8), cv2.MORPH_CLOSE, ker)


def tta_fuse(keep: list[dict], sel: list[dict], iou_thr: float = 0.6) -> list[dict]:
    if not keep:
        return keep
    kept_ids = {id(k) for k in keep}
    supp = [s for s in sel if id(s) not in kept_ids]
    for k in keep:
        for s in supp:
            inter = int((k["bin"] & s["bin"]).sum())
            uni = int((k["bin"] | s["bin"]).sum())
            if uni and inter / uni >= iou_thr:
                k["bin"] = ((k["bin"] | s["bin"]) > 0).astype(np.uint8)
    return keep


def bucket_of(r: float) -> str:
    if r < BUCKET_EDGES[0]:
        return "small"
    if r < BUCKET_EDGES[1]:
        return "medium"
    return "large"


def _tile_grid(W: int, H: int) -> list[tuple[int, int, int, int]]:
    side = min(TILE, W, H)
    xs = sorted({0, max(0, W - side)}) if W > side else [0]
    ys = sorted({0, max(0, H - side)}) if H > side else [0]
    return [(x, y, min(x + side, W), min(y + side, H))
            for y in ys for x in xs]


def _nms_xyxy(boxes: np.ndarray, confs: np.ndarray, thr: float = 0.5):
    if len(boxes) == 0:
        return boxes, confs, np.zeros(0, int)
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
        if ok:
            keep.append(i)
    keep = np.array(sorted(keep), int)
    return boxes[keep], confs[keep], keep


def s1_pass(model, image_path: str, c1: float = C1, tiled: bool = True
            ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Class-0 boxes (orig coords), confs, original-size binary masks.

    Wide images (max(W,H) >= WIDE_MIN): 2x2 zoomed-tile grid (TILE px windows,
    25% overlap), per-tile predict, merge with box NMS.  Closeups: whole image.
    """
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), []
    H, W = img.shape[:2]

    def _extract(res, ox=0.0, oy=0.0):
        if res.boxes is None or len(res.boxes) == 0 or res.masks is None:
            return (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), [])
        cls = res.boxes.cls.cpu().numpy().astype(int)
        keep = np.where(cls == 0)[0]
        if len(keep) == 0:
            return (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), [])
        mk = res.masks.data.cpu().numpy()
        boxes = res.boxes.xyxy.cpu().numpy()[keep].copy()
        boxes[:, [0, 2]] += ox
        boxes[:, [1, 3]] += oy
        confs = res.boxes.conf.cpu().numpy()[keep]
        masks = [(mk[i] > 0.5).astype(np.uint8) for i in keep]
        return boxes, confs, masks

    if not tiled or max(W, H) < WIDE_MIN:
        res = model.predict(image_path, conf=c1, iou=0.6, imgsz=TARGET,
                            device=0, retina_masks=True, max_det=150,
                            verbose=False)[0]
        b, c, m = _extract(res)
        return b, c, m

    all_b, all_c, all_m = [], [], []
    all_meta = []  # (tile_x0, tile_y0, tw, th) per candidate
    tmp = "/tmp/opencode/_v3c_tile.jpg"
    for (x0, y0, x1, y1) in _tile_grid(W, H):
        cv2.imwrite(tmp, img[y0:y1, x0:x1])
        res = model.predict(tmp, conf=c1, iou=0.6, imgsz=TARGET, device=0,
                            retina_masks=True, max_det=60, verbose=False)[0]
        b, c, m = _extract(res, float(x0), float(y0))
        if len(b):
            all_b.append(b); all_c.append(c)
            all_m.extend(m)
            all_meta.extend([(x0, y0, x1 - x0, y1 - y0)] * len(b))
    if not all_b:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), []
    boxes = np.concatenate(all_b, 0)
    confs = np.concatenate(all_c, 0)
    boxes, confs, keep_idx = _nms_xyxy(boxes, confs, 0.5)
    masks_out = []
    for i in keep_idx:
        tx, ty, tw, th = all_meta[i]
        m = cv2.resize(all_m[i], (tw, th), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((H, W), np.uint8)
        canvas[ty:ty + th, tx:tx + tw] = m
        masks_out.append(canvas)
    return boxes, confs, masks_out


def s2_instances(model, crop: np.ndarray, c2: float = C2) -> list[dict]:
    """Class-0 raw instances: {bin640, s2_conf, bbox640, area}."""
    res = model.predict(crop, conf=c2, iou=0.6, imgsz=TARGET, device=0,
                        retina_masks=True, max_det=100, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0 or res.masks is None:
        return []
    cls = res.boxes.cls.cpu().numpy().astype(int)
    mk = res.masks.data.cpu().numpy()
    out = []
    for i in np.where(cls == 0)[0]:
        mi = (mk[i] > 0.5).astype(np.uint8)
        a = int(mi.sum())
        if a == 0:
            continue
        gx, gy, gw, gh = cv2.boundingRect(mi)
        out.append({"bin640": mi, "s2_conf": float(res.boxes.conf[i]),
                    "bbox640": (gx, gy, gx + gw, gy + gh), "area": a})
    return out


def s2_pad_canvas(crop: np.ndarray):
    """Embed crop centered on a gray canvas; returns (canvas, ox, oy)."""
    ch, cw = crop.shape[:2]
    cs = int(round(S2_PAD_FACTOR * max(cw, ch)))
    canvas = np.full((cs, cs, 3), 114, np.uint8)
    ox, oy = (cs - cw) // 2, (cs - ch) // 2
    canvas[oy:oy + ch, ox:ox + cw] = crop
    return canvas, ox, oy


def s2_bin_to_crop_space(bin640: np.ndarray, cw: int, ch: int, ox: int, oy: int,
                         cs: int) -> np.ndarray:
    """Remap a canvas-space 640 mask back to crop@640 space (as if the crop
    itself were resized to TARGET) so NMS/placement stay consistent."""
    big = cv2.resize(bin640, (cs, cs), interpolation=cv2.INTER_NEAREST)
    region = big[oy:oy + ch, ox:ox + cw]
    return cv2.resize(region, (TARGET, TARGET), interpolation=cv2.INTER_NEAREST)


def image_gts(img_path: str, raw_mask_path: str, H: int, W: int,
              min_px: int = 5) -> tuple[list[dict], np.ndarray]:
    m = cv2.imread(raw_mask_path, cv2.IMREAD_UNCHANGED)
    if m is None:
        m = np.zeros((H, W), np.uint8)
    if m.ndim == 3:
        m = m[:, :, 0]
    calc = (m == 3)
    other = np.isin(m, [1, 2, 4]).astype(np.uint8)
    gts = []
    if calc.any():
        n, comp = cv2.connectedComponents(calc.astype(np.uint8), 8)
        for i in range(1, n):
            px = int((comp == i).sum())
            if px < min_px:
                continue
            gts.append({"mask": (comp == i).astype(np.uint8), "px": px,
                        "bucket": bucket_of(px / float(H * W))})
    return gts, other


def make_verifier() -> nn.Module:
    vb = tvmodels.efficientnet_v2_s(
        weights=tvmodels.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    conv = vb.features[0][0]
    new = nn.Conv2d(4, conv.out_channels, conv.kernel_size, conv.stride,
                    conv.padding, conv.dilation, conv.groups, conv.bias)
    with torch.no_grad():
        new.weight[:, :3] = conv.weight
        new.weight[:, 3] = conv.weight.mean(dim=1)
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    vb.features[0][0] = new
    vb.classifier[1] = nn.Linear(vb.classifier[1].in_features, 1)
    vb.load_state_dict(torch.load(OUT / "verifier" / "best.pt", weights_only=False))
    return vb.to(0).eval()


def make_imgclf() -> nn.Module:
    m = tvmodels.efficientnet_v2_s(
        weights=tvmodels.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    m.load_state_dict(torch.load(OUT / "imgclf" / "best.pt", weights_only=False))
    return m.to(0).eval()


def verifier_probs(vb, crop_items: list[dict]) -> np.ndarray:
    if not crop_items:
        return np.zeros(0)
    inp = []
    for it in crop_items:
        rgb = cv2.cvtColor(it["crop_img"], cv2.COLOR_BGR2RGB)
        t = TF.resize(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255,
                      (224, 224), TF.InterpolationMode.BILINEAR)
        t = TF.normalize(t, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        mm = torch.from_numpy(it["mask160"].astype(np.float32))[None]
        mm = TF.resize(mm, (224, 224), TF.InterpolationMode.NEAREST)
        inp.append(torch.cat([t, (mm > 0.5).float()], 0))
    with torch.no_grad():
        return vb(torch.stack(inp).to(0)).squeeze(1).sigmoid().cpu().numpy()


def imgclf_probs(clf, imgs: list[np.ndarray]) -> np.ndarray:
    """imgs: BGR originals (varied size) -> sigmoid probs."""
    if not len(imgs):
        return np.zeros(0)
    inp = []
    for im in imgs:
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        t = TF.resize(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255,
                      (384, 384), TF.InterpolationMode.BILINEAR)
        inp.append(TF.normalize(t, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    with torch.no_grad():
        return clf(torch.stack(inp).to(0)).squeeze(1).sigmoid().cpu().numpy()


def collect_split(s1_model, s2_model, man: pd.DataFrame, split: str,
                  with_verifier=True, vb=None, with_imgclf=False, clf=None,
                  log_every: int = 200) -> list[dict]:
    rows = man[man.split == split].reset_index(drop=True)
    imgs, pending_imgclf = [], []
    for ri, r in rows.iterrows():
        img = cv2.imread(r.image_path)
        H, W = img.shape[:2]
        gts, other = image_gts(r.image_path, str(RAW / f"{r.image_id}.png"), H, W)
        feats, bins, rects, crop_local = [], [], [], []
        boxes, confs, masks = s1_pass(s1_model, r.image_path)
        for (b, sc1, msk) in zip(boxes, confs, masks):
            bx = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
            side = SIDE
            if S2_ADAPTIVE:
                bw, bh = bx[2] - bx[0], bx[3] - bx[1]
                if max(bw, bh) >= S2_ADAPT_TRIG * SIDE:
                    side = int(min(max(W, H), max(SIDE, S2_ADAPT_MULT * max(bw, bh))))
            x0 = int(round((bx[0] + bx[2]) / 2 - side / 2))
            y0 = int(round((bx[1] + bx[3]) / 2 - side / 2))
            x0 = max(0, min(x0, W - side)); y0 = max(0, min(y0, H - side))
            x1, y1 = min(x0 + side, W), min(y0 + side, H)
            crop_img = img[y0:y1, x0:x1]
            cw_, ch_ = x1 - x0, y1 - y0
            crop = cv2.resize(crop_img, (TARGET, TARGET), interpolation=cv2.INTER_AREA)
            pad_ctx = None
            if S2_PAD:
                bw, bh = bx[2] - bx[0], bx[3] - bx[1]
                saturated = cw_ * ch_ >= 0.9 * W * H
                rel = max(bw, bh) / max(cw_, ch_)
                if saturated or rel >= S2_PAD_TRIG:
                    canvas, ox, oy = s2_pad_canvas(crop_img)
                    pad_ctx = (ox, oy, canvas.shape[0], cw_, ch_)
                    crop = cv2.resize(canvas, (TARGET, TARGET), interpolation=cv2.INTER_AREA)
            insts = s2_instances(s2_model, crop)
            if pad_ctx is not None:
                ox, oy, cs, cw_, ch_ = pad_ctx
                insts = [dict(i, bin640=s2_bin_to_crop_space(i["bin640"], cw_, ch_, ox, oy, cs))
                         for i in insts]
            if S2_TTA:
                bw_, bh_ = bx[2] - bx[0], bx[3] - bx[1]
                saturated2 = cw_ * ch_ >= 0.9 * W * H
                rel2 = max(bw_, bh_) / max(cw_, ch_)
                if saturated2 or rel2 >= S2_TTA_TRIG:
                    flipped = s2_instances(s2_model, crop[:, ::-1])
                    for it in flipped:
                        it["bin640"] = np.ascontiguousarray(it["bin640"][:, ::-1])
                    insts = insts + flipped
            sel = [{"x0": 0, "y0": 0, "cw": TARGET, "ch": TARGET,
                    "bin": i["bin640"], "s2_conf": i["s2_conf"],
                    "score": i["s2_conf"]} for i in insts]
            if FB_ENABLE and float(msk.sum()) / float(H * W) >= LARGE_EDGE_FB:
                s1c640 = cv2.resize(morph_close(msk[y0:y1, x0:x1], CLOSE_K),
                                    (TARGET, TARGET),
                                    interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                if int(s1c640.sum()) >= FB_MIN_PX:
                    sel.append({"x0": 0, "y0": 0, "cw": TARGET, "ch": TARGET,
                                "bin": s1c640, "s2_conf": float(sc1),
                                "score": float(sc1)})
            keep = greedy_nms(sel, "mask_nms", NMS_IOU, TARGET, TARGET)
            if S2_TTA:
                keep = tta_fuse(keep, sel, 0.6)
            rectl = {"x0": x0, "y0": y0, "cw": x1 - x0, "ch": y1 - y0}
            for k in keep:
                P = k["bin"]
                m160 = (cv2.resize(P, (SIDE, SIDE),
                                   interpolation=cv2.INTER_NEAREST) > 0)
                s1m = (cv2.resize(msk[y0:y1, x0:x1], (SIDE, SIDE),
                                  interpolation=cv2.INTER_AREA) > 0.5)
                gx, gy, gw, gh = cv2.boundingRect(P.astype(np.uint8))
                feats.append({
                    "s1_conf": float(sc1), "s2_conf": float(k["s2_conf"]),
                    "s1_area_ratio": float(s1m.sum()) / (SIDE * SIDE),
                    "s2_area_ratio": float(m160.sum()) / (SIDE * SIDE),
                    "s1s2_iou": float((s1m & m160).sum() / max((s1m | m160).sum(), 1)),
                    "bbox_area_ratio": (gw * gh) / float(TARGET * TARGET),
                    "bbox_aspect": float(min(gw, gh) / max(gw, gh)),
                    "verifier_prob": None})
                bins.append(cv2.resize(P.astype(np.uint8), (rectl["cw"], rectl["ch"]),
                                       interpolation=cv2.INTER_NEAREST))
                rects.append(rectl)
                crop_local.append({"crop_img": crop_img, "mask160": m160})
        if feats and with_verifier and vb is not None:
            vp = verifier_probs(vb, crop_local)
            for i in range(len(feats)):
                feats[i]["verifier_prob"] = float(vp[i])
        imgs.append({"image_id": r.image_id, "domain": r.domain, "H": H, "W": W,
                     "gts": gts, "other": other, "feats": feats, "bins": bins,
                     "rects": rects, "has": bool(r.has_calculus),
                     "image_path": r.image_path})
        if with_imgclf and clf is not None:
            pending_imgclf.append(img)
        if (ri + 1) % log_every == 0:
            print(f"  {split} {ri + 1}/{len(rows)}", flush=True)
    if with_imgclf and clf is not None:
        probs = imgclf_probs(clf, pending_imgclf)
        for im, p in zip(imgs, probs):
            im["imgclf_prob"] = float(p)
    return imgs
