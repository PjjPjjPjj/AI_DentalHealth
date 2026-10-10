"""Verify that the ONNX app backend matches the production torch pipeline.

Three stages, run in different environments:
  torch  (training env, has torch+ultralytics):
      python scripts/verify_onnx_consistency.py torch --out /tmp/opencode/vc_torch
  onnx   (appenv, has onnxruntime):
      python scripts/verify_onnx_consistency.py onnx  --out /tmp/opencode/vc_onnx
  compare (appenv): compare the two dirs, print per-domain consistency report.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# one image per domain (calculus closeup, wide; caries; tooth+plaque donor)
PICKS = {
    "calculus_closeup": ROOT / "datasets/processed/calculus_v5_zoom/images/val/(1009)__z0.jpg",
    "calculus_wide": ROOT / "demo_images/intraoral_real.jpg",
    "caries": ROOT / "datasets/processed/caries_det_v3/images/test/anonymous-frontalView-1726821735681.jpg",
    "tooth_plaque": ROOT / "demo_images/intraoral_163.jpg",
}


def _dump(out_dir: Path, image_id: str, det: list, seg_insts: list, cascade: list,
          plaque: float | None):
    d = out_dir / image_id
    d.mkdir(parents=True, exist_ok=True)
    cas_meta = []
    for x in cascade:
        ys, xs = np.nonzero(x["mask"] > 0)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        cas_meta.append({"bbox": bbox, "conf": x["conf"],
                         "s1_conf": x["s1_conf"], "area_ratio": x["area_ratio"]})
    data = {"det": [{"bbox": x["bbox"], "conf": x["conf"], "cls": x["cls"]} for x in det],
            "seg": [{"bbox": x["bbox"], "conf": x["conf"], "cls": x["cls"]} for x in seg_insts],
            "cascade": cas_meta,
            "plaque": plaque}
    (d / "meta.json").write_text(json.dumps(data, indent=1))
    for i, x in enumerate(seg_insts):
        cv2.imwrite(str(d / f"seg_{i:03d}.png"), x["mask"] * 255)
    for i, x in enumerate(cascade):
        cv2.imwrite(str(d / f"cas_{i:03d}.png"), x["mask"] * 255)


def _load(out_dir: Path, image_id: str):
    d = out_dir / image_id
    meta = json.loads((d / "meta.json").read_text())
    seg = []
    for p in sorted(d.glob("seg_*.png")):
        m = (cv2.imread(str(p), 0) > 127).astype(np.uint8)
        i = int(p.stem.split("_")[1])
        seg.append({"mask": m, **meta["seg"][i]})
    cas = []
    for p in sorted(d.glob("cas_*.png")):
        m = (cv2.imread(str(p), 0) > 127).astype(np.uint8)
        i = int(p.stem.split("_")[1])
        cas.append({"mask": m, **meta["cascade"][i]})
    return meta, seg, cas


def run_torch(out_dir: Path):
    sys.path.insert(0, str(ROOT))
    from src.inference.models import (CalculusCascadeModel, CariesModel,
                                      PlaqueModel, ToothModel)
    import yaml

    cfg = yaml.safe_load((ROOT / "configs/pipeline.yaml").read_text())
    icfg, wcfg = cfg["inference"], cfg["weights"]
    tooth = ToothModel(wcfg["tooth"], **icfg["tooth"])
    caries = CariesModel(wcfg["caries"], **icfg["caries"])
    plaque = PlaqueModel(wcfg["plaque"], **icfg["plaque"])
    cascade = CalculusCascadeModel(**icfg["calculus"], **wcfg["calculus"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, path in PICKS.items():
        if not Path(path).exists():
            print(f"[skip] {name}: {path}")
            continue
        img = cv2.imread(str(path))
        seg = tooth.predict(img)
        det = caries.predict(img)
        cas = cascade.predict(img)
        h, w = img.shape[:2]
        p = None
        if seg:
            x1, y1, x2, y2 = [int(v) for v in seg[0]["bbox"]]
            crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop.size:
                p = plaque.predict_proba(crop)
        _dump(out_dir, name, det, seg, cas, p)
        print(f"[torch] {name}: tooth={len(seg)} caries={len(det)} calculus={len(cas)} plaque={p}")


def run_onnx(out_dir: Path):
    from app.core.models_onnx import (CalculusCascadeOnnx, CariesModelOnnx,
                                      PlaqueModelOnnx, ToothModelOnnx)
    from app.core.pipeline import load_pipeline_config

    cfg = load_pipeline_config()
    icfg = cfg["inference"]
    md = ROOT / "app/models"
    tooth = ToothModelOnnx(md / "tooth_seg.onnx", **icfg["tooth"])
    caries = CariesModelOnnx(md / "caries_det.onnx", **icfg["caries"])
    plaque = PlaqueModelOnnx(md / "plaque_cls.onnx", imgsz=icfg["plaque"]["imgsz"])
    cascade = CalculusCascadeOnnx(
        s1=md / "calculus_s1.onnx", s2=md / "calculus_s2.onnx",
        verifier=md / "calculus_verifier.onnx",
        imgclf=md / "calculus_imgclf.onnx", fusion_npz=md / "calculus_fusion.npz",
        **icfg["calculus"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, path in PICKS.items():
        if not Path(path).exists():
            print(f"[skip] {name}: {path}")
            continue
        img = cv2.imread(str(path))
        seg = tooth.predict(img)
        det = caries.predict(img)
        cas = cascade.predict(img)
        h, w = img.shape[:2]
        p = None
        if seg:
            x1, y1, x2, y2 = [int(v) for v in seg[0]["bbox"]]
            crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop.size:
                p = plaque.predict_proba_batch([crop])[0]
        _dump(out_dir, name, det, seg, cas, p)
        print(f"[onnx ] {name}: tooth={len(seg)} caries={len(det)} calculus={len(cas)} plaque={p}")


def _match_iou(a: np.ndarray, bs: list) -> tuple[int, float]:
    best, bi = 0.0, -1
    for j, b in enumerate(bs):
        inter = int(((a > 0) & (b > 0)).sum())
        uni = int(((a > 0) | (b > 0)).sum())
        iou = inter / max(uni, 1)
        if iou > best:
            best, bi = iou, j
    return bi, best


def run_compare(torch_dir: Path, onnx_dir: Path):
    print(f"{'domain':18s} {'stage':10s} {'n_torch':>7s} {'n_onnx':>7s} {'boxIoU':>7s} {'maskIoU':>8s} {'dconf':>7s}")
    ok = True
    for name in PICKS:
        for stage, key in (("tooth", "seg"), ("cascade", "cascade")):
            tm, tseg, tcas = _load(torch_dir, name)
            om, oseg, ocas = _load(onnx_dir, name)
            ta, oa = (tseg, oseg) if key == "seg" else (tcas, ocas)
            conf_key = "conf"
            if not ta and not oa:
                print(f"{name:18s} {stage:10s} {0:7d} {0:7d} {'--':>7s} {'--':>8s} {'--':>7s}  (both empty)")
                continue
            used = set()
            box_ious, mask_ious, dconfs = [], [], []
            for x in ta:
                cands = [(j, y) for j, y in enumerate(oa) if id(y) not in used]
                j, mi = _match_iou(x["mask"], [y["mask"] for _, y in cands])
                if j < 0 or mi < 0.5:
                    print(f"{name:18s} {stage:10s} MISSED torch instance (maskIoU={mi:.3f})")
                    ok = False
                    continue
                y = cands[j][1]
                used.add(id(y))
                tb = np.array(x["bbox"], float)
                ob = np.array(y["bbox"], float)
                ix1, iy1 = np.maximum(tb[:2], ob[:2])
                ix2, iy2 = np.minimum(tb[2:], ob[2:])
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                ua = ((tb[2] - tb[0]) * (tb[3] - tb[1]) + (ob[2] - ob[0]) * (ob[3] - ob[1]) - inter)
                b_iou = inter / max(ua, 1e-9)
                box_ious.append(b_iou)
                mask_ious.append(mi)
                dconfs.append(abs(float(x[conf_key]) - float(y[conf_key])))
            unmatched = len(oa) - len(used)
            if unmatched:
                print(f"{name:18s} {stage:10s} NOTE: {unmatched} onnx-only instances")
                ok = False
            if box_ious:
                print(f"{name:18s} {stage:10s} {len(ta):7d} {len(oa):7d} "
                      f"{np.mean(box_ious):7.4f} {np.mean(mask_ious):8.4f} "
                      f"{max(dconfs):7.4f}")
    # plaque / caries meta
    for name in PICKS:
        tm, *_ = _load(torch_dir, name)
        om, *_ = _load(onnx_dir, name)
        tp, op = tm.get("plaque"), om.get("plaque")
        if tp is not None and op is not None:
            d = abs(tp - op)
            flag = "OK " if d < 0.01 else "DIFF"
            print(f"{name:18s} plaque     prob torch={tp:.4f} onnx={op:.4f} d={d:.4f} {flag}")
            ok &= d < 0.01
        tc, oc = tm["det"], om["det"]
        print(f"{name:18s} caries     n torch={len(tc)} onnx={len(oc)}"
              + ("" if len(tc) == len(oc) else "  <-- COUNT MISMATCH"))
        ok &= len(tc) == len(oc)
    print("RESULT:", "CONSISTENT" if ok else "INCONSISTENT")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["torch", "onnx", "compare"])
    ap.add_argument("--out", default="/tmp/opencode/vc_onnx")
    ap.add_argument("--torch-out", default="/tmp/opencode/vc_torch")
    args = ap.parse_args()
    if args.stage == "torch":
        run_torch(Path(args.out))
    elif args.stage == "onnx":
        run_onnx(Path(args.out))
    else:
        run_compare(Path(args.torch_out), Path(args.out))


if __name__ == "__main__":
    main()
