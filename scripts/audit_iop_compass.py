"""IOP-Compass full audit.

Checks: completeness (patients x 5 views), resolution (PNG header, all 5000),
md5 exact dups, dHash near-dups (hamming<=8), JSON annotation quality (teeth
count, FDI coverage per view, contour area sanity), calculus-visibility spot
check (v3c S1 on 30 work-resized images), thumbnails.

Outputs: outputs/reports/iop_compass_audit.json (+ thumbnails)
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import hashlib
import json
import re
import struct
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

ROOT = Path(f"{_ROOT}/datasets/raw/IOP_Compass/IOP-Compass")
REPORT = Path(f"{_ROOT}/outputs/reports/iop_compass_audit.json")
S1_W = "outputs/calculus_v3c_stage1/weights/best.pt"


def md5_of(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def png_size(p: Path) -> tuple[int, int]:
    with open(p, "rb") as f:
        head = f.read(24)
    w, h = struct.unpack(">II", head[16:24])
    return w, h


def dhash_work(p: Path) -> tuple[str, bytes]:
    img = cv2.imread(str(p), cv2.IMREAD_REDUCED_COLOR_8)
    if img is None:
        return p.name, b""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (17, 16))
    return p.name, np.packbits((g[:, 1:] > g[:, :-1]).astype(np.uint8).ravel()).tobytes()


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def contour_area_ok(teeth):
    bad = 0
    for t in teeth:
        cont = t.get("contour")
        if not cont or len(cont) < 3:
            bad += 1
            continue
        a = cv2.contourArea(np.array(cont, np.float32))
        bx = max(1, (t["x_max"] - t["x_min"]) * (t["y_max"] - t["y_min"]))
        if a <= 0 or a / bx > 1.05 or a / bx < 0.05:
            bad += 1
    return bad


def s1_spot_check(paths):
    model = YOLO(S1_W)
    out = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        H, W = img.shape[:2]
        s = 1536 / max(H, W)
        img = cv2.resize(img, (int(W * s), int(H * s)))
        cv2.imwrite("/tmp/opencode/_iop_spot.jpg", img)
        res = model.predict("/tmp/opencode/_iop_spot.jpg", conf=0.02, iou=0.6,
                            imgsz=640, device=0, verbose=False)[0]
        n = 0
        cmax = 0.0
        if res.boxes is not None and len(res.boxes):
            cls = res.boxes.cls.cpu().numpy().astype(int)
            m = cls == 0
            n = int(m.sum())
            cmax = float(res.boxes.conf.cpu().numpy()[m].max()) if n else 0.0
        out.append({"img": p.name, "n_cls0": n, "max_conf": round(cmax, 3)})
    return out


def main() -> None:
    pngs = sorted(ROOT.rglob("*.png"))
    jsons = sorted(ROOT.rglob("*.json"))
    print(f"png {len(pngs)} json {len(jsons)}", flush=True)

    # completeness
    pat_view = defaultdict(set)
    for p in pngs:
        m = re.match(r"IOP_(\w+)_(\d+)\.png", p.name)
        if m:
            pat_view[p.parent.name].add(m.group(1))
    n_pat = len(pat_view)
    views_per = Counter(len(v) for v in pat_view.values())
    full5 = sum(1 for v in pat_view.values() if len(v) == 5)

    # resolution (header, all)
    res = []
    bad_png = []
    for p in pngs:
        try:
            w, h = png_size(p)
            res.append((w, h))
        except Exception:
            bad_png.append(p.name)
    ws = np.array([r[0] for r in res]); hs = np.array([r[1] for r in res])
    long_side = np.maximum(ws, hs)

    # md5 (all) in parallel
    with ProcessPoolExecutor(12) as ex:
        md5s = list(ex.map(md5_of, pngs, chunksize=20))
    md5_counter = Counter(md5s)
    dup_md5_groups = {k: v for k, v in md5_counter.items() if v > 1}

    # dHash near-dups
    with ProcessPoolExecutor(12) as ex:
        hlist = list(ex.map(dhash_work, pngs, chunksize=20))
    uf = UF(len(pngs))
    packed = np.stack([np.frombuffer(h, np.uint8) for _, h in hlist])
    POP = np.array([bin(b).count("1") for b in range(256)], np.uint8)
    near = 0
    for a in range(0, len(pngs), 256):
        chunk = packed[a:a + 256]
        d = POP[np.bitwise_xor(chunk[:, None, :], packed[None, :, :])].sum(axis=2)
        for bi, bj in zip(*np.where((d <= 8) & (d > 0))):
            uf.union(a + bi, int(a + bj))
            near += 1
    groups = len({uf.find(i) for i in range(len(pngs))})

    # JSON audit
    cls_c, arch_c = Counter(), Counter()
    teeth_counts, fdi_c, bad_contours = [], Counter(), 0
    fdi_by_view = defaultdict(Counter)
    n_json_bad = 0
    for j in jsons:
        try:
            o = json.loads(j.read_text())
        except Exception:
            n_json_bad += 1
            continue
        vm = re.match(r"IOP_(\w+)_", j.name)
        view = vm.group(1) if vm else "?"
        for t in o.get("teeth", []):
            cls_c[t.get("class_name")] += 1
            arch_c[t.get("arch")] += 1
            fdi = str(t.get("FDI_NUM"))
            fdi_c[fdi] += 1
            fdi_by_view[view][fdi] += 1
            bad_contours += contour_area_ok([t])
        teeth_counts.append(o.get("annotation_count", 0))
    molar_fd = {"16", "17", "18", "26", "27", "28",
                "36", "37", "38", "46", "47", "48"}
    molar_total = sum(v for k, v in fdi_c.items() if k in molar_fd)

    # calculus visibility spot check (30 images, work-resized 1536)
    import random
    random.seed(0)
    spot_paths = random.sample(pngs, 30)
    spot = s1_spot_check(spot_paths)
    n_pos_spot = sum(1 for s in spot if s["n_cls0"] > 0)
    cmaxs = [s["max_conf"] for s in spot if s["max_conf"] > 0]

    # thumbnails: 2 per view
    tdir = Path(f"{_ROOT}/outputs/iop_thumbnails")
    tdir.mkdir(exist_ok=True)
    seen = defaultdict(int)
    for p in pngs:
        vm = re.match(r"IOP_(\w+)_", p.name)
        v = vm.group(1) if vm else "?"
        if seen[v] < 3:
            img = cv2.imread(str(p), cv2.IMREAD_REDUCED_COLOR_8)
            if img is not None:
                cv2.imwrite(str(tdir / f"thumb_{v}_{seen[v]}_{p.name[:20]}.jpg"), img)
                seen[v] += 1

    rep = {
        "png": len(pngs), "json": len(jsons), "unreadable_png": len(bad_png),
        "bad_json": n_json_bad,
        "patients": n_pat, "views_per_patient": dict(views_per),
        "patients_with_all_5": full5,
        "long_side_px": {"min": int(long_side.min()),
                         "p5": int(np.percentile(long_side, 5)),
                         "median": int(np.median(long_side)),
                         "p95": int(np.percentile(long_side, 95)),
                         "max": int(long_side.max())},
        "under_1280_images": int((long_side < 1280).sum()),
        "md5_dup_groups": len(dup_md5_groups),
        "dhash_near_groups": near, "unique_groups": groups,
        "class_names": dict(cls_c), "arch": dict(arch_c),
        "teeth_per_img": {"min": int(min(teeth_counts)),
                          "median": float(np.median(teeth_counts)),
                          "max": int(max(teeth_counts))},
        "fdi_unique": len(fdi_c), "molar_annotations": molar_total,
        "fdi_by_view_sample": {v: dict(c.most_common(6))
                               for v, c in fdi_by_view.items()},
        "bad_contours": bad_contours,
        "s1_spot_check": {"n_images": len(spot),
                          "images_with_cls0": n_pos_spot,
                          "conf_p50_p90": (np.percentile(cmaxs, [50, 90])
                                           .round(3).tolist() if cmaxs else [])},
        "thumbnails": str(tdir),
    }
    REPORT.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
