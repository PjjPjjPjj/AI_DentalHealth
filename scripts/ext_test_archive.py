"""External test on the new (md5-filtered) Kaggle archive images.

Leakage policy: archive images whose md5 exists in calculus_seg_v3 /
caries_det_v2 are DROPPED (they are our own train/val/test images).

Tests:
  A. calculus image-level screening on new Calculus(+) vs other-folder(-)
     images: streams B (EffNet imgclf@384) and C (yolocls@448), AUROC.
  B. caries detection on new YOLO-subset images carrying class-0 boxes:
     C2 @conf 0.55 (frozen), IoU>=0.5 matching -> P/R/F1.
  C. tooth segmentation qualitative overlays (3 new images).
Outputs: outputs/archive_ext_test/
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

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

sys.path.insert(0, f"{_ROOT}")
ROOT = Path(f"{_ROOT}")
A = ROOT / "datasets/raw/archive/extracted"
YIMG = A / "Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/Data/images"
YLBL = A / "Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/Data/labels"
OUT = ROOT / "outputs/archive_ext_test"
OUT.mkdir(parents=True, exist_ok=True)


def md5(p, chunk=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def collect_new():
    import pandas as pd
    ours = set()
    df = pd.read_csv(ROOT / "datasets/processed/calculus_seg_v3/split_manifest_v3.csv")
    ours |= {md5(p) for p in df.image_path}
    df2 = pd.read_csv(ROOT / "datasets/processed/caries_det_v2/split_manifest.csv")
    ours |= {md5(p) for p in df2.img}
    folders = {}
    for d in sorted(A.iterdir()):
        if d.is_dir() and not d.name.startswith("Caries_Gingivitus"):
            folders[d.name] = [p for p in d.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    folders["YOLO_subset"] = sorted(YIMG.rglob("*.jpg"))
    # global dedup: calculus first priority
    seen, new, kept_md5 = set(), {}, {}
    order = ["Calculus"] + [k for k in folders if k != "Calculus"]
    for fname in order:
        keep = []
        for p in folders[fname]:
            h = md5(p)
            if h in ours or h in seen:
                continue
            seen.add(h)
            kept_md5[h] = p
            keep.append(p)
        new[fname] = keep
    return new, kept_md5


def score_streams(paths):
    import torch
    from torchvision import models as tvmodels
    from torchvision.transforms import functional as TF
    from ultralytics import YOLO

    m = tvmodels.efficientnet_v2_s(weights=None)
    m.classifier[1] = __import__("torch").nn.Linear(m.classifier[1].in_features, 1)
    m.load_state_dict(torch.load(ROOT / "outputs/calculus_v3_cls/imgclf/best.pt",
                                 weights_only=False))
    m = m.to(0).eval()
    yc = YOLO(str(ROOT / "outputs/calculus_v3b_yolocls/weights/best.pt"))
    pb, pc = [], []
    for i in range(0, len(paths), 32):
        batch = paths[i:i + 32]
        xs = []
        for p in batch:
            img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            t = TF.resize(torch.from_numpy(img).permute(2, 0, 1).float() / 255, (384, 384))
            xs.append(TF.normalize(t, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        with torch.no_grad():
            pb += torch.sigmoid(m(torch.stack(xs).to(0))).squeeze(1).cpu().numpy().tolist()
    for i in range(0, len(paths), 64):
        res = yc.predict([str(p) for p in paths[i:i + 64]], imgsz=448, device=0, verbose=False)
        pc += [float(r.probs.data[1].item()) if r.probs is not None else 0.0 for r in res]
    return np.array(pb), np.array(pc)


def main() -> None:
    from sklearn.metrics import roc_auc_score
    new, kept_md5 = collect_new()
    print({k: len(v) for k, v in new.items()}, flush=True)

    # ---- A. calculus image-level ----
    pos = [p for p in new["Calculus"]]
    negs, neg_folder = [], []
    cap = 700
    for k in ["Mouth Ulcer", "Tooth Discoloration", "hypodontia", "Gingivitis", "Data caries"]:
        take = new[k][:cap]
        negs += take
        neg_folder += [k] * len(take)
    paths = pos + negs
    y = np.array([1] * len(pos) + [0] * len(negs))
    pb, pc = score_streams(paths)
    from sklearn.metrics import roc_auc_score as _auc
    import pandas as _pd
    per_neg = {}
    dfn = _pd.DataFrame({"y": y[len(pos):], "pB": pb[len(pos):], "pC": pc[len(pos):],
                         "folder": neg_folder})
    for k, g in dfn.groupby("folder"):
        per_neg[k] = {"mean_B": round(float(g.pB.mean()), 4),
                      "mean_C": round(float(g.pC.mean()), 4), "n": int(len(g))}
    rep = {"n_pos": len(pos), "n_neg": len(negs),
           "auroc_B_effnet": round(float(roc_auc_score(y, pb)), 4),
           "auroc_C_yolocls": round(float(roc_auc_score(y, pc)), 4),
           "auroc_B_per_negfolder": per_neg,
           "mean_prob_B_pos": round(float(pb[y == 1].mean()), 4),
           "mean_prob_B_neg": round(float(pb[y == 0].mean()), 4),
           "mean_prob_C_pos": round(float(pc[y == 1].mean()), 4),
           "mean_prob_C_neg": round(float(pc[y == 0].mean()), 4)}
    print("A:", rep, flush=True)

    # thumbnail grid of positives (domain check)
    thumbs = []
    for p in pos[:24]:
        im = cv2.resize(cv2.imread(str(p)), (200, 133))
        cv2.putText(im, Path(p).stem, (3, 14), 0, 0.4, (0, 255, 0), 1)
        thumbs.append(im)
    grid = np.vstack([np.hstack(thumbs[i:i + 4]) for i in range(0, len(thumbs), 4)])
    cv2.imwrite(str(OUT / "calculus_pos_thumbs.jpg"), grid)

    # ---- B. caries external ----
    # YOLO-subset label files map by md5 to images kept anywhere in the new set
    # (the (N).jpg names repeat across folders; physical identity = md5).
    from src.inference.models import CariesModel
    cm = CariesModel(weights=str(ROOT / "outputs/caries_c2/weights/best.pt"),
                     conf=0.55, iou=0.6, imgsz=960)
    # index: md5 -> list of caries gt boxes (normalized cx,cy,w,h) + a sample path
    caries_gt = {}
    for sp in ("val", "train"):
        for lf in (YLBL / sp).glob("*.txt"):
            lines = [l.split() for l in lf.read_text().splitlines() if l.strip()]
            caries = [l for l in lines if len(l) >= 5 and l[0] == "0"]
            if not caries:
                continue
            for ext in (".jpg", ".jpeg"):
                imgf = YIMG / sp / (lf.stem + ext)
                if imgf.exists():
                    caries_gt[md5(imgf)] = [(float(l[1]), float(l[2]),
                                             float(l[3]), float(l[4])) for l in caries]
                    break
    tp = fp = n_gt = 0
    tested = 0
    vis_saved = 0
    for h, p in kept_md5.items():
        gts_norm = caries_gt.get(h)
        if not gts_norm:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        H, W = img.shape[:2]
        gts = [(cx * W, cy * H, w * W, hh * H) for cx, cy, w, hh in gts_norm]
        n_gt += len(gts)
        dets = cm.predict(p)
        tested += 1
        used = set()
        for d in dets:
            bx, by = (d["bbox"][0] + d["bbox"][2]) / 2, (d["bbox"][1] + d["bbox"][3]) / 2
            bw, bh = d["bbox"][2] - d["bbox"][0], d["bbox"][3] - d["bbox"][1]
            best, bi = 0.0, None
            for gi, g in enumerate(gts):
                if gi in used:
                    continue
                ix0 = max(bx - bw / 2, g[0] - g[2] / 2)
                iy0 = max(by - bh / 2, g[1] - g[3] / 2)
                ix1 = min(bx + bw / 2, g[0] + g[2] / 2)
                iy1 = min(by + bh / 2, g[1] + g[3] / 2)
                inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
                ua = bw * bh + g[2] * g[3] - inter
                io = inter / ua if ua else 0
                if io > best:
                    best, bi = io, gi
            if best >= 0.5 and bi is not None:
                used.add(bi)
                tp += 1
            else:
                fp += 1
        if vis_saved < 4:
            for gi, g in enumerate(gts):
                cv2.rectangle(img, (int(g[0] - g[2] / 2), int(g[1] - g[3] / 2)),
                              (int(g[0] + g[2] / 2), int(g[1] + g[3] / 2)), (0, 255, 0), 1)
            for d in dets:
                x0, y0, x1, y1 = [int(v) for v in d["bbox"]]
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 1)
                cv2.putText(img, f"{d['conf']:.2f}", (x0, max(10, y0 - 3)), 0, 0.4, (0, 0, 255), 1)
            cv2.imwrite(str(OUT / f"caries_ext_{Path(p).stem}.jpg"), img)
            vis_saved += 1
    prec = tp / max(tp + fp, 1)
    rec = tp / max(n_gt, 1)
    rep_b = {"images_tested": tested, "n_gt_boxes": n_gt, "tp": tp, "fp": fp,
             "P": round(prec, 4), "R": round(rec, 4),
             "F1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
             "note": "conf=0.55 frozen; label protocol of the external set may differ from caries6313"}
    print("B:", rep_b, flush=True)

    # ---- C. tooth seg qualitative ----
    from src.inference.models import ToothModel
    tm = ToothModel(weights=str(ROOT / "outputs/iop_toothseg/weights/best.pt"),
                    conf=0.3, iou=0.7, imgsz=896)
    c_rows = []
    for k in ["Mouth Ulcer", "Tooth Discoloration", "Gingivitis"]:
        for p in new[k][:2]:
            img = cv2.imread(str(p))
            if img is None:
                continue
            teeth = tm.predict(img)
            for t in teeth:
                cnts, _ = cv2.findContours(t["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
            cv2.putText(img, f"{k} n={len(teeth)}", (5, 20), 0, 0.6, (0, 255, 255), 2)
            f = OUT / f"tooth_{k.replace(' ','_')}_{Path(p).stem}.jpg"
            cv2.imwrite(str(f), img)
            c_rows.append({"folder": k, "file": Path(p).name, "n_teeth": len(teeth)})
    rep_c = {"samples": c_rows}

    out = {"A_calculus_imgcls": rep, "B_caries_ext": rep_b, "C_tooth_seg": rep_c}
    (OUT / "ext_test_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
