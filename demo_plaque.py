"""Plaque one-click demo: random tooth crop -> P(plaque), shown IN THE FULL IMAGE.

Just run (no args needed):
    python demo_plaque.py
Loads the trained YOLO11s-cls plaque model (configs/pipeline.yaml -> weights.plaque)
and its frozen sens-first threshold (inference.plaque.conf_threshold), picks a RANDOM
crop from datasets/processed/plaque_bin_v1/<split>/{normal,plaque}, runs inference, and
renders the crop **in the context of the full intraoral photo it came from** (the tooth
box is highlighted on the full image; the crop is shown as an inset). The crop->full-image
mapping is recovered from plaque_crops.csv + the raw Mendeley images/labels.

Outputs (directly observable):
    outputs/demo_plaque/plaque_demo.jpg            (single annotated card)
    outputs/demo_plaque/plaque_demo_<i>.jpg        (when --n > 1, one per crop)
    outputs/demo_plaque/plaque_demo_montage.jpg    (when --n > 1)
    outputs/demo_plaque/plaque_demo.json           (machine-readable result)

More:
    python demo_plaque.py --split val --n 6 --seed 7
    python demo_plaque.py --image datasets/processed/plaque_bin_v1/test/plaque/<file>.jpg
    python demo_plaque.py --crop-only        # don't reconstruct the full image
    python demo_plaque.py --list             # show dataset layout, no model
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

import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets/processed/plaque_bin_v1"
RAW = ROOT / "datasets/raw/Plaque"
PARTS = {1: RAW / "mendeley-dataset-materials_Part_1",
         2: RAW / "mendeley-dataset-materials_Part_2"}
OUT = ROOT / "outputs/demo_plaque"

NAME = {0: "normal", 1: "plaque"}
GREEN = (0, 170, 0)
RED = (0, 0, 220)
GRAY = (128, 128, 128)
BLACK = (30, 30, 30)
WHITE = (255, 255, 255)

_CSV: list | None = None


def load_model_and_thr():
    import sys
    sys.path.insert(0, str(ROOT))
    from src.inference.models import PlaqueModel
    from src.utils.config import load_yaml
    cfg = load_yaml(ROOT / "configs/pipeline.yaml")
    weights = cfg["weights"]["plaque"]
    thr = float(cfg.get("inference", {}).get("plaque", {}).get("conf_threshold", 0.5))
    return PlaqueModel(weights=weights), thr


def list_split(split: str) -> list[Path]:
    files = []
    for lab in ("normal", "plaque"):
        files.extend((DATA / split / lab).glob("*.jpg"))
    return sorted(files)


def _csv_rows():
    global _CSV
    if _CSV is None:
        _CSV = list(csv.DictReader(open(DATA / "plaque_crops.csv")))
    return _CSV


def find_row(crop_path: Path):
    name = crop_path.name
    for r in _csv_rows():
        if Path(r["crop"]).name == name:
            return r
    return None


def _match_box(lines, tid, bw, bh, W, H):
    """Pick the label line for tooth `tid` whose pixel box matches the crop size."""
    cands = []
    for f in lines:
        if len(f) != 6 or int(f[5]) != tid:
            continue
        cx, cy, w, h = map(float, f[1:5])
        x1 = int(np.clip((cx - w / 2) * W, 0, W)); x2 = int(np.clip((cx + w / 2) * W, 0, W))
        y1 = int(np.clip((cy - h / 2) * H, 0, H)); y2 = int(np.clip((cy + h / 2) * H, 0, H))
        cands.append(((x1, y1, x2, y2), abs((x2 - x1) - bw) + abs((y2 - y1) - bh)))
    if not cands:
        return None
    cands.sort(key=lambda c: c[1])
    return cands[0][0]


def locate_full_image(crop_path: Path):
    """Return (full_bgr, (x1,y1,x2,y2)) for a crop, or None if unrecoverable."""
    r = find_row(crop_path)
    if r is None:
        return None
    part = PARTS.get(int(r["part"]))
    if part is None:
        return None
    pat, iid = r["patient"], r["image_id"]
    img_dir, lbl_dir = part / "data" / "images" / pat, part / "data" / "labels" / pat
    hits = [p for d in (img_dir, part / "data" / "images") for p in d.glob(f"{iid}.*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not hits:
        return None
    full = cv2.imread(str(hits[0]))
    lf = lbl_dir / f"{iid}.txt"
    if full is None or not lf.exists():
        return None
    H, W = full.shape[:2]
    lines = [ln.split() for ln in lf.read_text().splitlines() if ln.strip()]
    box = _match_box(lines, int(r["tooth_id"]), int(r["box_w"]), int(r["box_h"]), W, H)
    if box is None:
        return None
    return full, box


def _bar_and_verdict(card, w, y, p, thr, pred, ok):
    color = RED if pred == 1 else GREEN
    bx0, bx1 = 24, w - 24
    cv2.putText(card, "P(plaque)", (bx0, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRAY, 1, cv2.LINE_AA)
    cv2.rectangle(card, (bx0, y), (bx1, y + 26), (220, 220, 220), -1)
    cv2.rectangle(card, (bx0, y), (bx0 + int((bx1 - bx0) * float(np.clip(p, 0, 1))), y + 26), color, -1)
    tx = bx0 + int((bx1 - bx0) * thr)
    cv2.line(card, (tx, y - 6), (tx, y + 32), BLACK, 2)
    cv2.putText(card, f"thr={thr:.2f}", (min(tx + 6, bx1 - 74), y + 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, BLACK, 1, cv2.LINE_AA)
    fy = y + 74
    cv2.putText(card, f"P(plaque) = {p:.4f}", (bx0, fy), cv2.FONT_HERSHEY_SIMPLEX, 0.62, BLACK, 2, cv2.LINE_AA)
    cv2.putText(card, f"pred = {NAME[pred].upper()}  ({'+' if p >= thr else '-'})",
                (bx0, fy + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    cv2.putText(card, ok, (w - 130, fy + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                GREEN if ok == "OK" else (255, 120, 0), 2, cv2.LINE_AA)
    return card


def render_card(crop, crop_path, p, thr, full=None, box=None, img_h=440, crop_h=190):
    pred = 1 if p >= thr else 0
    color = RED if pred == 1 else GREEN
    gt = crop_path.parent.name
    ok = "OK" if NAME[pred] == gt else "MISS"
    pad = 24

    if full is None or box is None:            # crop-only fallback
        h, w = crop.shape[:2]
        img = cv2.resize(crop, (max(1, int(w * img_h / h)), img_h))
        ih, iw = img.shape[:2]
        card_w = max(iw + 2 * pad, 560)
        card = np.full((44 + ih + 2 * pad + 26 + 108, card_w, 3), WHITE, np.uint8)
        cv2.putText(card, crop_path.name[:34], (pad, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)
        cv2.putText(card, f"GT: {gt}", (card_w - 150, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)
        y0 = 44 + pad
        card[y0:y0 + ih, pad:pad + iw] = img
        cv2.rectangle(card, (pad - 3, y0 - 3), (pad + iw + 3, y0 + ih + 3), color, 4)
        return _bar_and_verdict(card, card_w, y0 + ih + pad, p, thr, pred, ok)

    # full-image context
    fh, fw = full.shape[:2]
    disp_w = 640
    s = disp_w / fw
    disp = cv2.resize(full, (disp_w, max(1, int(fh * s))))
    x1, y1, x2, y2 = [int(v * s) for v in box]
    cv2.rectangle(disp, (x1, y1), (x2, y2), color, 3)
    cv2.putText(disp, f"{NAME[pred]} {p:.2f}", (x1, max(14, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    ch, cw = crop.shape[:2]
    DWH, DWW = disp.shape[:2]
    sc = min(max(40, min(crop_h, DWH - 12)) / ch, max(40, DWW - 12) / cw)
    iw, ih = max(1, int(cw * sc)), max(1, int(ch * sc))
    ins = cv2.resize(crop, (iw, ih))
    ix = DWW - iw - 6
    iy = DWH - ih - 6
    disp[iy:iy + ih, ix:ix + iw] = ins
    cv2.rectangle(disp, (ix - 2, iy - 2), (ix + iw + 2, iy + ih + 2), WHITE, 2)
    cv2.rectangle(disp, (ix - 2, iy - 2), (ix + iw + 2, iy + ih + 2), color, 2)
    cv2.arrowedLine(disp, (ix, iy), (max(0, min((x1 + x2) // 2, disp_w - 1)), (y1 + y2) // 2),
                    color, 1, tipLength=0.08)

    ihh, iww = disp.shape[:2]
    card_w = max(iww + 2 * pad, 560)
    card = np.full((44 + ihh + 2 * pad + 26 + 108, card_w, 3), WHITE, np.uint8)
    cv2.putText(card, crop_path.name[:34], (pad, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)
    cv2.putText(card, f"GT: {gt}  (full image)", (card_w - 300, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)
    y0 = 44 + pad
    card[y0:y0 + ihh, pad:pad + iww] = disp
    return _bar_and_verdict(card, card_w, y0 + ihh + pad, p, thr, pred, ok)


def montage(cards, out: Path) -> None:
    h = 460
    tiles = [cv2.resize(c, (int(c.shape[1] * h / c.shape[0]), h)) for c in cards]
    ncol = int(np.ceil(np.sqrt(len(tiles))))
    nrow = int(np.ceil(len(tiles) / ncol))
    tw = max(t.shape[1] for t in tiles)
    canvas = np.full((nrow * h + 12 * (nrow - 1), ncol * tw + 12 * (ncol - 1), 3), 235, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, ncol)
        canvas[r * (h + 12):r * (h + 12) + t.shape[0], c * (tw + 12):c * (tw + 12) + t.shape[1]] = t
    cv2.imwrite(str(out), canvas)


_PALETTE = [(230, 57, 70), (69, 123, 157), (42, 157, 143), (233, 196, 106),
            (231, 111, 81), (145, 109, 190), (52, 168, 83), (214, 93, 176)]


def render_seg_panel(full, teeth) -> np.ndarray:
    """Full-image panel showing tooth instance segmentation results."""
    fh, fw = full.shape[:2]
    disp_w = 640
    s = disp_w / fw
    disp = cv2.resize(full, (disp_w, max(1, int(fh * s))))
    H, W = disp.shape[:2]
    fill = np.zeros_like(disp)
    for i, t in enumerate(teeth):
        m = t["mask"]
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        color = _PALETTE[i % len(_PALETTE)]
        fill[m > 0] = color
    panel = cv2.addWeighted(disp, 0.65, fill, 0.35, 0)
    for i, t in enumerate(teeth):
        m = t["mask"]
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        color = _PALETTE[i % len(_PALETTE)]
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel, cnts, -1, color, 1, cv2.LINE_AA)
        x1, y1, x2, y2 = [int(v * s) for v in t["bbox"]]
        cv2.rectangle(panel, (x1, y1), (x2, y2), color, 1)
        cv2.putText(panel, f"T{i + 1}", (max(2, x1), max(12, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    cv2.putText(panel, f"Tooth segmentation: {len(teeth)} instances (iop_toothseg @896)",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, WHITE, 1, cv2.LINE_AA)
    return panel


def main() -> None:
    ap = argparse.ArgumentParser(description="Plaque random-crop demo (full-image view, click to run).")
    ap.add_argument("--image", help="explicit crop path (skip random pick)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=1, help="number of random crops (>=2 makes a montage)")
    ap.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    ap.add_argument("--threshold", type=float, default=None, help="override frozen thr")
    ap.add_argument("--crop-only", action="store_true", help="do not reconstruct the full image")
    ap.add_argument("--seg", action="store_true",
                    help="also render a tooth-segmentation panel per case "
                         "(plaque_demo<i>_seg.jpg + seg montage)")
    ap.add_argument("--verify", type=int, default=0,
                    help="self-check: N random crops PER CLASS from --split, scored against "
                         "human GT; prints agreement + saves miss montage")
    ap.add_argument("--list", action="store_true", help="just print dataset layout, no model")
    args = ap.parse_args()

    if args.list:
        print(f"data root: {DATA} (exists={DATA.exists()})")
        for sp in ("train", "val", "test"):
            counts = {lab: len(list((DATA / sp / lab).glob("*.jpg"))) for lab in ("normal", "plaque")}
            print(f"  {sp:5}: {counts}")
        return

    model, thr = load_model_and_thr()
    if args.threshold is not None:
        thr = args.threshold

    if args.verify:
        rnd = random.Random(args.seed)
        per_class = {}
        for lab in ("plaque", "normal"):
            files = sorted((DATA / args.split / lab).glob("*.jpg"))
            per_class[lab] = rnd.sample(files, min(args.verify, len(files)))
        tally = {"plaque": {"n": 0, "ok": 0}, "normal": {"n": 0, "ok": 0}}
        rows, miss_cards = [], []
        all_files = per_class["plaque"] + per_class["normal"]
        for cp in all_files:
            crop = cv2.imread(str(cp))
            if crop is None:
                continue
            p = model.predict_proba(crop)
            pred = 1 if p >= thr else 0
            gt = 1 if cp.parent.name == "plaque" else 0
            ok = pred == gt
            tally[NAME[gt]]["n"] += 1
            tally[NAME[gt]]["ok"] += int(ok)
            rows.append({"crop": str(cp.relative_to(ROOT)), "gt": NAME[gt],
                         "pred": NAME[pred], "p_plaque": round(float(p), 4), "correct": ok})
            if not ok and len(miss_cards) < 12:
                miss_cards.append(render_card(crop, cp, p, thr))
        n_ok = tally["plaque"]["ok"] + tally["normal"]["ok"]
        n_all = tally["plaque"]["n"] + tally["normal"]["n"]
        sens = tally["plaque"]["ok"] / max(tally["plaque"]["n"], 1)
        spec = tally["normal"]["ok"] / max(tally["normal"]["n"], 1)
        print(f"\n=== 抽样自检 vs 人工标注（split={args.split}, "
              f"每类 {args.verify}, seed={args.seed}, thr={thr:.3f}）===")
        print(f"GT=plaque : 与标注一致 {tally['plaque']['ok']}/{tally['plaque']['n']}"
              f"  (Sens {sens:.3f})")
        print(f"GT=normal : 与标注一致 {tally['normal']['ok']}/{tally['normal']['n']}"
              f"  (Spec {spec:.3f})")
        print(f"总体一致率: {n_ok}/{n_all} = {n_ok / max(n_all, 1):.3f}")
        (OUT / "plaque_demo_verify.json").write_text(json.dumps(
            {"split": args.split, "n_per_class": args.verify, "seed": args.seed,
             "threshold": thr, "sens_gt_plaque": round(sens, 4),
             "spec_gt_normal": round(spec, 4),
             "accuracy": round(n_ok / max(n_all, 1), 4), "records": rows},
            indent=2, ensure_ascii=False), encoding="utf-8")
        if miss_cards:
            montage(miss_cards, OUT / "plaque_demo_misses.jpg")
            print(f"错判案例拼图 -> {OUT / 'plaque_demo_misses.jpg'}")
        print(f"JSON -> {OUT / 'plaque_demo_verify.json'}")
        return

    if args.image:
        picks = [Path(args.image)]
    else:
        files = list_split(args.split)
        if not files:
            print(f"no crops found under {DATA / args.split}")
            return
        rnd = random.Random(args.seed)
        picks = rnd.sample(files, min(args.n, len(files)))

    OUT.mkdir(parents=True, exist_ok=True)
    results, cards, seg_cards = [], [], []
    seg_model = None
    for i, cp in enumerate(picks):
        crop = cv2.imread(str(cp))
        if crop is None:
            print(f"skip undecodable: {cp}")
            continue
        p = model.predict_proba(crop)
        pred = 1 if p >= thr else 0
        gt = 1 if cp.parent.name == "plaque" else 0
        full = box = None
        if not args.crop_only:
            loc = locate_full_image(cp)
            if loc:
                full, box = loc
        card = render_card(crop, cp, p, thr, full, box)
        suffix = f"_{i}" if len(picks) > 1 else ""
        vis = OUT / f"plaque_demo{suffix}.jpg"
        cv2.imwrite(str(vis), card)
        cards.append(card)
        if args.seg and full is not None:
            if seg_model is None:
                from src.inference.models import ToothModel
                seg_model = ToothModel(ROOT / "outputs/iop_toothseg/weights/best.pt")
            teeth = seg_model.predict(full)
            segp = render_seg_panel(full, teeth)
            seg_vis = OUT / f"plaque_demo{suffix}_seg.jpg"
            cv2.imwrite(str(seg_vis), segp)
            seg_cards.append(segp)
        results.append({"crop": str(cp.relative_to(ROOT)), "split": args.split,
                        "view": "full_image" if full is not None else "crop_only",
                        "gt": NAME[gt], "pred": NAME[pred],
                        "p_plaque": round(float(p), 4), "threshold": round(thr, 4),
                        "correct": pred == gt, "visualization": str(vis.relative_to(ROOT))})
        print(f"{NAME[pred]:>6}  P(plaque)={p:.4f}  (GT={NAME[gt]})  "
              f"[{'full' if full is not None else 'crop'}]  {cp.name}  -> {vis.name}")

    if seg_cards:
        montage(seg_cards, OUT / "plaque_demo_seg_montage.jpg")
        print(f"分割结果拼图 -> {OUT / 'plaque_demo_seg_montage.jpg'}")
    if len(picks) > 1:
        montage(cards, OUT / "plaque_demo_montage.jpg")
        print(f"montage -> {OUT / 'plaque_demo_montage.jpg'}")
    (OUT / "plaque_demo.json").write_text(json.dumps(
        {"model": "plaque YOLO11s-cls", "threshold": thr, "results": results},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nJSON -> {OUT / 'plaque_demo.json'}")
    print(f"看图 -> {OUT / 'plaque_demo.jpg'}" if len(picks) == 1 else
          f"看图 -> {OUT / 'plaque_demo_montage.jpg'}")


if __name__ == "__main__":
    main()
