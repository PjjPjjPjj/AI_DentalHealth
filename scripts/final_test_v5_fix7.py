"""FINAL TEST v5-fix7: large-lesion recall via stage1-mask fallback + boundary close.

Hypothesis (16-large-GT FN trace of v5-fix4): 9/16 large FNs are stage2 failures
where stage1 ALREADY boxed the blob -- 5 near-miss (best s2 IoU 0.34-0.49, just
under the 0.5 TP bar) + 4 mask-off (s2 emitted nothing overlapping the GT). The
stage1 retina mask is a real detector-verified region the cascade throws away
(it is only used as a fusion feature, never emitted -- collect_split / models.py).

Two inference-only hooks, all weights frozen (v5 stage1 / v3b stage2 / verifier),
scored through the SAME verifier + M3 fusion so precision is still decided by the
calibrated head:
  FB    (dir 1): for LARGE s1 boxes (full-image area ratio >= LARGE_EDGE) append
        an extra candidate = morph-closed stage1 crop mask (proxy s2_conf = s1_conf).
        Mask-NMS dedups it vs any s2 instance that already claimed the region, so
        it survives only where s2 under-covered -> recovers "mask-off" FNs.
  CLOSE (dir 2): morph CLOSE on the s2 instance mask (TARGET space) for large boxes
        -- smooths ragged/eroded boundaries to nudge "near-miss" IoU over the bar.

CONFIG-INDEPENDENT YOLO work (stage1 boxes + per-box stage2 instances + GT) is
cached to disk ONCE (resumable in small chunks, so no single call runs too long);
per-config aggregation (fallback/close/verifier/M3/metrics) is cheap.

Self-contained: defines its own cache+aggregation, mutates NO shared code. Roll
back = delete this file and outputs/calculus_v5_fix7/ (nothing else is touched).

CLI (run as module from repo root):
  python -m scripts.final_test_v5_fix7 build <val|test> <max_n>   # cache up to max_n
  python -m scripts.final_test_v5_fix7 status
  python -m scripts.final_test_v5_fix7 agg A,B,C,D                 # M3 fit + metrics
Configs:  A FB0 CLOSE0 (parity vs fix4) | B FB1 CLOSE0 | C FB1 CLOSE1 | D FB0 CLOSE1
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
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from ultralytics import YOLO

import scripts.v3_pipeline_lib as _lib
_lib.OUT = Path("outputs/calculus_v5_cls")   # shared weights live here (read-only)
_lib.C1 = 0.005
_lib.S2_ADAPTIVE = True
_lib.S2_TTA = True
from scripts.v3_pipeline_lib import (BUCKETS, DATA, FCOLS, NMS_IOU, SIDE, TARGET,
                                     bucket_of, image_gts, make_verifier, s1_pass,
                                     s2_instances, tta_fuse, verifier_probs)
from scripts.eval_ensemble_v3b import effnet_tta_probs, yolocls_probs
from src.inference.fusion import greedy_nms, rec_bin

OUT = Path("outputs/calculus_v5_fix7")
CACHE = OUT / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

IOU_MATCH = 0.5
THR_STD = 0.6
THR_LARGE = 0.45
LARGE_EDGE = 0.01565
TAU = 0.4
THR_RESCUE = 0.25
FB_MIN_PX = 30
CLOSE_K = 5


# ---------------- pack/unpack helpers (bool -> uint8 bit packed) --------------
def _pack(arr: np.ndarray):
    return np.packbits(arr.astype(np.uint8).ravel()), tuple(arr.shape)


def _unpack(p, shape):
    return np.unpackbits(p)[: int(np.prod(shape))].reshape(shape).astype(np.uint8)


def _close(mask: np.ndarray, k: int) -> np.ndarray:
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx((mask > 0).astype(np.uint8), cv2.MORPH_CLOSE, ker)


def _rect_for(bx, W, H):
    side = SIDE
    bw, bh = bx[2] - bx[0], bx[3] - bx[1]
    if _lib.S2_ADAPTIVE and max(bw, bh) >= _lib.S2_ADAPT_TRIG * SIDE:
        side = int(min(max(W, H), max(SIDE, _lib.S2_ADAPT_MULT * max(bw, bh))))
    x0 = int(round((bx[0] + bx[2]) / 2 - side / 2))
    y0 = int(round((bx[1] + bx[3]) / 2 - side / 2))
    x0 = max(0, min(x0, W - side)); y0 = max(0, min(y0, H - side))
    x1, y1 = min(x0 + side, W), min(y0 + side, H)
    return {"x0": x0, "y0": y0, "cw": x1 - x0, "ch": y1 - y0}


# --------------------------- expensive cache build ---------------------------
def build_one(s1m, s2m, r) -> dict:
    img = cv2.imread(r.image_path)
    H, W = img.shape[:2]
    gts, _ = image_gts(r.image_path, str(_lib.RAW / f"{r.image_id}.png"), H, W)
    boxes, confs, masks = s1_pass(s1m, r.image_path)
    rec_boxes = []
    for (b, sc1, msk) in zip(boxes, confs, masks):
        bx = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
        rect = _rect_for(bx, W, H)
        x0, y0, cw, ch = rect["x0"], rect["y0"], rect["cw"], rect["ch"]
        x1, y1 = x0 + cw, y0 + ch
        crop_img = img[y0:y1, x0:x1]
        crop = cv2.resize(crop_img, (TARGET, TARGET), interpolation=cv2.INTER_AREA)
        insts = s2_instances(s2m, crop)
        if _lib.S2_TTA:
            bw, bh = bx[2] - bx[0], bx[3] - bx[1]
            saturated = cw * ch >= 0.9 * W * H
            rel = max(bw, bh) / max(cw, ch)
            if saturated or rel >= _lib.S2_TTA_TRIG:
                flipped = s2_instances(s2m, crop[:, ::-1])
                for it in flipped:
                    it["bin640"] = np.ascontiguousarray(it["bin640"][:, ::-1])
                insts = insts + flipped
        s1crop_p, s1crop_sh = _pack(msk[y0:y1, x0:x1])
        insts_p = [(np.packbits(i["bin640"].astype(np.uint8).ravel()),
                    tuple(i["bin640"].shape), float(i["s2_conf"])) for i in insts]
        rec_boxes.append({"conf": float(sc1), "rect": rect,
                          "s1crop_p": s1crop_p, "s1crop_sh": s1crop_sh,
                          "s1area_full": float(msk.sum()) / float(H * W),
                          "insts": insts_p})
    return {"image_id": str(r.image_id), "H": H, "W": W,
            "has": bool(r.has_calculus), "image_path": r.image_path,
            "gts": [(_pack(g["mask"]), g["px"], g["bucket"]) for g in gts],
            "boxes": rec_boxes}


def cmd_build(split: str, max_n: int) -> None:
    s1m = YOLO("outputs/calculus_v5_stage1/weights/best.pt")
    s2m = YOLO("outputs/calculus_v3b_stage2/weights/best.pt")
    man = pd.read_csv(DATA / "split_manifest_v3.csv")
    rows = man[man.split == split].reset_index(drop=True)
    done = todo = 0
    for r in rows.itertuples():
        f = CACHE / f"{r.image_id}.pkl"
        if f.exists():
            done += 1
            continue
        if todo >= max_n:
            break
        rec = build_one(s1m, s2m, r)
        tmp = f.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(rec, protocol=pickle.HIGHEST_PROTOCOL))
        tmp.rename(f)
        todo += 1
        print(f"  built {split} {todo}/{max_n} id={r.image_id} "
              f"boxes={len(rec['boxes'])} gts={len(rec['gts'])}", flush=True)
    print(f"[build {split}] this call={todo} cached={done + todo}/{len(rows)}", flush=True)


def cmd_status() -> None:
    man = pd.read_csv(DATA / "split_manifest_v3.csv")
    for sp in ("val", "test"):
        ids = man[man.split == sp].image_id.astype(str).tolist()
        have = sum(1 for i in ids if (CACHE / f"{i}.pkl").exists())
        print(f"{sp}: {have}/{len(ids)} cached", flush=True)


# --------------------------- cheap aggregation -------------------------------
def load_cached(man, split):
    rows = man[man.split == split].reset_index(drop=True)
    recs = []
    for r in rows.itertuples():
        f = CACHE / f"{r.image_id}.pkl"
        if not f.exists():
            continue
        recs.append(pickle.loads(f.read_bytes()))
    return recs


def aggregate(man, split, vb, cfg):
    FB, CLOSE = cfg["FB"], cfg["CLOSE"]
    recs = load_cached(man, split)
    out = []
    for rec in recs:
        H, W = rec["H"], rec["W"]
        img = cv2.imread(rec["image_path"])
        gts = [{"mask": _unpack(*gp), "px": px, "bucket": bk} for gp, px, bk in rec["gts"]]
        feats, bins, rects, crop_local = [], [], [], []
        for b in rec["boxes"]:
            rect = b["rect"]
            x0, y0, cw, ch = rect["x0"], rect["y0"], rect["cw"], rect["ch"]
            x1, y1 = x0 + cw, y0 + ch
            sc1 = b["conf"]
            s1crop = _unpack(b["s1crop_p"], b["s1crop_sh"])   # native rect res
            insts = [{"bin640": _unpack(p, sh), "s2_conf": c}
                     for (p, sh, c) in b["insts"]]
            cands = [(i["bin640"], float(i["s2_conf"])) for i in insts]
            large_box = b["s1area_full"] >= LARGE_EDGE
            if FB and large_box:
                s1c640 = cv2.resize(_close(s1crop, CLOSE_K), (TARGET, TARGET),
                                    interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                if int(s1c640.sum()) >= FB_MIN_PX:
                    cands.append((s1c640, float(sc1)))
            sel = [{"x0": 0, "y0": 0, "cw": TARGET, "ch": TARGET,
                    "bin": P, "s2_conf": c, "score": c} for (P, c) in cands]
            keep = greedy_nms(sel, "mask_nms", NMS_IOU, TARGET, TARGET)
            if _lib.S2_TTA:
                keep = tta_fuse(keep, sel, 0.6)
            rectl = {"x0": x0, "y0": y0, "cw": cw, "ch": ch}
            crop_img = img[y0:y1, x0:x1]
            s1m = (cv2.resize(s1crop, (SIDE, SIDE), interpolation=cv2.INTER_AREA) > 0.5)
            for k in keep:
                P = k["bin"]
                if CLOSE and large_box:
                    P = _close(P, CLOSE_K)
                m160 = (cv2.resize(P, (SIDE, SIDE),
                                   interpolation=cv2.INTER_NEAREST) > 0)
                gx, gy, gw, gh = cv2.boundingRect(P.astype(np.uint8))
                feats.append({
                    "s1_conf": float(sc1), "s2_conf": float(k["s2_conf"]),
                    "s1_area_ratio": float(s1m.sum()) / (SIDE * SIDE),
                    "s2_area_ratio": float(m160.sum()) / (SIDE * SIDE),
                    "s1s2_iou": float((s1m & m160).sum() / max((s1m | m160).sum(), 1)),
                    "bbox_area_ratio": (gw * gh) / float(TARGET * TARGET),
                    "bbox_aspect": float(min(gw, gh) / max(gw, gh)),
                    "verifier_prob": None})
                bins.append(cv2.resize(P.astype(np.uint8), (cw, ch),
                                       interpolation=cv2.INTER_NEAREST))
                rects.append(rectl)
                crop_local.append({"crop_img": crop_img, "mask160": m160})
        if feats and vb is not None:
            vp = verifier_probs(vb, crop_local)
            for i in range(len(feats)):
                feats[i]["verifier_prob"] = float(vp[i])
        out.append({"image_id": rec["image_id"], "H": H, "W": W, "gts": gts,
                    "feats": feats, "bins": bins, "rects": rects,
                    "has": rec["has"], "image_path": rec["image_path"]})
    return out


# ------------------------------ metrics (as fix4) ----------------------------
def fit_m3_on_val(imgs):
    X, y = [], []
    for im in imgs:
        H, W = im["H"], im["W"]
        for i, f in enumerate(im["feats"]):
            rect = im["rects"][i]
            P = np.zeros((H, W), np.uint8)
            P[rect["y0"]:rect["y0"] + rect["ch"], rect["x0"]:rect["x0"] + rect["cw"]] = im["bins"][i]
            best = 0.0
            for G in im["gts"]:
                inter = int((P & G["mask"]).sum()); uni = int((P | G["mask"]).sum())
                if uni:
                    best = max(best, inter / uni)
            X.append([f[c] for c in FCOLS]); y.append(int(best >= 0.5))
    X = np.array(X, float); y = np.array(y)
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(X), y)
    m3 = LogisticRegression(max_iter=1000, class_weight="balanced")
    m3.coef_, m3.intercept_, m3.classes_ = lr.coef_, lr.intercept_, np.array([0, 1])
    return m3, (sc.mean_, sc.scale_)


def apply_fused(im, m3, mu, sg):
    f = im["feats"]
    if f:
        X = np.array([[x[c] for c in FCOLS] for x in f], float)
        im["fused"] = m3.predict_proba((X - mu) / sg)[:, 1]
        im["s1c"] = np.array([x["s1_conf"] for x in f])
    else:
        im["fused"] = np.zeros(0); im["s1c"] = np.zeros(0)


def _select_ok(im):
    fused, s1c = im["fused"], im["s1c"]
    if len(fused) == 0:
        return np.zeros(0, bool)
    ar = np.array([f["s2_area_ratio"] for f in im["feats"]])
    return (s1c >= 0.02) & (fused >= np.where(ar >= LARGE_EDGE, THR_LARGE, THR_STD))


def _preds(im, ok):
    recs = []
    for i in np.argwhere(ok).ravel():
        rect = im["rects"][i]
        recs.append({"x0": rect["x0"], "y0": rect["y0"], "cw": rect["cw"],
                     "ch": rect["ch"], "bin": im["bins"][i],
                     "score": im["feats"][i]["s2_conf"]})
    return [rec_bin(k, im["H"], im["W"])
            for k in greedy_nms(recs, "mask_nms", NMS_IOU, im["H"], im["W"])] if recs else []


def _match_pairs(preds, gts):
    pairs = []
    for pi, P in enumerate(preds):
        for gi, G in enumerate(gts):
            inter = int((P & G["mask"]).sum()); uni = int((P | G["mask"]).sum())
            if uni and inter / uni >= IOU_MATCH:
                pairs.append((inter / uni, pi, gi))
    pairs.sort(reverse=True)
    used_p, used_g, matched = set(), set(), []
    for io, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi); used_g.add(gi); matched.append((pi, gi, io))
    return used_p, used_g, matched


def inst_metrics(imgs, mode):
    tp = fp = preds_tot = rescued = 0
    gtb = {b: 0 for b in BUCKETS}; tpb = {b: 0 for b in BUCKETS}
    td = []; fn_large = []
    for im in imgs:
        ok = _select_ok(im)
        ru = False
        if mode == "cascade" and not ok.any() and im.get("imgB", 0) >= TAU:
            ok = im["fused"] >= THR_RESCUE; ru = True
        rescued += int(ru)
        preds = _preds(im, ok)
        preds_tot += len(preds)
        used_p, used_g, matched = _match_pairs(preds, im["gts"])
        for gi, G in enumerate(im["gts"]):
            gtb[G["bucket"]] += 1
            if G["bucket"] == "large" and gi not in used_g:
                fn_large.append({"image_id": str(im["image_id"]), "px": int(G["px"])})
        for pi, gi, io in matched:
            tpb[im["gts"][gi]["bucket"]] += 1
            P, G = preds[pi], im["gts"][gi]
            inter = int((P & G["mask"]).sum())
            td.append(2 * inter / max(int(P.sum()) + int(G["mask"].sum()), 1))
        tp += len(used_g); fp += len(preds) - len(used_p)
    rec = tp / max(sum(gtb.values()), 1); prec = tp / max(preds_tot, 1)
    m = {"precision": round(prec, 4), "recall": round(rec, 4),
         "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
         "r_small": round(tpb["small"] / max(gtb["small"], 1), 4),
         "r_medium": round(tpb["medium"] / max(gtb["medium"], 1), 4),
         "r_large": round(tpb["large"] / max(gtb["large"], 1), 4),
         "tp_dice": round(float(np.mean(td)), 4) if td else 0.0,
         "fp_img": round(fp / len(imgs), 3),
         "pred_img": round(preds_tot / len(imgs), 3),
         "n_gt_large": int(gtb["large"]), "tp_gt_large": int(tpb["large"])}
    return m, rescued, fn_large


def image_metrics(imgs, ys, imgB, imgC, lr_img, thr_img):
    imgA = []
    for im in imgs:
        ok = _select_ok(im)
        if ok.any():
            imgA.append(float(im["fused"][ok].max()))
        elif im["imgB"] >= TAU and (im["fused"] >= THR_RESCUE).any():
            imgA.append(float(im["fused"][im["fused"] >= THR_RESCUE].max()))
        else:
            imgA.append(0.0)
    prob = lr_img.predict_proba(np.stack([np.array(imgA), imgB, imgC], 1))[:, 1]
    pred = (prob >= thr_img).astype(int)
    tp = int(((pred == 1) & (ys == 1)).sum()); fp_i = int(((pred == 1) & (ys == 0)).sum())
    fn_i = int(((pred == 0) & (ys == 1)).sum()); tn_i = int(((pred == 0) & (ys == 0)).sum())
    return {"threshold": thr_img, "accuracy": round((tp + tn_i) / len(ys), 4),
            "sensitivity": round(tp / max(tp + fn_i, 1), 4),
            "specificity": round(tn_i / max(tn_i + fp_i, 1), 4),
            "precision": round(tp / max(tp + fp_i, 1), 4),
            "f1": round(2 * tp / max(2 * tp + fp_i + fn_i, 1), 4),
            "roc_auc": round(float(roc_auc_score(ys, prob)), 4),
            "pr_auc": round(float(average_precision_score(ys, prob)), 4)}


def cmd_agg(names):
    cfgs = {"A": {"FB": 0, "CLOSE": 0}, "B": {"FB": 1, "CLOSE": 0},
            "C": {"FB": 1, "CLOSE": 1}, "D": {"FB": 0, "CLOSE": 1}}
    man = pd.read_csv(DATA / "split_manifest_v3.csv")
    cache = OUT / "ensemble_cache.npz"
    if cache.exists():
        z = np.load(cache); valB, valC, testB, testC = z["valB"], z["valC"], z["testB"], z["testC"]
        print("loaded ensemble cache", flush=True)
    else:
        valB = effnet_tta_probs(man, "val"); valC = yolocls_probs("val")
        testB = effnet_tta_probs(man, "test"); testC = yolocls_probs("test")
        np.savez(cache, valB=valB, valC=valC, testB=testB, testC=testC)
    ntest = len([i for i in man[man.split == "test"].image_id.astype(str)
                 if (CACHE / f"{i}.pkl").exists()])
    ys = np.array([int(bool(r.has_calculus)) for r in man[man.split == "test"].itertuples()
                   if (CACHE / f"{r.image_id}.pkl").exists()], float)
    testB = testB[:ntest]; testC = testC[:ntest]
    vb = make_verifier()
    all_res = {}; attrib = {}
    for name in names:
        cfg = cfgs[name]
        print(f"\n===== fix7 config {name}: FB={cfg['FB']} CLOSE={cfg['CLOSE']} =====", flush=True)
        val = aggregate(man, "val", vb, cfg)
        val_y = np.array([int(im["has"]) for im in val])
        m3, (mu, sg) = fit_m3_on_val(val)
        for im in val:
            apply_fused(im, m3, mu, sg)
        valA = np.array([float(im["fused"].max()) if len(im["fused"]) else 0.0 for im in val])
        vlen = len(valA)
        lr_img = LogisticRegression(max_iter=1000).fit(
            np.stack([valA, valB[:vlen], valC[:vlen]], 1), val_y)
        thr_img = 0.38
        test = aggregate(man, "test", vb, cfg)
        tb = testB[:len(test)]; tc = testC[:len(test)]; yy = ys[:len(test)]
        for im, b in zip(test, tb):
            im["imgB"] = float(b); apply_fused(im, m3, mu, sg)
        inst_n, _, _ = inst_metrics(test, "normal")
        inst_c, n_res, fn_large = inst_metrics(test, "cascade")
        img_m = image_metrics(test, yy, tb, tc, lr_img, thr_img)
        all_res[name] = {"config": {"FB": cfg["FB"], "CLOSE": cfg["CLOSE"]},
                         "instance_no_cascade": inst_n, "instance_cascade": inst_c,
                         "rescued_images": n_res, "image": img_m, "test_n": len(test)}
        attrib[name] = fn_large
        print(f"[{name}] cascade: P={inst_c['precision']} R={inst_c['recall']} "
              f"F1={inst_c['f1']} r_large={inst_c['r_large']} "
              f"(tp_large {inst_c['tp_gt_large']}/{inst_c['n_gt_large']}) "
              f"fp_img={inst_c['fp_img']} tp_dice={inst_c['tp_dice']}", flush=True)
        (OUT / "final_fix7_metrics.json").write_text(json.dumps(all_res, indent=2))
        (OUT / "fix7_large_fn_attribution.json").write_text(json.dumps(attrib, indent=2))
    print("\n---- parity vs fix4 (instance_cascade) ----", flush=True)
    print("fix4 ref: P=0.4242 R=0.4982 F1=0.4583 r_large=0.5 fp_img=0.271 tp_dice=0.817", flush=True)
    for name in names:
        c = all_res[name]["instance_cascade"]
        print(f"  {name}: P={c['precision']} R={c['recall']} F1={c['f1']} "
              f"r_large={c['r_large']} fp_img={c['fp_img']} tp_dice={c['tp_dice']}", flush=True)


def cmd_fitfb(out_npz: str) -> None:
    """Fit M3 on val with config B (FB on, CLOSE off) and write fusion npz
    (keys mu/sg/coef/intercept) for the promoted production artifacts."""
    man = pd.read_csv(DATA / "split_manifest_v3.csv")
    vb = make_verifier()
    val = aggregate(man, "val", vb, {"FB": 1, "CLOSE": 0})
    m3, (mu, sg) = fit_m3_on_val(val)
    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, mu=mu, sg=sg, coef=m3.coef_, intercept=m3.intercept_)
    print(f"wrote {out_npz}  (n_val_cands={sum(len(v['feats']) for v in val)})", flush=True)
    print("coef:", np.round(m3.coef_, 3).tolist(), "b0:",
          np.round(m3.intercept_, 3).tolist(), flush=True)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] == "status":
        cmd_status(); return
    if sys.argv[1] == "build":
        split = sys.argv[2]; max_n = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        cmd_build(split, max_n); return
    if sys.argv[1] == "fitfb":
        cmd_fitfb(sys.argv[2] if len(sys.argv) > 2
                  else str(OUT / "artifacts" / "fusion_v3.npz")); return
    if sys.argv[1] == "agg":
        names = (sys.argv[2].split(",") if len(sys.argv) > 2 else ["A", "B", "C", "D"])
        cmd_agg([n for n in names if n in "ABCD"]); return
    print(__doc__)


if __name__ == "__main__":
    main()
