"""Full-pipeline step-by-step demo (tooth + caries + calculus + plaque).

Steps: 0 input -> 1 quality -> 2 tooth seg -> 3 caries -> 3.5 calculus ->
4 plaque -> 5 association -> 6 final JSON + panel figure.
Artifacts: outputs/demo/pipeline_walkthrough/
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, f"{_ROOT}")
from src.inference.associate import associate_calculus, associate_caries  # noqa: E402
from src.inference.image_quality import check_quality  # noqa: E402
from src.inference.models import (CalculusCascadeModel, CariesModel,  # noqa: E402
                                  PlaqueModel, ToothModel)
from src.utils.config import load_yaml  # noqa: E402

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

ROOT = Path(f"{_ROOT}")
OUT = ROOT / "outputs/demo/pipeline_walkthrough"
OUT.mkdir(parents=True, exist_ok=True)

IMAGE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "demo_images/intraoral_real.jpg"


def log(step, msg):
    print(f"\n===== STEP {step}: {msg}", flush=True)


def main() -> None:
    cfg = load_yaml(ROOT / "configs/pipeline.yaml")
    icfg = cfg["inference"]

    log(0, "输入图像")
    img = cv2.imread(str(IMAGE))
    H, W = img.shape[:2]
    cv2.imwrite(str(OUT / "step0_input.jpg"), img)
    print(f"input: {IMAGE.name}  {W}x{H}x3 BGR uint8")

    log(1, "质量检查 check_quality(img, quality_cfg)")
    q = check_quality(img, cfg["quality"])
    (OUT / "step1_quality.json").write_text(json.dumps(q, indent=2))
    print(json.dumps(q, indent=2))

    log(2, f"牙齿实例分割 ToothModel({icfg['tooth']['imgsz']}px, conf {icfg['tooth']['conf']})")
    tm = ToothModel(weights=str(ROOT / cfg["weights"]["tooth"]), **icfg["tooth"])
    teeth = tm.predict(img)
    print(f"input: BGR image -> output: {len(teeth)} tooth instances")
    vis_tooth = img.copy()
    rng = np.random.default_rng(42)
    for i, t in enumerate(teeth):
        c = tuple(int(v) for v in rng.integers(60, 255, 3))
        cnts, _ = cv2.findContours(t["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_tooth, cnts, -1, c, 2)
        x0, y0 = int(t["bbox"][0]), int(t["bbox"][1])
        cv2.putText(vis_tooth, f"T{i+1} {t['conf']:.2f}", (x0, max(12, y0 - 4)), 0, 0.5, c, 1)
    cv2.imwrite(str(OUT / "step2_tooth.jpg"), vis_tooth)
    stats = [{"tooth_id": i + 1, "bbox": [round(v, 1) for v in t["bbox"]],
              "conf": round(t["conf"], 3)} for i, t in enumerate(teeth)]
    (OUT / "step2_tooth_instances.json").write_text(json.dumps(stats, indent=2))
    print(f"teeth: {len(stats)} instances")

    log(3, f"龋齿检测 CariesModel(整图一次, {icfg['caries']['imgsz']}px, conf {icfg['caries']['conf']})")
    cm = CariesModel(weights=str(ROOT / cfg["weights"]["caries"]), **icfg["caries"])
    dets = cm.predict(img)
    vis_caries = img.copy()
    for d in dets:
        x0, y0, x1, y1 = [int(v) for v in d["bbox"]]
        cv2.rectangle(vis_caries, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(vis_caries, f"caries {d['conf']:.2f}", (x0, max(14, y0 - 5)), 0, 0.55, (0, 0, 255), 2)
    cv2.imwrite(str(OUT / "step3_caries.jpg"), vis_caries)
    det_json = [{"bbox": [round(v, 1) for v in d["bbox"]], "conf": round(d["conf"], 3)} for d in dets]
    (OUT / "step3_caries_dets.json").write_text(json.dumps(det_json, indent=2))
    print(f"output: {len(dets)} caries detections: {det_json}")

    log(3.5, "牙结石级联 CalculusCascadeModel(S1分块 -> S2自适应窗口 -> Verifier -> M3融合 -> 尺寸阈值+TTA)")
    calc = CalculusCascadeModel(**icfg["calculus"], **cfg["weights"]["calculus"])
    calc_insts = calc.predict(str(IMAGE))
    print(f"input: BGR image -> output: {len(calc_insts)} calculus instances")
    vis_calc = img.copy()
    for i, c in enumerate(calc_insts):
        cnts, _ = cv2.findContours(c["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_calc, cnts, -1, (0, 165, 255), 2)
        ys, xs = np.where(c["mask"] > 0)
        if len(xs):
            cv2.putText(vis_calc, f"calc {c['conf']:.2f}", (int(xs.min()), max(12, int(ys.min()) - 4)),
                        0, 0.5, (0, 165, 255), 1)
    cv2.imwrite(str(OUT / "step35_calculus.jpg"), vis_calc)
    calc_json = [{"inst_id": i + 1, "conf_fused": round(c["conf"], 3),
                  "area_ratio": round(c["area_ratio"], 5),
                  "px": int(c["mask"].sum())} for i, c in enumerate(calc_insts)]
    (OUT / "step35_calculus.json").write_text(json.dumps(calc_json, indent=2))
    print(json.dumps(calc_json, indent=2))

    log(4, f"牙菌斑逐牙分级 PlaqueModel(crop {icfg['plaque']['imgsz']}px, thr {icfg['plaque']['conf_threshold']})")
    pm = PlaqueModel(weights=str(ROOT / cfg["weights"]["plaque"]), **icfg["plaque"])
    plaque = []
    vis_plaque = img.copy()
    thr = float(icfg["plaque"]["conf_threshold"])
    for i, t in enumerate(teeth):
        x0, y0, x1, y1 = [int(max(0, v)) for v in t["bbox"]]
        x1, y1 = min(W, x1), min(H, y1)
        crop = img[y0:y1, x0:x1]
        prob = round(pm.predict_proba(crop), 4) if crop.size else None
        plaque.append({"tooth_id": i + 1, "p_plaque": prob,
                       "present": bool(prob is not None and prob >= thr)})
        if prob is not None:
            c = (0, 0, 255) if prob >= thr else (0, 200, 0)
            cv2.rectangle(vis_plaque, (x0, y0), (x1, y1), c, 2)
            cv2.putText(vis_plaque, f"p{prob:.2f}", (x0, max(14, y1 + 16)), 0, 0.5, c, 2)
    cv2.imwrite(str(OUT / "step4_plaque.jpg"), vis_plaque)
    (OUT / "step4_plaque.json").write_text(json.dumps(plaque, indent=2))
    print(f"plaque present: {sum(1 for p in plaque if p['present'])}/{len(plaque)}")

    log(5, "病灶-牙齿关联 associate_caries / associate_calculus")
    assoc_out = []
    tooth_masks = [t["mask"] for t in teeth]
    for d in dets:
        a = associate_caries(d["bbox"], tooth_masks, center_rule=True,
                             min_iou=float(cfg["pipeline"]["association"]["caries_min_iou"]))
        assoc_out.append({"type": "caries", "assigned": bool(a.get("assigned")),
                          "tooth": a.get("tooth", -1) + 1 if a.get("assigned") else None})
    for i, c in enumerate(calc_insts):
        a = associate_calculus(c["mask"], tooth_masks,
                               min_overlap=float(cfg["pipeline"]["association"]["calculus_min_overlap"]))
        assoc_out.append({"type": "calculus", "inst_id": i + 1,
                          "assigned": bool(a.get("assigned")),
                          "tooth": a.get("tooth", -1) + 1 if a.get("assigned") else None,
                          "overlap_ratio": round(a.get("overlap_ratio", 0), 3)})
    (OUT / "step5_association.json").write_text(json.dumps(assoc_out, indent=2))
    print(json.dumps(assoc_out, indent=2))

    log(6, "统一 JSON + 六面板汇总图")
    result = {"image_id": IMAGE.name, "image_size": {"width": W, "height": H},
              "quality": q, "teeth": [], "unassigned_lesions": []}
    for i, t in enumerate(teeth):
        result["teeth"].append({
            "tooth_id": i + 1, "bbox": [round(float(v), 1) for v in t["bbox"]], "mask": None,
            "plaque": {"score": plaque[i]["p_plaque"], "present": plaque[i]["present"],
                       "confidence": plaque[i]["p_plaque"]},
            "calculus": {"present": False, "confidence": 0.0, "area_ratio": 0.0},
            "caries": {"present": False, "confidence": 0.0}})
    for d, a in zip(dets, assoc_out):
        if a["type"] != "caries":
            continue
        entry = {"type": "caries", "confidence": round(d["conf"], 3),
                 "bbox": [round(v, 1) for v in d["bbox"]]}
        if a["assigned"]:
            result["teeth"][a["tooth"] - 1]["caries"] = {"present": True, "confidence": round(d["conf"], 3)}
        else:
            entry["reason"] = "unassigned"
            result["unassigned_lesions"].append(entry)
    for c, a in zip(calc_insts, assoc_out):
        if a["type"] != "calculus":
            continue
        entry = {"type": "calculus", "confidence": round(c["conf"], 3),
                 "area_ratio": round(c["area_ratio"], 5)}
        if a["assigned"]:
            t = result["teeth"][a["tooth"] - 1]["calculus"]
            t["present"] = True
            t["confidence"] = round(c["conf"], 3)
            t["area_ratio"] = round(a.get("overlap_ratio", 0), 3)
        else:
            entry["reason"] = "below_min_overlap"
            result["unassigned_lesions"].append(entry)
    (OUT / "step6_final_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

    # final overlay
    vis_final = cv2.addWeighted(vis_tooth, 0.45, vis_plaque, 0.35, 0)
    for c in calc_insts:
        cnts, _ = cv2.findContours(c["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_final, cnts, -1, (0, 165, 255), 2)
    for d, a in zip(dets, [x for x in assoc_out if x["type"] == "caries"]):
        x0, y0, x1, y1 = [int(v) for v in d["bbox"]]
        cv2.rectangle(vis_final, (x0, y0), (x1, y1), (0, 0, 255), 2)
        tag = f"caries {d['conf']:.2f}" + (f" -> T{a['tooth']}" if a["assigned"] else "")
        cv2.putText(vis_final, tag, (x0, max(14, y0 - 6)), 0, 0.55, (0, 0, 255), 2)
    cv2.imwrite(str(OUT / "step6_final.jpg"), vis_final)

    # panel figure
    try:
        make_panel_figure(OUT, IMAGE.name)
    except Exception as exc:
        print("panel figure failed:", exc)

    n_calc_a = sum(1 for a in assoc_out if a["type"] == "calculus" and a["assigned"])
    print(f"teeth={len(teeth)} caries={len(dets)}(assigned {sum(1 for a in assoc_out if a['type']=='caries' and a['assigned'])}) "
          f"calculus={len(calc_insts)}(assigned {n_calc_a}) plaque_present={sum(1 for p in plaque if p['present'])}/{len(plaque)}")


def make_panel_figure(out_dir: Path, img_name: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    fm.fontManager.addfont(str(ROOT / "assets/fonts/NotoSansSC.otf"))
    plt.rcParams["font.family"] = "Noto Sans CJK SC"
    plt.rcParams["axes.unicode_minus"] = False

    panels = [("输入图像", "step0_input.jpg"), ("① 牙齿实例分割", "step2_tooth.jpg"),
              ("② 龋齿检测", "step3_caries.jpg"), ("③ 牙结石级联检测", "step35_calculus.jpg"),
              ("④ 牙菌斑分级", "step4_plaque.jpg"), ("最终结果（含牙位归属）", "step6_final.jpg")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), dpi=150)
    for ax, (title, fn) in zip(axes.ravel(), panels):
        im = cv2.imread(str(out_dir / fn))
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=15)
        ax.axis("off")
    fig.suptitle(f"口腔病灶智能检测完整流程 Demo — {img_name}", fontsize=17)
    fig.tight_layout()
    fig.savefig(out_dir / "pipeline_panels.jpg", bbox_inches="tight")
    plt.close(fig)
    print("panel figure saved:", out_dir / "pipeline_panels.jpg")


if __name__ == "__main__":
    main()
