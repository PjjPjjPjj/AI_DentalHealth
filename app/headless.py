"""Headless end-to-end runner (CLI / tests): image -> detection -> risk ->
visualization -> HTML/PDF report. The GUI calls the same functions."""
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

ROOT = Path(__file__).resolve().parents[1]


def outputs_root() -> Path:
    """Where results/ live: next to the exe when frozen, repo root in dev."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return ROOT


def run_image(image_path: str | Path, out_dir: str | Path | None = None,
              models_dir: str | Path | None = None,
              save_report: bool = True, fast_mode: bool = False) -> dict:
    import cv2

    from app.core.pipeline import DentalPipeline
    from app.core.risk import assess
    from app.core.visualize import annotate, collect_lesion_masks

    image_path = Path(image_path)
    out_dir = Path(out_dir) if out_dir else outputs_root() / "results" / "result_image"
    pipe = DentalPipeline(models_dir=models_dir, fast_mode=fast_mode)
    result = pipe.run(image_path)
    risk = assess(result)
    result["risk"] = risk

    img = cv2.imread(str(image_path))
    masks = collect_lesion_masks(result, img.shape[:2])
    vis = annotate(img, result, show_masks=True, masks=masks)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    vis_path = out_dir / f"{stem}_detected.jpg"
    cv2.imwrite(str(vis_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])

    artifacts = {"image": str(vis_path)}
    if save_report:
        from app.core.report import save_results as _save

        rep_dir = outputs_root() / "results" / "reports"
        arts = _save(image_path, vis_path, result, risk, rep_dir, stem)
        artifacts.update({k: str(v) for k, v in arts.items()})
    return {"result": result, "risk": risk, "artifacts": artifacts}


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="DentalAI headless runner")
    ap.add_argument("image", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fast", action="store_true",
                    help="skip TTA/tiling in the calculus cascade (faster CPU)")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()
    for p in args.image:
        out = run_image(p, args.out, save_report=not args.no_report,
                        fast_mode=args.fast)
        r, risk = out["result"], out["risk"]
        print(f"== {Path(p).name} ==")
        print(f"  teeth={r['summary']['tooth_count']} "
              f"plaque={risk['plaque']['count']} "
              f"calculus={risk['calculus']['count']} "
              f"caries={risk['caries']['count']}")
        for it in risk["items"]:
            print(f"  [{it['lesion']}] " + "；".join(it["advice"][:2]))
        print(f"  annotated: {out['artifacts']['image']}")
        if "pdf" in out["artifacts"]:
            print(f"  report:    {out['artifacts']['pdf']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
