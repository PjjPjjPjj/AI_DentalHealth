"""Inference worker thread (keeps the GUI responsive)."""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

from pathlib import Path

from app.qt_compat import QtCore, Signal


class InferenceWorker(QtCore.QThread):
    progress = Signal(str)
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, image_path: str, parent=None, fast_mode: bool = False,
                 save_report: bool = True):
        super().__init__(parent)
        self.image_path = str(image_path)
        self.fast_mode = fast_mode
        self.save_report = save_report

    def run(self) -> None:
        try:
            from app.headless import run_image

            self.progress.emit("正在加载模型与推理…")
            out = run_image(self.image_path,
                            save_report=self.save_report,
                            fast_mode=self.fast_mode)
            self.progress.emit("完成")
            self.finished_ok.emit(out)
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            self.failed.emit(str(exc))


class BatchDemoWorker(QtCore.QThread):
    """Runs a sequence of demo cases, emitting each step for the GUI."""

    step = Signal(str, dict)   # case_name, payload(result+risk+artifacts)
    progress = Signal(str)
    failed = Signal(str)
    all_done = Signal()

    def __init__(self, cases: list[tuple[str, str]], parent=None,
                 fast_mode: bool = False):
        super().__init__(parent)
        self.cases = cases
        self.fast_mode = fast_mode

    def run(self) -> None:
        try:
            from app.headless import run_image

            pipe = None
            for name, path in self.cases:
                if not Path(path).exists():
                    continue
                self.progress.emit(f"运行案例 {name}: {Path(path).name}")
                from app.core.pipeline import DentalPipeline
                from app.core.risk import assess
                from app.core.visualize import annotate, collect_lesion_masks
                import cv2

                if pipe is None:
                    pipe = DentalPipeline(fast_mode=self.fast_mode)
                result = pipe.run(path)
                risk = assess(result)
                result["risk"] = risk
                img = cv2.imread(path)
                masks = collect_lesion_masks(result, img.shape[:2])
                vis = annotate(img, result, show_masks=True, masks=masks)
                import sys

                root = (Path(sys.executable).resolve().parent
                        if getattr(sys, "frozen", False)
                        else Path(__file__).resolve().parents[2])
                out_dir = root / "results" / "demo"
                out_dir.mkdir(parents=True, exist_ok=True)
                vis_path = out_dir / f"{Path(path).stem}_detected.jpg"
                cv2.imwrite(str(vis_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
                self.step.emit(name, {
                    "image": str(path), "annotated": str(vis_path),
                    "result": result, "risk": risk})
            self.all_done.emit()
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            self.failed.emit(str(exc))
