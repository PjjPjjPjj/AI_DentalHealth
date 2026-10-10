"""Review queue tooling: list pending / dump statistics.

Usage:
    python -m src.annotation.review_queue --action list
    python -m src.annotation.review_queue --action stats
    python -m src.annotation.review_queue --action approve --image IMG --candidate 0 --class tooth
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

from src.utils.paths import ANNOTATION_QUEUE_DIR


def _entries_for(image_id: str) -> tuple[Path, list[dict]]:
    p = ANNOTATION_QUEUE_DIR / image_id / "queue.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return p, json.loads(p.read_text())


def approve(image_id: str, candidate_id: int, cls: str) -> None:
    p, entries = _entries_for(image_id)
    if candidate_id >= len(entries):
        print(f"candidate {candidate_id} not in queue ({len(entries)} entries)")
        sys.exit(1)
    entries[candidate_id]["status"] = "approved"
    entries[candidate_id]["approved_class"] = cls
    entries[candidate_id]["approved_by"] = "review_cli"
    p.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    print(f"Approved candidate {candidate_id} as {cls}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["list", "stats", "approve"], default="list")
    parser.add_argument("--image", default=None)
    parser.add_argument("--candidate", type=int, default=0)
    parser.add_argument("--class", dest="cls", default="tooth")
    args = parser.parse_args()

    qdir = ANNOTATION_QUEUE_DIR
    if args.action == "approve":
        approve(args.image, args.candidate, args.cls)
        return
    print(json.dumps({
        "queue_dir": str(qdir),
        "images": [str(p.name) for p in qdir.glob("*/queue.json")]
        if qdir.exists() else [],
    }, indent=2))


if __name__ == "__main__":
    main()
