"""Parallel chunked downloader for Zenodo Dataset.zip (16 parts, resumable)."""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
import os
import subprocess
import sys
import time
from pathlib import Path

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

URL = "https://zenodo.org/api/records/14827784/files/Dataset.zip/content"
OUT = Path(f"{_ROOT}/datasets/raw/caries6313/Dataset.zip")
TOTAL = 1576115314
NPART = 16
PART_DIR = Path(f"{_ROOT}/datasets/raw/caries6313/parts")
PART_DIR.mkdir(exist_ok=True)

chunk = TOTAL // NPART
ranges = [(i * chunk, (TOTAL - 1) if i == NPART - 1 else (i + 1) * chunk - 1)
          for i in range(NPART)]


def part_file(i):
    return PART_DIR / f"part_{i:02d}"


def download_part(i, s, e):
    pf = part_file(i)
    expected = e - s + 1
    while True:
        cur = pf.stat().st_size if pf.exists() else 0
        if cur >= expected:
            return i
        rs = s + cur
        cmd = ["curl", "-sL", "-r", f"{rs}-{e}", "--max-time", "1800",
               "-o", str(pf) + (".tmp" if cur == 0 else ".add"), URL]
        subprocess.run(cmd)
        add = Path(str(pf) + ".add")
        if add.exists() and add.stat().st_size > 0:
            with open(pf, "ab") as out_f, open(add, "rb") as in_f:
                out_f.write(in_f.read())
            add.unlink()
        time.sleep(2)


def main():
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(16) as ex:
        futs = [ex.submit(download_part, i, s, e)
                for i, (s, e) in enumerate(ranges)]
        done = [f.result() for f in futs]
    print("parts done:", sorted(done))
    # assemble
    with open(OUT, "wb") as out_f:
        for i in range(NPART):
            with open(part_file(i), "rb") as in_f:
                while True:
                    b = in_f.read(1 << 22)
                    if not b:
                        break
                    out_f.write(b)
    sz = OUT.stat().st_size
    print("assembled:", sz, "expected:", TOTAL, "match:", sz == TOTAL)


if __name__ == "__main__":
    main()
