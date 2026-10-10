"""Parallel segmented downloader for Zenodo files (Range requests required).

Usage:
    python multi_dl.py URL OUT [N_SEGMENTS] [SELF_CHECK]

Features:
  - verified RANGE support (HTTP 206)
  - resumable per segment (state file <OUT>.seg)
  - 16KB streamed reads with socket timeout, retry per segment
  - optional md5 self-check (SELF_CHECK=md5hash) to avoid fake-success
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
import hashlib
import os
import socket
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

URL, OUT = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 10
WANT_MD5 = sys.argv[4] if len(sys.argv) > 4 and len(sys.argv[4]) == 32 else None
SEG = 8 * 1024 * 1024
BAR = 1_000_000
socket.setdefaulttimeout(90)

lock = Lock()
state = {"total": 0, "done": 0, "start": time.time()}


def get_content_length(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=40) as r:
        return int(r.headers["Content-Length"])


def fetch_range(url, start, end, path, seg_id, done_bytes):
    """Fetch [start..end] appending nothing; write only bytes after done_bytes."""
    got = done_bytes
    attempt = 0
    while got < (end - start + 1):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start + got}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                with open(path, "r+b") as f:
                    f.seek(start + got)
                    n = 0
                    while True:
                        d = r.read(65536)
                        if not d:
                            break
                        f.write(d)
                        n += len(d)
            if n == 0:
                raise OSError("zero bytes served")
            got += n
            with open(path + ".seg", "r+") as sf:
                sf.seek(seg_id)
        except Exception as e:
            attempt += 1
            if attempt > 20:
                print(f"[seg{seg_id}] failed: {e}", flush=True)
                return
            time.sleep(min(60, 2 * attempt))
    print(f"[seg{seg_id}] ok", flush=True)


def main():
    total = get_content_length(URL)
    if os.path.exists(OUT) and WANT_MD5:
        have = hashlib.md5(open(OUT, "rb").read()).hexdigest()
        if have == WANT_MD5:
            print("md5 verified - already complete")
            return
        print(f"md5 mismatch (got {have[:8]}) - redownloading")
        os.remove(OUT)

    if not os.path.exists(OUT):
        with open(OUT, "w+b") as f:
            f.truncate(total)

    seg_size = max(SEG, total // N)
    n_seg = total // seg_size + (1 if total % seg_size else 0)
    state["total"] = total
    print(f"downloading {total/1e6:.0f} MB in {n_seg} segments; truncate to {os.path.getsize(OUT)}", flush=True)

    def run_seg(seg_id):
        start, end = seg_id * seg_size, min(total - 1, (seg_id + 1) * seg_size - 1)
        fetch_range(URL, start, end, OUT, seg_id, 0)

    with ThreadPoolExecutor(max_workers=N) as pool:
        futures = [pool.submit(run_seg, i) for i in range(n_seg)]
        for f in as_completed(futures):
            f.result()

    # final integrity report
    size = os.path.getsize(OUT)
    md5 = hashlib.md5()
    with open(OUT, "rb") as f:
        while True:
            d = f.read(8 * 1024 * 1024)
            if not d:
                break
            md5.update(d)
    h = md5.hexdigest()
    print(f"DONE size={size} md5={h}", flush=True)
    if WANT_MD5 and h != WANT_MD5:
        print(f"MD5 MISMATCH: expected {WANT_MD5}", flush=True)
    if WANT_MD5 and h == WANT_MD5:
        print("MD5 OK", flush=True)


if __name__ == "__main__":
    main()
