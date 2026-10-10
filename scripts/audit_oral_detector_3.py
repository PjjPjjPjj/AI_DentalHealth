# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
import os, collections
import numpy as np
from PIL import Image

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

ROOT = f'{_ROOT}/datasets/raw/oral_detector/extracted'
SPLITS = ['train', 'valid']

def ahash(p, size=8):
    with Image.open(p) as im:
        g = im.convert('L').resize((size + 1, size), Image.LANCZOS)
        a = np.asarray(g, dtype=np.float32)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return np.packbits(bits).tobytes()

files = [(s, f, os.path.join(ROOT, s, 'images', f)) for s in SPLITS
         for f in sorted(os.listdir(os.path.join(ROOT, s, 'images')))]
buckets = collections.defaultdict(list)
for s, f, p in files:
    buckets[ahash(p)].append((s, f, p))
multi = {h: v for h, v in buckets.items() if len(v) > 1}
npairs = sum(len(v) * (len(v) - 1) // 2 for v in multi.values())
print(f'ahash 桶: {len(buckets)}, 非唯一桶: {len(multi)}, 组成对数: {npairs}')
size_dist = collections.Counter(len(v) for v in multi.values())
print('桶大小分布:', dict(sorted(size_dist.items())))

# pixel-level verification on a sample
print('\n像素级验证 (每桶取前2张):')
print(f'{"bucket":8s} {"dims":>12s} {"bytesA":>9s} {"bytesB":>9s} {"pixel_equal":>11s} {"meanAbsDiff":>11s}')
verified_equal = 0
verified_total = 0
for h, v in list(multi.items())[:40]:
    (s1, f1, p1), (s2, f2, p2) = v[0], v[1]
    with Image.open(p1) as a, Image.open(p2) as b:
        aa = np.asarray(a.convert('RGB'), dtype=np.int16)
        bb = np.asarray(b.convert('RGB'), dtype=np.int16)
    if aa.shape == bb.shape:
        eq = np.array_equal(aa, bb)
        diff = float(np.abs(aa - bb).mean())
    else:
        eq = False
        diff = -1
    verified_total += 1
    verified_equal += eq
    print(f'{len(v):8d} {str(aa.shape[1]) + "x" + str(aa.shape[0]):>12s} {os.path.getsize(p1):9d} {os.path.getsize(p2):9d} {str(eq):>11s} {diff:11.3f}')
print(f'\n抽样验证: {verified_equal}/{verified_total} 像素完全一致')

# dims of d=0 bucket members overall
print('\n非唯一桶成员尺寸示例(前10桶全部成员):')
for h, v in list(multi.items())[:10]:
    info = []
    for s, f, p in v:
        with Image.open(p) as im:
            info.append(f'{f[:28]}({im.size[0]}x{im.size[1]},{os.path.getsize(p)//1024}KB)')
    print('  ' + ' | '.join(info))
