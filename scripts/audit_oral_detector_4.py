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
keys, hashes = [], []
for s, f, p in files:
    keys.append((s, f))
    hashes.append(np.frombuffer(ahash(p), dtype=np.uint8))
bits = np.unpackbits(np.stack(hashes), axis=1).astype(np.int8)
n = len(keys)
# correct full-pair distance matrix (n=1404 -> 1404x1404 fine)
D = (bits[:, None, :] != bits[None, :, :]).sum(-1)
iu = np.triu_indices(n, k=1)
dvals = D[iu]
print('汉明距离分布 (<=8):', {d: int((dvals == d).sum()) for d in range(9)})
pairs = [(keys[iu[0][i]], keys[iu[1][i]], int(dvals[i])) for i in np.where(dvals <= 4)[0]]
cross = [(k1, k2, d) for k1, k2, d in pairs if k1[0] != k2[0]]
same = [(k1, k2, d) for k1, k2, d in pairs if k1[0] == k2[0]]
print(f'\n近重复对 (d<=4): 总 {len(pairs)}, 同split {len(same)}, 跨split {len(cross)}')
print('\n同split近重复 (d<=4):')
for k1, k2, d in sorted(same, key=lambda x: x[2]):
    print(f'  d={d}: {k1[1][:44]} <-> {k2[1][:44]}')
print('\n跨split近重复像素验证:')
for k1, k2, d in sorted(cross, key=lambda x: x[2]):
    p1 = os.path.join(ROOT, k1[0], 'images', k1[1])
    p2 = os.path.join(ROOT, k2[0], 'images', k2[1])
    with Image.open(p1) as a, Image.open(p2) as b:
        aa = np.asarray(a.convert('RGB'), dtype=np.int16)
        bb = np.asarray(b.convert('RGB'), dtype=np.int16)
    if aa.shape == bb.shape:
        eq = np.array_equal(aa, bb)
        diff = float(np.abs(aa - bb).mean())
        note = f'同尺寸 meanAbsDiff={diff:.2f} pixel_equal={eq}'
    else:
        note = f'尺寸不同 {aa.shape[1]}x{aa.shape[0]} vs {bb.shape[1]}x{bb.shape[0]}'
    print(f'  d={d}: {k1[1][:30]} <-> {k2[1][:30]} | {note}')
