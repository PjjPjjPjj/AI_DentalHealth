# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
import os, re, collections
import numpy as np
from PIL import Image

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

ROOT = f'{_ROOT}/datasets/raw/oral_detector/extracted'
MAMBA = f'{_ROOT}/datasets/raw/oral_mamba/images'
SPLITS = ['train', 'valid']

def md5(p):
    h = hashlib_md5()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def hashlib_md5():
    import hashlib
    return hashlib.md5()

def ahash(p, size=8):
    try:
        with Image.open(p) as im:
            g = im.convert('L').resize((size + 1, size), Image.LANCZOS)
            a = np.asarray(g, dtype=np.float32)
    except Exception:
        return None
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return np.packbits(bits).tobytes()

keys, hashes = [], []
for s in SPLITS:
    for f in sorted(os.listdir(os.path.join(ROOT, s, 'images'))):
        h = ahash(os.path.join(ROOT, s, 'images', f))
        if h is not None:
            keys.append((s, f))
            hashes.append(np.frombuffer(h, dtype=np.uint8))
bits = np.unpackbits(np.stack(hashes), axis=1).astype(np.int8)
n = len(keys)
print(f'计算 {n} 张图的 ahash 完成')

hist = collections.Counter()
pairs = []
BATCH = 256
for i0 in range(0, n, BATCH):
    d = (bits[i0:i0 + BATCH, None, :] != bits[None, :, :]).sum(-1)
    mask = np.triu(np.ones_like(d, dtype=bool), k=1)
    sel = mask & (d <= 4)
    for a, b in np.argwhere(sel):
        pairs.append((keys[i0 + a], keys[b], int(d[a, b])))
        hist[int(d[a, b])] += 1
print('汉明距离分布 (d<=4):', dict(sorted(hist.items())))
cross = [(k1, k2, d) for k1, k2, d in pairs if k1[0] != k2[0]]
same = [(k1, k2, d) for k1, k2, d in pairs if k1[0] == k2[0]]
print(f'\n近重复对 (dist<=4): 总 {len(pairs)}, 同split {len(same)}, 跨split {len(cross)}')
print('\n-- 跨split近重复全部列表 --')
for k1, k2, d in sorted(cross, key=lambda x: x[2]):
    print(f'  d={d}: {k1[1]} <-> {k2[1]}')

# same-base groups: split composition
base_groups = collections.defaultdict(list)
for s in SPLITS:
    for f in sorted(os.listdir(os.path.join(ROOT, s, 'images'))):
        base = re.sub(r'\.rf\.[A-Za-z0-9]+$', '', os.path.splitext(f)[0])
        base_groups[base].append((s, f))
multi = {b: v for b, v in base_groups.items() if len(v) > 1}
cross_base = {b: v for b, v in multi.items() if len({s for s, _ in v}) > 1}
print(f'\n同源名多副本组: {len(multi)} 组; 其中跨 train/valid: {len(cross_base)} 组')
for b, v in list(cross_base.items())[:10]:
    print(f'  {b}: {v}')
# ahash distance within same-base groups
print('\n-- 同源名组内部 ahash 距离 (判断是否同图变体) --')
def find_key(sk, f):
    return (sk, f)
dists_in_group = []
for b, v in multi.items():
    idx = []
    for sk, f in v:
        idx.append(keys.index((sk, f)))
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            dd = int((bits[idx[i]] != bits[idx[j]]).sum())
            dists_in_group.append((b, v[i], v[j], dd))
dd_hist = collections.Counter(min(x[3], 30) for x in dists_in_group)
print('组内距离分布(>=30归并为30):', dict(sorted(dd_hist.items())))
print('距离最小的12组:')
for b, v1, v2, dd in sorted(dists_in_group, key=lambda x: x[3])[:12]:
    print(f'  d={dd}: {b}: {v1} <-> {v2}')

# oral_mamba overlap details
print('\n-- oral_mamba 重叠细节 --')
mamba_files = os.listdir(MAMBA)
pat = re.compile(r'^(gingivitis_)?\(?(\d+)\)?\.(jpg|jpeg|png)$', re.I)
mcat = collections.Counter()
for f in mamba_files:
    m = pat.match(f)
    if m:
        mcat['gingivitis_' if m.group(1) else 'plain'] += 1
    else:
        mcat['other'] += 1
print('oral_mamba 命名分布:', dict(mcat))
det_md5 = {}
for s in SPLITS:
    for f in sorted(os.listdir(os.path.join(ROOT, s, 'images'))):
        det_md5[md5(os.path.join(ROOT, s, 'images', f))] = (s, f)
m2d = collections.defaultdict(list)
det_hit = set()
for f in mamba_files:
    h = md5(os.path.join(MAMBA, f))
    if h in det_md5:
        m2d[f].append(det_md5[h])
        det_hit.add(det_md5[h])
print(f'oral_mamba 中与 oral_detector 内容相同的文件: {len(m2d)} / {len(mamba_files)}')
print(f'oral_detector 中与 oral_mamba 内容相同的文件: {len(det_hit)} / {n}')
mc2 = collections.Counter()
for f in m2d:
    m = pat.match(f)
    mc2['gingivitis_' if m and m.group(1) else ('plain' if m else 'other')] += 1
print('重叠的 mamba 文件命名分布:', dict(mc2))
dsplit = collections.Counter(s for (s, f) in det_hit)
print('重叠的 detector 文件split分布:', dict(dsplit))
