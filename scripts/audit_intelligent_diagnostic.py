# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
import os, collections, json
import numpy as np
from PIL import Image
import hashlib

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

SRC = f'{_ROOT}/datasets/raw/Intelligent_diagnostic_dataset/extracted/Intelligent diagnostic data set'
MAMBA_I = f'{_ROOT}/datasets/raw/oral_mamba/images'
MAMBA_L = f'{_ROOT}/datasets/raw/oral_mamba/lables'
DET = f'{_ROOT}/datasets/raw/oral_detector/extracted'

def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()

def ahash(p, size=8):
    try:
        with Image.open(p) as im:
            g = im.convert('L').resize((size + 1, size), Image.LANCZOS)
            a = np.asarray(g, dtype=np.float32)
    except Exception:
        return None
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return np.packbits(bits).tobytes()

imgs = sorted(os.listdir(os.path.join(SRC, 'images')))
labs = sorted(os.listdir(os.path.join(SRC, 'lables')))
print(f'images: {len(imgs)}, lables: {len(labs)}')

# name categories
def cat(f):
    if f.startswith('caries_0_'): return 'caries_0_*'
    if f.startswith('gingivitis_'): return 'gingivitis_*'
    return 'plain'
icat = collections.Counter(cat(f) for f in imgs)
lcat = collections.Counter(os.path.splitext(f)[1] for f in labs)
stem_map = {os.path.splitext(f)[0]: f for f in labs}
print('image 命名分布:', dict(icat))
print('label 扩展名分布:', dict(lcat))
missing_lab = [f for f in imgs if os.path.splitext(f)[0] not in stem_map]
print('无对应 label 的 images:', len(missing_lab), missing_lab[:10])
unmatched_lab = [f for f in labs if os.path.splitext(f)[0] not in {os.path.splitext(x)[0] for x in imgs}]
print('无对应 image 的 labels:', len(unmatched_lab), unmatched_lab[:10])

# label content: mask or yolo txt? it's png -> check a few
print('\nlabel png 示例:')
for f in labs[:3]:
    p = os.path.join(SRC, 'lables', f)
    with Image.open(p) as im:
        a = np.asarray(im)
    print(f'  {f}: mode={im.mode} size={im.size} dtype={a.dtype} uniq={np.unique(a)[:8]}')

# MD5 overlap with oral_mamba images
print('\n-- 与 oral_mamba 对比 --')
mamba_imgs = sorted(os.listdir(MAMBA_I))
mamba_md5 = {md5(os.path.join(MAMBA_I, f)): f for f in mamba_imgs}
same_i, diff_i = [], []
for f in imgs:
    h = md5(os.path.join(SRC, 'images', f))
    (same_i if h in mamba_md5 else diff_i).append(f)
print(f'images 与 oral_mamba 完全相同: {len(same_i)} / {len(imgs)}')
sc = collections.Counter(cat(f) for f in same_i)
dc = collections.Counter(cat(f) for f in diff_i)
print('  相同部分命名分布:', dict(sc))
print('  新增部分命名分布:', dict(dc))

# mamba labels compare
mamba_lab_md5 = {md5(os.path.join(MAMBA_L, f)): f for f in os.listdir(MAMBA_L)}
same_l = sum(1 for f in labs if md5(os.path.join(SRC, 'lables', f)) in mamba_lab_md5)
print(f'lables 与 oral_mamba/lables 完全相同: {same_l} / {len(labs)}')

# check that identical images map to identical labels (same filename pairs)
mismatch = 0
for f in same_i[:50]:
    li = stem_map.get(os.path.splitext(f)[0])
    ml = mamba_md5[md5(os.path.join(SRC, 'images', f))]
    mlstem = os.path.splitext(ml)[0]
    # find mamba label by same stem
    mp = os.path.join(MAMBA_L, mlstem + '.png')
    if li and os.path.exists(mp):
        h1 = md5(os.path.join(SRC, 'lables', li))
        h2 = md5(mp)
        if h1 != h2:
            mismatch += 1
print(f'抽样50对: 图同但label不同的数量: {mismatch}')

# overlap with oral_detector
print('\n-- 与 oral_detector 对比 --')
det_md5 = collections.defaultdict(list)
for s in ['train', 'valid']:
    for f in os.listdir(os.path.join(DET, s, 'images')):
        det_md5[md5(os.path.join(DET, s, 'images', f))].append((s, f))
det_hit = [f for f in imgs if md5(os.path.join(SRC, 'images', f)) in det_md5]
print(f'与 oral_detector 内容相同的 images: {len(det_hit)}')

# resolution distribution
print('\n-- 分辨率分布 --')
res = collections.Counter()
res_by_cat = collections.defaultdict(collections.Counter)
dims_cache = {}
for f in imgs:
    with Image.open(os.path.join(SRC, 'images', f)) as im:
        wh = im.size
    dims_cache[f] = wh
    res[wh] += 1
    res_by_cat[cat(f)][wh] += 1
for wh, n in res.most_common(12):
    print(f'  {wh[0]}x{wh[1]}: {n} ({n/len(imgs)*100:.1f}%)')
print(f'  共 {len(res)} 种分辨率')
for c, rc in res_by_cat.items():
    top = rc.most_common(3)
    print(f'  [{c}] top3: {top}')

# duplicates within
print('\n-- 内部重复 --')
h2f = collections.defaultdict(list)
for f in imgs:
    h2f[md5(os.path.join(SRC, 'images', f))].append(f)
dups = {h: v for h, v in h2f.items() if len(v) > 1}
print(f'MD5 重复组: {len(dups)}, 涉及文件: {sum(len(v) for v in dups.values())}')
for h, v in list(dups.items())[:10]:
    print('  ', v)

# ahash near dup within new (non-mamba) subset only, to save time
new_files = diff_i
print(f'\n-- 新增部分 ({len(new_files)} 张) 内部近重复 (ahash d<=4) --')
keys, hashes = [], []
for f in new_files:
    h = ahash(os.path.join(SRC, 'images', f))
    if h is not None:
        keys.append(f)
        hashes.append(np.frombuffer(h, dtype=np.uint8))
bits = np.unpackbits(np.stack(hashes), axis=1).astype(np.int8)
n = len(keys)
D = (bits[:, None, :] != bits[None, :, :]).sum(-1)
iu = np.triu_indices(n, k=1)
dv = D[iu]
hist = {d: int((dv == d).sum()) for d in range(5)}
print('距离分布(<=4):', hist)
sel = np.where(dv <= 4)[0]
crosscat = 0
for i in sel[:40]:
    k1, k2 = keys[iu[0][i]], keys[iu[1][i]]
    if cat(k1) != cat(k2):
        crosscat += 1
    print(f'  d={dv[i]}: {k1[:40]} <-> {k2[:40]}')
print(f'  其中跨命名类别对: {crosscat}')
json.dump({'same_as_mamba': len(same_i), 'new': len(diff_i), 'new_by_cat': dict(dc),
           'label_same_as_mamba': same_l, 'det_overlap': len(det_hit),
           'md5_dup_groups': len(dups)}, open(f'{_ROOT}/outputs/intelligent_diagnostic_audit.json', 'w'), indent=1)
print('\nsaved -> outputs/intelligent_diagnostic_audit.json')
