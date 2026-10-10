# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
import os, json, hashlib, collections, re
import numpy as np
from PIL import Image

# Project root resolved from this file's location (no hardcoded paths).
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

ROOT = f'{_ROOT}/datasets/raw/oral_detector/extracted'
MAMBA = f'{_ROOT}/datasets/raw/oral_mamba/images'
SPLITS = ['train', 'valid']
IMG_EXT = {'.jpg', '.jpeg', '.png'}

def md5(p, buf=1 << 20):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        while chunk := f.read(buf):
            h.update(chunk)
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

data = {}
for s in SPLITS:
    img_dir = os.path.join(ROOT, s, 'images')
    lab_dir = os.path.join(ROOT, s, 'labels')
    imgs = sorted(os.listdir(img_dir))
    recs = []
    for f in imgs:
        ip = os.path.join(img_dir, f)
        stem = os.path.splitext(f)[0]
        lp = os.path.join(lab_dir, stem + '.txt')
        boxes = []
        if os.path.exists(lp):
            with open(lp) as fh:
                for line in fh:
                    t = line.split()
                    if len(t) >= 5:
                        boxes.append((int(t[0]), *map(float, t[1:5])))
        recs.append({'file': f, 'path': ip, 'boxes': boxes})
    data[s] = recs

names = ['calculus', 'caries', 'gingivitis', 'lessions', 'plaque', 'tooth discoloration', 'ulcer', 'xerostomia']

# ---------- 1/2: class stats ----------
cls_boxes = collections.Counter()
cls_imgs = collections.defaultdict(set)
for s in SPLITS:
    for r in data[s]:
        for (c, *_ ) in r['boxes']:
            cls_boxes[c] += 1
            cls_imgs[c].add((s, r['file']))

print('=' * 70)
print('1) 类别框数统计 (train+valid)')
print('=' * 70)
for c in range(8):
    n_img = len({f for s, f in cls_imgs[c]})
    n_tr = len([1 for s, f in cls_imgs[c] if s == 'train'])
    n_va = len([1 for s, f in cls_imgs[c] if s == 'valid'])
    print(f'  [{c}] {names[c]:22s} boxes={cls_boxes[c]:5d}  images={n_img:5d} (train={n_tr}, valid={n_va})')
print(f'  总框数: {sum(cls_boxes.values())}, 总图数: {sum(len(v) for v in data.values())}')

# ---------- 3: resolution ----------
print()
print('=' * 70)
print('2) 图像分辨率分布')
print('=' * 70)
res_counter = collections.Counter()
res_by_split = collections.defaultdict(collections.Counter)
for s in SPLITS:
    for r in data[s]:
        with Image.open(r['path']) as im:
            wh = im.size
        r['size'] = wh
        res_counter[wh] += 1
        res_by_split[s][wh] += 1
for wh, n in res_counter.most_common(15):
    print(f'  {wh[0]}x{wh[1]}: {n} ({n / sum(res_counter.values()) * 100:.1f}%)')
print(f'  共 {len(res_counter)} 种分辨率')

# ---------- 4: exact duplicates ----------
print()
print('=' * 70)
print('3) 精确重复 (MD5)')
print('=' * 70)
h2f = collections.defaultdict(list)
for s in SPLITS:
    for r in data[s]:
        h2f[md5(r['path'])].append((s, r['file']))
dups = {h: v for h, v in h2f.items() if len(v) > 1}
cross = [(h, v) for h, v in dups.items() if len({s for s, _ in v}) > 1]
print(f'  精确重复组: {len(dups)} 组, 涉及 {sum(len(v) for v in dups.values())} 个文件')
print(f'  跨 train/valid 精确重复: {len(cross)} 组')
for h, v in cross[:10]:
    print(f'    {v}')
if dups and not cross:
    for h, v in list(dups.items())[:5]:
        print(f'    同split内: {v}')

# ---------- roboflow base-name groups (potential aug copies) ----------
print()
print('=' * 70)
print('4) Roboflow 同源名分组 (潜在增强/裁剪副本)')
print('=' * 70)
base_groups = collections.defaultdict(list)
for s in SPLITS:
    for r in data[s]:
        base = re.sub(r'\.rf\.[A-Za-z0-9]+$', '', os.path.splitext(r['file'])[0])
        base_groups[base].append((s, r['file']))
multi = {b: v for b, v in base_groups.items() if len(v) > 1}
print(f'  同源名组: {len(multi)} (共 {len(base_groups)} 个唯一源名)')
for b, v in list(sorted(multi.items(), key=lambda kv: -len(kv[1])))[:12]:
    print(f'    {b}: {len(v)} 份 -> {[x[1][:40] for x in v[:4]]}')

# ---------- 5: near-duplicates via ahash ----------
print()
print('=' * 70)
print('5) 近重复检测 (perceptual hash, 汉明距<=6/64)')
print('=' * 70)
hashes, keys = [], []
for s in SPLITS:
    for r in data[s]:
        h = ahash(r['path'])
        if h is not None:
            hashes.append(np.frombuffer(h, dtype=np.uint8))
            keys.append((s, r['file']))
H = np.stack(hashes)  # (n,8)
bits = np.unpackbits(H, axis=1).astype(np.int8)  # (n,64)
n = len(keys)
near = []
BATCH = 256
for i0 in range(0, n, BATCH):
    b = bits[i0:i0 + BATCH]
    d = (b[:, None, :] != bits[None, :, :]).sum(-1)
near = []
for i0 in range(0, n, BATCH):
    b = bits[i0:i0 + BATCH]
    d = (b[:, None, :] != bits[None, :, :]).sum(-1)
    idx = np.argwhere(np.triu(d, k=1) <= 6)
    for a, bb in idx:
        near.append((keys[i0 + a], keys[bb], int(d[a, bb])))
cross_near = [(k1, k2, d) for k1, k2, d in near if k1[0] != k2[0]]
same_near = [(k1, k2, d) for k1, k2, d in near if k1[0] == k2[0]]
print(f'  近重复对总数: {len(near)}  (跨split: {len(cross_near)}, 同split内: {len(same_near)})')
print('  跨split近重复示例:')
for k1, k2, d in cross_near[:15]:
    print(f'    {k1} <-> {k2} (dist={d})')

# ---------- 6: bbox size distribution ----------
print()
print('=' * 70)
print('6) bbox 尺寸分布 (像素宽x高; 相对面积% of image)')
print('=' * 70)
print(f'  {"class":22s} {"n":>5s} {"w_med":>7s} {"w_p25":>7s} {"w_p75":>7s} {"h_med":>7s} {"h_p25":>7s} {"h_p75":>7s} {"area%_med":>9s} {"area%_p75":>9s}')
for c in range(8):
    ws, hs, ar = [], [], []
    for s in SPLITS:
        for r in data[s]:
            if not r.get('size'):
                continue
            W, Hh = r['size']
            for (cc, x, y, w, h) in r['boxes']:
                if cc == c:
                    ws.append(w * W); hs.append(h * Hh); ar.append(w * h * 100)
    if not ws:
        print(f'  {names[c]:22s} {0:5d}')
        continue
    q = lambda a, p: np.percentile(a, p)
    print(f'  {names[c]:22s} {len(ws):5d} {q(ws,50):7.1f} {q(ws,25):7.1f} {q(ws,75):7.1f} {q(hs,50):7.1f} {q(hs,25):7.1f} {q(hs,75):7.1f} {q(ar,50):9.2f} {q(ar,75):9.2f}')

# ---------- 7: oral_mamba overlap ----------
print()
print('=' * 70)
print('7) 与 oral_mamba 重叠检查')
print('=' * 70)
mamba_files = os.listdir(MAMBA)
mamba_md5 = {}
for f in mamba_files:
    mamba_md5[md5(os.path.join(MAMBA, f))] = f
mamba_stems = {os.path.splitext(f)[0] for f in mamba_files}
det_stem2file = {}
for s in SPLITS:
    for r in data[s]:
        stem = re.sub(r'\.rf\.[A-Za-z0-9]+$', '', os.path.splitext(r['file'])[0])
        det_stem2file.setdefault(stem, []).append((s, r['file']))
stem_overlap = {st: v for st, v in det_stem2file.items() if st in mamba_stems}
md5_overlap = []
for s in SPLITS:
    for r in data[s]:
        h = md5(r['path'])
        if h in mamba_md5:
            md5_overlap.append((s, r['file'], mamba_md5[h]))
print(f'  oral_mamba 图片数: {len(mamba_files)}')
print(f'  文件名词干重合 (可能同源): {len(stem_overlap)}')
for st, v in list(stem_overlap.items())[:10]:
    print(f'    {st} -> {v[:3]}')
print(f'  内容 MD5 完全相同: {len(md5_overlap)}')
for s, f, mf in md5_overlap[:10]:
    print(f'    {s}/{f} == oral_mamba/{mf}')

# per-class images for the 4 asked classes
print()
print('=' * 70)
print('8) 指定类别明细')
print('=' * 70)
for c, label in [(0, 'calculus'), (4, 'plaque'), (5, 'tooth discoloration'), (2, 'gingivitis')]:
    imgs_c = {f for s, f in cls_imgs[c]}
    tr = [1 for s, f in cls_imgs[c] if s == 'train']
    va = [1 for s, f in cls_imgs[c] if s == 'valid']
    print(f'  {label:20s}: {cls_boxes[c]} 个框 / {len(imgs_c)} 张图 (train图 {len(tr)}, valid图 {len(va)})')

# images with zero boxes
zero = [(s, r['file']) for s in SPLITS for r in data[s] if len(r['boxes']) == 0]
print(f'\n  无标注背景图: {len(zero)}')
json.dump({'cls_boxes': {names[c]: cls_boxes[c] for c in range(8)},
           'near_dups_cross': [(k1, k2, d) for k1, k2, d in cross_near]},
          open(f'{_ROOT}/outputs/oral_detector_audit.json', 'w'), ensure_ascii=False, indent=1)
print('saved -> outputs/oral_detector_audit.json')
