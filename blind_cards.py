"""Blinded review cards for human labels (correct / wrong / unresolved).

Each card: one fiber (yellow circle) and one candidate patch (cyan curve) in
six true-aspect CT cross-sections (4.8 um), perpendicular to the fiber at
evenly spaced positions over the stretch where the patch covers the fiber
(padded to >= 1.5 mm). Nothing on the card reveals any arm's decision or
evidence summary. Card order is shuffled with a fixed seed; the key mapping
card id -> pair and strata is written separately (key.json) and must not be
shown to the reviewer.

Sampling (per region, seeded): strata over native-accepted pairs
  S1 A accept & D propose    S2 A accept & D review(contradiction)
  S3 A accept & D review(short support only)    S4 D propose & A not linked
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import ct
import evidence as ev
import native_adapter as na
import review_export as rx
from pair_evidence import pair_states, ABSENT, DenseIndex

FIBER_ROOTS = ['C:/VesuviusProgressData/PHercParis4/eval_fibers',
               'C:/VesuviusProgressData/PHercParis4/eval_fibers_remote_20260923']


def stratum(p):
    a = p['decisions']['A'] == 'accept'
    d = p['decisions']['D']
    reasons = p['reason_codes'].get('D', [])
    if a and d == 'propose':
        return 'S1'
    if a and 'in_domain_contradiction' in reasons:
        return 'S2'
    if a and d == 'review':
        return 'S3'
    if not a and d == 'propose':
        return 'S4'
    return None


def section(level, origin, e_t, inward, half_um=360.0):
    scale = 2 ** (2 - level)
    n_ct = ct.sheet_normals(level, origin[None] * scale, half=12 * scale, sigma_tensor=4.0 * scale)[0]
    r = np.asarray(inward(origin[None]), dtype=np.float64)[0]
    e_u = n_ct if np.isfinite(n_ct).all() else r
    e_u = e_u * (np.sign(e_u @ r) or 1.0)
    e_u = e_u - (e_u @ e_t) * e_t
    e_u /= np.linalg.norm(e_u)
    e_b = np.cross(e_t, e_u)
    half = half_um / na.UM_PER_VOXEL
    g = np.arange(-half, half + 1e-6, 1.0 / scale)
    coords = origin + g[:, None, None] * e_u + g[None, :, None] * e_b
    return ct.sample(level, coords * scale), e_u, e_b


def render_card(path_png, line, line_arc, dense, cid, patch_name, patch, inward, level=1):
    st, _ = pair_states(dense, cid, patch_name)
    arc_s = dense['arclen_%d' % cid]
    cov = np.flatnonzero(st != ABSENT)
    lo, hi = arc_s[cov].min(), arc_s[cov].max()
    if hi - lo < 1500:
        mid = (lo + hi) / 2
        lo, hi = mid - 750, mid + 750
    lo, hi = max(lo, line_arc[0]), min(hi, line_arc[-1])
    positions = np.linspace(lo, hi, 6)
    # Prefetch the chunks the six section planes cross (radial frame; the CT
    # normal used for the final plane differs by a small tilt, covered by the
    # stencil margin), in parallel.
    scale = 2 ** (2 - level)
    half = 360 / na.UM_PER_VOXEL
    grid = np.arange(-half, half + 1e-6, 8.0)
    plane_pts = []
    for pos in positions:
        idx = int(np.argmin(np.abs(line_arc - pos)))
        a, b = max(0, idx - 12), min(len(line) - 1, idx + 12)
        e_t = (line[b] - line[a]) / max(np.linalg.norm(line[b] - line[a]), 1e-9)
        r = np.asarray(inward(line[idx][None]), dtype=np.float64)[0]
        e_u = r - (r @ e_t) * e_t
        e_u /= max(np.linalg.norm(e_u), 1e-9)
        e_b = np.cross(e_t, e_u)
        plane_pts.append((line[idx] + grid[:, None, None] * e_u + grid[None, :, None] * e_b).reshape(-1, 3))
    ct.prefetch(level, np.concatenate(plane_pts) * scale, margin=12 * scale)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.2))
    for ax, pos in zip(axes.ravel(), positions):
        idx = int(np.argmin(np.abs(line_arc - pos)))
        a, b = max(0, idx - 12), min(len(line) - 1, idx + 12)
        e_t = line[b] - line[a]
        e_t /= np.linalg.norm(e_t)
        origin = line[idx]
        img, e_u, e_b = section(level, origin, e_t, inward)
        vals = img[img > 0]
        vmin, vmax = np.percentile(vals, [1, 99.5]) if len(vals) else (0, 255)
        ext = [-360, 360, -360, 360]
        ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax, origin='lower', extent=ext)
        m = dense.pair_rows(cid, patch_name)
        near = m[np.argmin(np.abs(arc_s[dense['sample'][m]] - pos))]
        if abs(arc_s[dense['sample'][near]] - pos) <= 400:
            ij = rx._foot_ij(patch, dense['foot_zyx'][near])
            curve = rx.patch_plane_curve(patch, ij, origin, e_u, e_b, e_t)
            if len(curve):
                ax.plot(curve[:, 0] * na.UM_PER_VOXEL, curve[:, 1] * na.UM_PER_VOXEL, '.',
                        ms=1.0, color='#22b8cf')
        ax.plot([0], [0], 'o', mfc='none', mec='#ffd43b', ms=10, mew=1.6)
        ax.set_xlim(-360, 360)
        ax.set_ylim(-360, 360)
        ax.set_xticks([-300, 0, 300])
        ax.set_yticks([-300, 0, 300])
        ax.tick_params(labelsize=6)
        ax.set_title(f'{(pos - positions[0]) / 1000:.1f} mm', fontsize=8)
    fig.suptitle('yellow circle: fiber   cyan: candidate patch   (um; up = toward umbilicus)',
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path_png, dpi=85, pil_kwargs={'optimize': True})
    plt.close(fig)
    return [float(p) for p in positions]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--per-stratum', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20260923)
    parser.add_argument('--out', required=True)
    parser.add_argument('--umbilicus', default='C:/VesuviusProgressData/PHercParis4/umbilicus.json')
    parser.add_argument('--patch-root', default='C:/VesuviusProgressData/PHercParis4/verified_patches')
    args = parser.parse_args()
    os.makedirs(os.path.join(args.out, 'cards'), exist_ok=True)
    inward = na.umbilicus_inward(args.umbilicus)
    rng = random.Random(args.seed)
    chosen = []
    for run in args.runs:
        pairs = [json.loads(l) for l in open(os.path.join(run, 'pairs.jsonl'))]
        by = {}
        for p in pairs:
            s = stratum(p)
            if s:
                by.setdefault(s, []).append(p)
        for s in sorted(by):
            pool = sorted(by[s], key=lambda p: (p['fiber_file'], p['patch']))
            # at most two pairs per parent fiber per stratum (correlated pairs)
            rng.shuffle(pool)
            picked, per_fiber = [], {}
            for p in pool:
                if per_fiber.get(p['fiber_file'], 0) >= 2:
                    continue
                picked.append(p)
                per_fiber[p['fiber_file']] = per_fiber.get(p['fiber_file'], 0) + 1
                if len(picked) >= args.per_stratum:
                    break
            chosen += [(run, s, p) for p in picked]
    rng.shuffle(chosen)
    key = []
    dense_cache, cid_cache = {}, {}
    for n, (run, s, p) in enumerate(chosen):
        card_id = hashlib.sha256(f"{args.seed}:{p['fiber_file']}:{p['patch']}".encode()).hexdigest()[:10]
        if run not in dense_cache:
            dense_cache[run] = DenseIndex(np.load(os.path.join(run, 'dense.npz')))
            rep = json.load(open(os.path.join(run, 'replay_report.json')))
            cid_cache[run] = {v['file']: int(k) for k, v in rep['fiber_provenance'].items()}
        dense = dense_cache[run]
        cid = cid_cache[run][p['fiber_file']]
        fpath = next(os.path.join(r, p['fiber_file']) for r in FIBER_ROOTS
                     if os.path.exists(os.path.join(r, p['fiber_file'])))
        _, _, _, line, line_arc = ev.resample_fiber(fpath, 8.0)
        patches, _ = na.load_patches(args.patch_root, [p['patch']], 1)
        png = os.path.join(args.out, 'cards', f'{card_id}.png')
        if not os.path.exists(png):
            positions = render_card(png, line.astype(np.float64), line_arc, dense, cid,
                                    p['patch'], patches[p['patch']], inward)
        else:
            positions = None
        key.append({'card_id': card_id, 'order': n, 'run': run, 'stratum': s,
                    'fiber_file': p['fiber_file'], 'patch': p['patch'],
                    'decisions': p['decisions'], 'positions_um': positions})
        print(f'{n + 1}/{len(chosen)} {card_id} {s} ct_bytes={ct.downloaded_bytes()}', flush=True)
    with open(os.path.join(args.out, 'key.json'), 'w') as handle:
        json.dump(key, handle, indent=1)
    with open(os.path.join(args.out, 'order.json'), 'w') as handle:
        json.dump([k['card_id'] for k in key], handle)


if __name__ == '__main__':
    main()
