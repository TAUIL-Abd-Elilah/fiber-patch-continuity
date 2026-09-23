"""Freeze the region/lineage split before any policy output exists on new data.

Rules (fit-frame z, 9.6 um voxels):
  dev   : every previously used fiber (the 115 explicit-frame cache fibers),
          plus new fibers with >= 50% of points in z < 12000
  cal   : new fibers with >= 90% of points in [12500, 14500) and none in z < 12000
  test  : new fibers with >= 90% of points in [15000, 17500) and none below 14500
  excluded: everything else (straddlers), and any new fiber that retraces an
          already-used fiber (>= 20% of its points within 8 voxels of one)
Evaluation spans are later clipped to the fiber's own band.
Writes splits_20260923.json with file hashes; refuses to overwrite a different split.
"""
import contextlib
import hashlib
import io
import json
import os

import numpy as np
from scipy.spatial import cKDTree

import native_adapter as na

BANDS = {'cal': (12500.0, 14500.0), 'test': (15000.0, 17500.0)}
OUT = 'splits_20260923.json'


def points(path):
    with contextlib.redirect_stdout(io.StringIO()):
        _, col = na.villa_load_fiber(path, 0, 8.0)
    return np.stack([np.asarray(v['p'][::-1], dtype=np.float64) for v in col['points'].values()])


def main():
    inv = json.load(open('fiber_inventory.json'))
    roots = {'dev_cache_20260827': 'C:/VesuviusProgressData/PHercParis4/eval_fibers',
             'new_20260923': 'C:/VesuviusProgressData/PHercParis4/eval_fibers_remote_20260923'}
    old = [r for r in inv['accepted'] if r['origin'] == 'dev_cache_20260827']
    new = [r for r in inv['accepted'] if r['origin'] == 'new_20260923']
    old_pts = np.concatenate([points(os.path.join(roots[r['origin']], r['file'])) for r in old])
    tree = cKDTree(old_pts)
    out = {'rule_version': 1, 'bands': BANDS, 'dev_z_max': 12000.0,
           'assignment': {}, 'reasons': {}}
    for r in old:
        out['assignment'][r['file']] = 'dev'
        out['reasons'][r['file']] = 'previously used (19-21 Sep development)'
    for r in new:
        pts = points(os.path.join(roots[r['origin']], r['file']))
        z = pts[:, 0]
        near_old = (tree.query(pts, distance_upper_bound=8.0)[0] < np.inf).mean()
        frac = {k: ((z >= lo) & (z < hi)).mean() for k, (lo, hi) in BANDS.items()}
        if near_old >= 0.2:
            split, why = 'excluded', f'retraces a used fiber ({near_old:.0%} within 8 vox)'
        elif (z < 12000).mean() >= 0.5:
            split, why = 'dev', 'majority below z 12000'
        elif frac['cal'] >= 0.9 and not (z < 12000).any():
            split, why = 'cal', f"{frac['cal']:.0%} in cal band"
        elif frac['test'] >= 0.9 and not (z < 14500).any():
            split, why = 'test', f"{frac['test']:.0%} in test band"
        else:
            split, why = 'excluded', 'straddles bands'
        out['assignment'][r['file']] = split
        out['reasons'][r['file']] = why
    out['sha256'] = {r['file']: r['sha256'] for r in inv['accepted']}
    out['counts'] = {s: sum(1 for v in out['assignment'].values() if v == s)
                     for s in ('dev', 'cal', 'test', 'excluded')}
    body = json.dumps(out, indent=1, sort_keys=True)
    digest = hashlib.sha256(body.encode()).hexdigest()
    if os.path.exists(OUT):
        prior = open(OUT).read()
        if hashlib.sha256(prior.encode()).hexdigest() != digest:
            raise SystemExit(f'{OUT} exists with different content; not overwriting')
    with open(OUT, 'w') as handle:
        handle.write(body)
    print(out['counts'], 'sha256', digest)


if __name__ == '__main__':
    main()
