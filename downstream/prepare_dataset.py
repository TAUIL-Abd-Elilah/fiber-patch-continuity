"""Bounded downstream dataset for arms P / A / D (fixed before any fit).

Region: fit-frame z [12500, 13500) inside the calibration band. Fibers: the
calibration-split fibers with >= 4 villa-decimated points in the region,
split by whole file (sha256 order, every third -> held-out evaluation).
Held-out fibers never enter the dataset. Patches: directory junctions to
every verified patch whose meta bbox meets z [12400, 13600) (the fitter's
own z filter then applies). D's rejected pairs for the fit fibers come from
the calibration run's frozen decisions (runs/cal_region/pairs.jsonl).
Writes manifest.json; refuses to overwrite a different one.
"""
import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FPA = os.path.join(os.path.dirname(HERE), 'fiber_patch_assoc')
sys.path.insert(0, FPA)
import native_adapter as na  # noqa: E402

Z0, Z1 = 12500.0, 13500.0
PATCH_Z0, PATCH_Z1 = 12400.0, 13600.0
SEED = 'fpa-downstream-20260923'
FIBERS = 'C:/VesuviusProgressData/PHercParis4/eval_fibers_remote_20260923'
PATCHES = 'C:/VesuviusProgressData/PHercParis4/verified_patches'


def main():
    splits = json.load(open(os.path.join(FPA, 'splits_20260923.json')))
    cal = sorted(f for f, s in splits['assignment'].items() if s == 'cal')
    usable = []
    for name in cal:
        with contextlib.redirect_stdout(io.StringIO()):
            _, col = na.villa_load_fiber(os.path.join(FIBERS, name), 0, 40.0)
        z = np.array([v['p'][2] for v in col['points'].values()])
        n_in = int(((z >= Z0) & (z < Z1)).sum())
        if n_in >= 4:
            usable.append((name, n_in))
    order = sorted(usable, key=lambda t: hashlib.sha256(f'{SEED}:{t[0]}'.encode()).hexdigest())
    evaluation = [n for i, (n, _) in enumerate(order) if i % 3 == 0]
    fit = [n for i, (n, _) in enumerate(order) if i % 3 != 0]

    pairs = [json.loads(l) for l in open(os.path.join(FPA, 'runs', 'cal_region', 'pairs.jsonl'))]
    rejected = {}
    for p in pairs:
        if p['fiber_file'] in fit and p['decisions']['D'] != 'propose':
            rejected.setdefault(os.path.splitext(p['fiber_file'])[0], []).append(p['patch'])
    rejected = {k: sorted(v) for k, v in sorted(rejected.items())}

    index = np.load(os.path.join(FPA, 'patch_meta_index.npz'))
    zlo, zhi = index['bbox_lo_xyz'][:, 2], index['bbox_hi_xyz'][:, 2]
    patch_names = sorted(index['names'][(zhi >= PATCH_Z0) & (zlo < PATCH_Z1)].tolist())

    manifest = {
        'version': 1, 'region_z': [Z0, Z1], 'patch_prefilter_z': [PATCH_Z0, PATCH_Z1], 'seed': SEED,
        'fit_fibers': {n: na.sha256_file(os.path.join(FIBERS, n)) for n in fit},
        'eval_fibers': {n: na.sha256_file(os.path.join(FIBERS, n)) for n in evaluation},
        'in_region_points': dict(usable),
        'patches': len(patch_names),
        'patch_listing_sha256': hashlib.sha256('\n'.join(patch_names).encode()).hexdigest(),
        'd_rejected_pairs': rejected,
        'd_rejected_pair_count': sum(len(v) for v in rejected.values()),
        'arms': {'P': 'verified patches only', 'A': 'patches + fit fibers, native linking',
                 'D': 'patches + fit fibers, native linking with D-reviewed pairs rejected'},
        'primary_metric': 'point-weighted native strip satisfaction of eval fibers '
                          '(satisfaction_metrics.get_unattached_pcl_satisfied_counts), D vs A',
        'secondary': ['per-fiber mean satisfaction', 'physical projection error um (median, p90)',
                      'fit fibers attached-point counts per arm'],
    }
    body = json.dumps(manifest, indent=1, sort_keys=True)
    out = os.path.join(HERE, 'manifest.json')
    if os.path.exists(out) and open(out).read() != body:
        raise SystemExit('manifest.json exists with different content')

    ds = os.path.join(HERE, 'dataset')
    os.makedirs(os.path.join(ds, 'fibers'), exist_ok=True)
    os.makedirs(os.path.join(ds, 'verified_patches'), exist_ok=True)
    scroll = json.load(open('C:/VesuviusProgressData/PHercParis4/spiral-scroll.json'))
    scroll['base_shape_zyx'] = na.FIT_BASE_SHAPE_ZYX
    scroll['provenance_note'] = (scroll.get('provenance_note', '') +
                                 ' base_shape_zyx added from the published zarr level 2 shape.')
    json.dump(scroll, open(os.path.join(ds, 'spiral-scroll.json'), 'w'), indent=2)
    shutil.copyfile('C:/VesuviusProgressData/PHercParis4/umbilicus.json', os.path.join(ds, 'umbilicus.json'))
    for n in fit:
        shutil.copyfile(os.path.join(FIBERS, n), os.path.join(ds, 'fibers', n))
    ev = os.path.join(HERE, 'eval_fibers')
    os.makedirs(ev, exist_ok=True)
    for n in evaluation:
        shutil.copyfile(os.path.join(FIBERS, n), os.path.join(ev, n))
    made = 0
    for name in patch_names:
        link = os.path.join(ds, 'verified_patches', name)
        if not os.path.exists(link):
            subprocess.run(['cmd', '/c', 'mklink', '/J', link, os.path.join(PATCHES, name).replace('/', '\\')],
                           check=True, capture_output=True)
            made += 1
    with open(out, 'w') as handle:
        handle.write(body)
    print(f'fit {len(fit)} eval {len(evaluation)} patches {len(patch_names)} (new junctions {made}) '
          f'D-rejected pairs {manifest["d_rejected_pair_count"]}')
    print('manifest sha256', hashlib.sha256(body.encode()).hexdigest())


if __name__ == '__main__':
    main()
