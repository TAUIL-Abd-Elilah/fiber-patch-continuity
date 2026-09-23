"""Inventory every cached PHercParis4 fiber: frame rule, z extent (fit frame),
H/V tag, human branch links. Rejected frames are listed, never guessed."""
import collections
import contextlib
import glob
import io
import json
import os

import numpy as np

import native_adapter as na

ROOTS = {
    'dev_cache_20260827': 'C:/VesuviusProgressData/PHercParis4/eval_fibers/*.json',
    'new_20260923': 'C:/VesuviusProgressData/PHercParis4/eval_fibers_remote_20260923/*.json',
}


def main():
    rows, rejected = [], []
    for origin, pattern in ROOTS.items():
        for path in sorted(glob.glob(pattern)):
            name = os.path.basename(path)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    raw, col = na.villa_load_fiber(path, 0, 40.0)
            except na.FrameError as error:
                rejected.append({'file': name, 'origin': origin, 'reason': str(error)})
                continue
            z = np.array([v['p'][2] for v in col['points'].values()])
            hv = raw.get('hv_classification', {})
            rows.append(dict(file=name, origin=origin, frame_rule=raw['_frame_rule'],
                             sha256=na.sha256_file(path), zmin=float(z.min()), zmax=float(z.max()),
                             points=len(z), auto_tag=hv.get('automatic_tag'),
                             auto_certainty=hv.get('automatic_certainty'),
                             branches=len(raw.get('branches') or [])))
    with open('fiber_inventory.json', 'w') as handle:
        json.dump({'accepted': rows, 'rejected': rejected}, handle, indent=0)
    print('accepted', collections.Counter(r['origin'] for r in rows))
    print('rejected', collections.Counter(r['origin'] for r in rejected))
    edges = np.arange(0, 19000, 1000)
    for origin in ROOTS:
        spans = [(r['zmin'], r['zmax']) for r in rows if r['origin'] == origin]
        counts = [sum(1 for a, b in spans if a < e + 1000 and b >= e) for e in edges]
        print(origin, ' '.join(f'{int(e / 1000)}k:{c}' for e, c in zip(edges, counts)))


if __name__ == '__main__':
    main()
