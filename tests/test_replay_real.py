"""Bounded real-data replay: the instrumented capture must reproduce villa's
unmodified link_points_to_patches exactly. Skipped unless the PHercParis4
fibers/patches and a villa checkout (VILLA_SPIRAL_FITTING) are present."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

DATA = os.environ.get('PARIS4_DATA', 'C:/VesuviusProgressData/PHercParis4')
FIBER = os.path.join(DATA, 'eval_fibers', 'anon_20260815T030050140_000047.json')
META = os.path.join(HERE, 'patch_meta_index.npz')

pytestmark = pytest.mark.skipif(
    not (os.environ.get('VILLA_SPIRAL_FITTING') and os.path.exists(FIBER) and os.path.exists(META)),
    reason='needs VILLA_SPIRAL_FITTING, the PHercParis4 cache and patch_meta_index.npz')


@pytest.mark.parametrize('config', [
    dict(),
    dict(window_points=5, window_min_points=3),
    dict(side_filter=True),
    dict(tolerance=1.5),
])
def test_replay_equals_unmodified_linker(config):
    import native_adapter as na
    cfg = na.NativeConfig(**config)
    cols, prov, rejected = na.load_fibers([FIBER], cfg.min_point_spacing)
    assert len(cols) == 1 and not rejected
    for c in cols.values():
        for pid in [k for k, v in c['points'].items() if not 11000 <= v['zyx'][0] < 11600]:
            del c['points'][pid]
    pts = np.concatenate([np.stack([p['zyx'] for p in c['points'].values()]) for c in cols.values()])
    names = na.select_patch_names(META, pts, 18.0)
    patches, _ = na.load_patches(os.path.join(DATA, 'verified_patches'), names, cfg.erode_cells)
    inward = na.umbilicus_inward(os.path.join(DATA, 'umbilicus.json'))
    cap = na.capture(cols, patches, cfg, 16.0, inward)
    result = na.replay_matches_native(cap, na.run_unmodified(cols, patches, cfg, inward))
    assert result['native_links'] > 0
    assert result['mismatches'] == 0, result['examples']


def test_side_rules_engage():
    import native_adapter as na
    cols, _, _ = na.load_fibers([FIBER], 40.0)
    opts = na.native_options(na.NativeConfig(side_filter=True), cols, lambda z: z)
    assert len(opts.side_rules) == 1  # the H/V tagger saw a fiber, not an untagged pcl
