"""Baseline replay on real PHercParis4 fibers + verified patches.

Writes <out>/capture.npz (all envelope hits), <out>/replay_report.json
(equality with the unmodified linker, runtime, memory, inputs, hashes).
"""
import argparse
import glob
import json
import os
import time
import tracemalloc

import numpy as np

import native_adapter as na


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fibers', nargs='+', required=True, help='fiber json files or globs')
    parser.add_argument('--patch-root', default='C:/VesuviusProgressData/PHercParis4/verified_patches')
    parser.add_argument('--meta-index', required=True)
    parser.add_argument('--umbilicus', default='C:/VesuviusProgressData/PHercParis4/umbilicus.json')
    parser.add_argument('--z-min', type=float, default=None, help='fit-frame z; drop fiber points outside')
    parser.add_argument('--z-max', type=float, default=None)
    parser.add_argument('--envelope', type=float, default=12.0)
    parser.add_argument('--tolerance', type=float, default=2.5)
    parser.add_argument('--window-points', type=int, default=1)
    parser.add_argument('--window-min-points', type=int, default=1)
    parser.add_argument('--side-filter', action='store_true')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    paths = sorted({p for pattern in args.fibers for p in glob.glob(pattern)})
    t0 = time.perf_counter()
    tracemalloc.start()
    config = na.NativeConfig(tolerance=args.tolerance, window_points=args.window_points,
                             window_min_points=args.window_min_points, side_filter=args.side_filter)
    collections, provenance, rejected = na.load_fibers(paths, config.min_point_spacing)
    if args.z_min is not None or args.z_max is not None:
        lo = -np.inf if args.z_min is None else args.z_min
        hi = np.inf if args.z_max is None else args.z_max
        for cid in list(collections):
            pts = collections[cid]['points']
            # Keep point ids untouched; drop points outside the band. Windows
            # are then taken over the kept points, like a fit ROI would see
            # after trimming (recorded in the report).
            for pid in [k for k, v in pts.items() if not lo <= v['zyx'][0] < hi]:
                del pts[pid]
            if not pts:
                del collections[cid]
                provenance.pop(cid)
    t_fibers = time.perf_counter()
    all_pts = np.concatenate([np.stack([p['zyx'] for p in c['points'].values()])
                              for c in collections.values()])
    names = na.select_patch_names(args.meta_index, all_pts, args.envelope + 2.0)
    patches, dropped = na.load_patches(args.patch_root, names, config.erode_cells)
    t_patches = time.perf_counter()
    inward = na.umbilicus_inward(args.umbilicus)
    cap = na.capture(collections, patches, config, args.envelope, inward)
    t_capture = time.perf_counter()
    unmodified = na.run_unmodified(collections, patches, config, inward)
    t_native = time.perf_counter()
    equality = na.replay_matches_native(cap, unmodified)
    _, peak = tracemalloc.get_traced_memory()

    np.savez_compressed(
        os.path.join(args.out, 'capture.npz'),
        collection_id=cap.collection_id, point_pos=cap.point_pos, point_id=cap.point_id,
        patch=cap.patch, distance=cap.distance, ij=cap.ij, foot_zyx=cap.foot_zyx,
        side_offset=cap.side_offset, within_tol=cap.within_tol, native_winner=cap.native_winner,
        point_cids=np.asarray(sorted(cap.points)),
        **{f'points_{cid}': cap.points[cid] for cid in cap.points},
        **{f'pointids_{cid}': cap.point_ids[cid] for cid in cap.point_ids})
    patch_meta = {name: {'area_vx2': float(patches[name].area),
                         'shape': list(patches[name].zyxs.shape[:2]),
                         'scale': [float(s) for s in patches[name].scale]} for name in patches}
    report = {
        'villa_revision': na.villa_revision(),
        'config': config.as_dict(), 'config_hash': config.hash(),
        'envelope_voxels': args.envelope, 'z_band': [args.z_min, args.z_max],
        'fibers_requested': len(paths), 'fibers_loaded': len(collections),
        'fibers_rejected': rejected,
        'fiber_provenance': {str(k): {kk: vv for kk, vv in v.items() if kk != 'kept_orig_indices'}
                             for k, v in provenance.items()},
        'fiber_points': int(len(all_pts)),
        'patches_prefiltered': len(names), 'patches_loaded': len(patches),
        'patches_dropped': dropped,
        'envelope_hits': int(len(cap.patch)), 'tolerance_hits': int(cap.within_tol.sum()),
        'equality': equality,
        'seconds': {'fibers': t_fibers - t0, 'patches': t_patches - t_fibers,
                    'capture': t_capture - t_patches, 'native_linker': t_native - t_capture},
        'python_peak_traced_MiB': peak / 2**20,
    }
    with open(os.path.join(args.out, 'replay_report.json'), 'w') as handle:
        json.dump(report, handle, indent=1, default=str)
    with open(os.path.join(args.out, 'patch_meta.json'), 'w') as handle:
        json.dump(patch_meta, handle)
    print(json.dumps({k: report[k] for k in ('fibers_loaded', 'fiber_points', 'patches_loaded',
                                             'envelope_hits', 'tolerance_hits', 'equality', 'seconds')},
                     indent=1, default=str))


if __name__ == '__main__':
    main()
