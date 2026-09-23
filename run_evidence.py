"""Native replay + dense evidence for one bounded region.

Outputs in --out: capture.npz (native, decimated points), dense.npz (dense
samples, all envelope hits with normal/tangent decomposition),
replay_report.json.
"""
import argparse
import glob
import json
import os
import time

import numpy as np

import evidence as ev
import native_adapter as na


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fibers', nargs='+', required=True)
    parser.add_argument('--patch-root', default='C:/VesuviusProgressData/PHercParis4/verified_patches')
    parser.add_argument('--meta-index', default='patch_meta_index.npz')
    parser.add_argument('--umbilicus', default='C:/VesuviusProgressData/PHercParis4/umbilicus.json')
    parser.add_argument('--z-min', type=float, required=True)
    parser.add_argument('--z-max', type=float, required=True)
    parser.add_argument('--envelope', type=float, default=16.0)
    parser.add_argument('--dense-spacing', type=float, default=8.0)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    paths = sorted({p for pattern in args.fibers for p in glob.glob(pattern)})
    t0 = time.perf_counter()
    config = na.NativeConfig()
    collections, provenance, rejected = na.load_fibers(paths, config.min_point_spacing)
    band = (args.z_min, args.z_max)
    for cid in list(collections):
        pts = collections[cid]['points']
        for pid in [k for k, v in pts.items() if not band[0] <= v['zyx'][0] < band[1]]:
            del pts[pid]
        if not pts:
            del collections[cid]
            provenance.pop(cid)
    all_pts = np.concatenate([np.stack([p['zyx'] for p in c['points'].values()])
                              for c in collections.values()])
    names = na.select_patch_names(args.meta_index, all_pts, args.envelope + 50.0)
    patches, dropped = na.load_patches(args.patch_root, names, config.erode_cells)
    t1 = time.perf_counter()
    inward = na.umbilicus_inward(args.umbilicus)
    cap = na.capture(collections, patches, config, args.envelope, inward)
    unmodified = na.run_unmodified(collections, patches, config, inward)
    equality = na.replay_matches_native(cap, unmodified)
    # Arm B variants: the existing matcher with its own tighter options, on the
    # same fibers and patches (pre-specified grid; chosen later on cal labels).
    variants = {
        'B_win3_2': na.NativeConfig(window_points=3, window_min_points=2),
        'B_win5_3': na.NativeConfig(window_points=5, window_min_points=3),
        'B_side': na.NativeConfig(side_filter=True),
        'B_tol1.5': na.NativeConfig(tolerance=1.5),
        'B_win5_3_side_tol2': na.NativeConfig(tolerance=2.0, window_points=5,
                                               window_min_points=3, side_filter=True),
    }
    variant_links = {}
    for name, vcfg in variants.items():
        links = na.run_unmodified(collections, patches, vcfg, inward)
        variant_links[name] = {'config': vcfg.as_dict(),
                               'links': [[c, p, pt] for (c, p), (pt, _, _) in sorted(links.items())]}
    t2 = time.perf_counter()
    fiber_paths = {cid: next(p for p in paths if os.path.basename(p) == provenance[cid]['file'])
                   for cid in collections}
    dense = ev.dense_capture(fiber_paths, patches, args.envelope, inward,
                             spacing=args.dense_spacing, z_band=band)
    t3 = time.perf_counter()

    np.savez_compressed(
        os.path.join(args.out, 'capture.npz'),
        collection_id=cap.collection_id, point_pos=cap.point_pos, point_id=cap.point_id,
        patch=cap.patch, distance=cap.distance, ij=cap.ij, foot_zyx=cap.foot_zyx,
        side_offset=cap.side_offset, within_tol=cap.within_tol, native_winner=cap.native_winner,
        point_cids=np.asarray(sorted(cap.points)),
        **{f'points_{c}': cap.points[c] for c in cap.points},
        **{f'pointids_{c}': cap.point_ids[c] for c in cap.point_ids},
        **{f'nativeorig_{c}': np.asarray(provenance[c]['kept_orig_indices'])[cap.point_ids[c]]
           for c in cap.point_ids})
    np.savez_compressed(
        os.path.join(args.out, 'dense.npz'),
        cid=dense.cid, sample=dense.sample, patch=dense.patch, distance=dense.distance,
        d_normal=dense.d_normal, r_tangent=dense.r_tangent, foot_zyx=dense.foot_zyx,
        normal_zyx=dense.normal_zyx, cids=np.asarray(sorted(dense.samples)),
        **{f'samples_{c}': dense.samples[c] for c in dense.samples},
        **{f'arclen_{c}': dense.arclen_um[c] for c in dense.arclen_um},
        **{f'orig_{c}': dense.orig_index[c] for c in dense.orig_index})
    patch_meta = {n: {'area_vx2': float(patches[n].area), 'shape': list(patches[n].zyxs.shape[:2]),
                      'family': 'band_seed' if n.startswith('band-seed') else 'legacy'}
                  for n in patches}
    with open(os.path.join(args.out, 'patch_meta.json'), 'w') as handle:
        json.dump(patch_meta, handle)
    with open(os.path.join(args.out, 'variant_links.json'), 'w') as handle:
        json.dump(variant_links, handle)
    report = {
        'villa_revision': na.villa_revision(), 'config': config.as_dict(),
        'config_hash': config.hash(), 'envelope_voxels': args.envelope,
        'dense_spacing_voxels': args.dense_spacing, 'z_band': band,
        'fibers_requested': len(paths), 'fibers_loaded': len(collections),
        'fibers_rejected': rejected,
        'fiber_provenance': {str(k): {kk: vv for kk, vv in v.items() if kk != 'kept_orig_indices'}
                             for k, v in provenance.items()},
        'patches_prefiltered': len(names), 'patches_loaded': len(patches),
        'patches_dropped': len(dropped), 'native_points': int(len(all_pts)),
        'native_links': equality['native_links'], 'equality': equality,
        'dense_samples': int(sum(len(v) for v in dense.samples.values())),
        'dense_hits': int(len(dense.cid)),
        'seconds': {'load': t1 - t0, 'native_capture_and_replay': t2 - t1, 'dense': t3 - t2},
    }
    with open(os.path.join(args.out, 'replay_report.json'), 'w') as handle:
        json.dump(report, handle, indent=1, default=str)
    print(json.dumps({k: report[k] for k in ('fibers_loaded', 'patches_loaded', 'native_points',
                                             'native_links', 'equality', 'dense_samples',
                                             'dense_hits', 'seconds')}, indent=1, default=str))


if __name__ == '__main__':
    main()
