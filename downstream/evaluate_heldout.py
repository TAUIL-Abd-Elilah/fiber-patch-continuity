"""Held-out fiber evaluation of a fitted checkpoint (read-only, no fitting).

Adapted from the 19 Sep pilot evaluator: villa's own fiber loader,
materializer, checkpoint model reconstruction and native strip-satisfaction
metric (satisfaction_metrics.get_unattached_pcl_satisfied_counts). The
held-out fibers never entered any fitting arm (manifest.json).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SPIRAL = HERE.parents[1] / '_worktrees' / 'villa-fiber-assoc' / 'spiral-fitting'
VOXEL_UM = 9.6


class ObserveInverse:
    def __init__(self, transform):
        self.transform = transform
        self.projected_cpu = []

    def __call__(self, points):
        return self.transform(points)

    @property
    def inv(self):
        base = self.transform.inv

        def project(points):
            out = base(points)
            self.projected_cpu.append(out.detach().cpu())
            return out
        return project


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    for k, v in {'FIT_SPIRAL_COMPILE': '0', 'FIT_SPIRAL_TRITON': '0', 'WANDB_MODE': 'disabled'}.items():
        os.environ[k] = v
    sys.path.insert(0, str(SPIRAL))
    import numpy as np
    import torch
    import fit_spiral as fs
    import satisfaction_metrics as metrics
    from checkpoint_io import load_checkpoint_cpu
    from flatten_spiral_checkpoint import _build_model, _checkpoint_config
    from spiral_helpers import load_fiber_point_collections

    manifest = json.loads((HERE / 'manifest.json').read_text())
    run = HERE / 'runs' / args.run
    summary = json.loads((run / 'summary.json').read_text())
    if summary['status'] != 'completed':
        raise SystemExit('run not completed')
    eval_dir = HERE / 'eval_fibers'
    names = sorted(p.name for p in eval_dir.glob('*.json'))
    if set(names) != set(manifest['eval_fibers']):
        raise SystemExit('eval fibers differ from manifest')
    for n in names:
        if hashlib.sha256((eval_dir / n).read_bytes()).hexdigest() != manifest['eval_fibers'][n]:
            raise SystemExit(f'eval fiber changed: {n}')
    fit_names = set(os.listdir(HERE / 'dataset' / 'fibers'))
    if fit_names & set(names):
        raise SystemExit('eval fiber inside fitting dataset')

    checkpoint = load_checkpoint_cpu(str(run / 'checkpoint_fitted.ckpt'))
    cfg = _checkpoint_config(checkpoint)
    z0, z1 = int(cfg['z_begin']), int(cfg['z_end'])
    collections, _ = load_fiber_point_collections(
        str(eval_dir), 0, min_point_spacing=cfg['pcl_fiber_min_point_spacing'],
        base_shape_zyx=checkpoint.get('base_shape_zyx'))
    catalog = {Path(p['source_file']).stem: p for p in collections.values()}
    radial = (float(cfg['pcl_vertical_fiber_radial_offset_voxels'])
              if cfg['pcl_vertical_fiber_radial_offset_enabled'] else 0.0)
    cross, strips, _, links, _ = fs.materialize_fiber_fit_inputs(
        catalog, {}, z_begin=z0, z_end=z1, z_margin=0,
        min_point_spacing=cfg['pcl_unattached_pcl_min_point_spacing'],
        use_links=False, use_pending_links=False,
        vertical_min_z_fraction=cfg['pcl_vertical_fiber_min_z_fraction'],
        vertical_min_auto_certainty=cfg['pcl_vertical_fiber_min_auto_certainty'],
        vertical_radial_offset=radial)
    strips = [s for s in strips if len(s['zyxs']) >= 2]
    device = torch.device(args.device)
    model = _build_model(checkpoint, cfg, str(HERE / 'dataset' / 'umbilicus.json'), device)
    transform = model.get_slice_to_spiral_transform()
    dr = model.get_dr_per_winding()
    observed = ObserveInverse(transform)
    with torch.no_grad():
        satisfied, totals, masks = metrics.get_unattached_pcl_satisfied_counts(
            observed, dr, strips, fs.get_or_build_unattached_pcl_flat)
        flat = fs.get_or_build_unattached_pcl_flat(strips, dr.device)
        projected = torch.cat(observed.projected_cpu, 0) if observed.projected_cpu else None
        source = flat['zyxs'].detach().cpu()
    dist = (torch.linalg.vector_norm(projected - source, dim=-1).numpy() * VOXEL_UM
            if projected is not None and projected.shape == source.shape else None)
    rows = []
    offs = np.cumsum([0] + totals.tolist())
    for i, (s, sc, tc) in enumerate(zip(strips, satisfied.tolist(), totals.tolist())):
        d = dist[offs[i]:offs[i + 1]] if dist is not None else None
        rows.append({'fiber': Path(s['source_file']).name, 'points': tc, 'satisfied': sc,
                     'fraction': sc / tc if tc else None,
                     'median_error_um': float(np.median(d)) if d is not None and len(d) else None,
                     'p90_error_um': float(np.percentile(d, 90)) if d is not None and len(d) else None})
    total_pts = sum(r['points'] for r in rows)
    out = {'run': args.run, 'arm': summary['arguments']['arm'], 'seed': summary['arguments']['seed'],
           'steps': summary['arguments']['steps'], 'z': [z0, z1],
           'eval_fibers_evaluated': len(rows), 'eval_points': total_pts,
           'point_weighted_satisfaction': sum(r['satisfied'] for r in rows) / total_pts if total_pts else None,
           'fiber_mean_satisfaction': float(np.mean([r['fraction'] for r in rows])) if rows else None,
           'fully_satisfied_fibers': sum(r['satisfied'] == r['points'] for r in rows),
           'median_error_um': float(np.median(dist)) if dist is not None else None,
           'metric_config': dict(metrics.metrics_config), 'per_fiber': rows}
    (run / 'heldout_eval.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ('per_fiber', 'metric_config')}))


if __name__ == '__main__':
    main()
