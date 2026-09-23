"""Matched spiral fits for the downstream comparison (arms P / A / D).

Same dataset, config, seed, steps and budget in every arm; the only
difference is the fiber input (P: none) and, for D, the fiber-to-patch pairs
the continuity policy routed to review are passed to villa's linker as
rejected patches (PatchLinkOptions.rejected_patches, branch
spiral-fiber-link-review-filter). Bounded development fit, not a production
recipe. GPU memory is capped (shared with another job).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
SPIRAL = HERE.parents[1] / '_worktrees' / 'villa-fiber-assoc' / 'spiral-fitting'
for key, value in {'AGENTS_AGENT_MODE': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'WANDB_MODE': 'disabled',
                   'FIT_SPIRAL_COMPILE': '0', 'FIT_SPIRAL_TRITON': '0', 'FIT_SPIRAL_SKIP_SAVE_MESH': '1',
                   'FIT_SPIRAL_SKIP_SAVE_OVERLAY': '1', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}.items():
    os.environ[key] = value
sys.path.insert(0, str(SPIRAL))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + '\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=['P', 'A', 'D'], required=True)
    p.add_argument('--steps', type=int, required=True)
    p.add_argument('--seed', type=int, default=20260923)
    p.add_argument('--name', required=True)
    p.add_argument('--max-seconds', type=float, default=5400)
    args = p.parse_args()
    out = HERE / 'runs' / args.name
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((HERE / 'manifest.json').read_text())
    started = time.perf_counter()
    summary = {'status': 'initializing', 'arguments': vars(args),
               'manifest_sha256': hashlib.sha256((HERE / 'manifest.json').read_bytes()).hexdigest(),
               'scope': 'bounded development fit; not a production recipe'}
    ctx = None
    try:
        import torch
        import numpy as np
        from config import Config
        from fit_spiral import FitContext, FitConfig
        from fit_session import conventional_input_paths, load_scroll_spec
        from spiral_helpers import scale_and_split_counts
        import point_collection as pc
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(0.83, 20 * 2**30 / total), 0)
        import subprocess
        summary['villa_head'] = subprocess.check_output(
            ['git', '-C', str(SPIRAL), 'rev-parse', 'HEAD'], text=True).strip()
        z0, z1 = manifest['region_z']
        config = Config().as_dict()
        config.update(z_begin=int(z0), z_end=int(z1), optimizer_random_seed=args.seed,
                      optimizer_num_training_steps=args.steps, output_save_png_visualizations=False)
        for key in config:
            if key.startswith('input_use_'):
                config[key] = key == 'input_use_verified_patches' or (
                    key == 'input_use_fibers' and args.arm in ('A', 'D'))
        for key in ('loss_weight_dense_normals', 'loss_weight_fiber_directions',
                    'loss_weight_dense_spacing', 'loss_weight_dense_spacing_density',
                    'loss_weight_shell_outer', 'loss_weight_shell_patch_radius'):
            if key in config:
                config[key] = 0.0
        keys = ('sample_count_patches_per_step', 'sample_count_patches_per_step_for_dt',
                'sample_count_unattached_pcls_per_step', 'sample_count_regularisation_points',
                'sample_count_dense_attachment_points')
        summary['sample_scaling'] = scale_and_split_counts(config, int(z0), int(z1), keys, world_size=1)
        write_json(out / 'resolved_config.json', config)
        dataset = HERE / 'dataset'
        scroll = load_scroll_spec(dataset)
        paths = conventional_input_paths(dataset, scroll)
        rejected_by_logical = {k: frozenset(v) for k, v in manifest['d_rejected_pairs'].items()}
        telemetry = (out / 'steps.jsonl').open('w', encoding='utf-8', buffering=1)

        class ArmFit(FitContext):
            def _patch_link_options(self, collections, voxel_scale, **kw):
                options = super()._patch_link_options(collections, voxel_scale, **kw)
                if args.arm != 'D':
                    return options
                rejected = {}
                for cid, pcl in collections.items():
                    logical = (pcl.get('metadata') or {}).get('logical_input_id')
                    if logical in rejected_by_logical:
                        rejected[cid] = rejected_by_logical[logical]
                return pc.PatchLinkOptions(
                    window_points=options.window_points, window_min_points=options.window_min_points,
                    side_rules=options.side_rules, inward_direction=options.inward_direction,
                    side_margin=options.side_margin, allowed_patches=options.allowed_patches,
                    rejected_patches=rejected)

            def build_device_state(self):
                super().build_device_state()
                torch.cuda.synchronize()
                cat = getattr(self, 'fiber_catalog', None) or {}
                summary['fiber_attached_points'] = {
                    k: sum('on_patch' in pt for pt in pcl['points'].values()) for k, pcl in cat.items()}
                summary['fiber_attached_patches'] = {
                    k: sorted({pt['on_patch']['id'] for pt in pcl['points'].values() if 'on_patch' in pt})
                    for k, pcl in cat.items()}
                summary['patches_loaded'] = len(self.verified_patches)
                summary['setup_seconds'] = time.perf_counter() - started
                write_json(out / 'summary.json', summary)

            def step(self, iteration):
                if time.perf_counter() - started > args.max_seconds:
                    raise RuntimeError('Wall-time cap reached')
                if iteration % 50 == 0:
                    # Release cached-but-free blocks before the periodic refreshes
                    # (theta crossing map every 100 steps): the first attempt hit a
                    # fragmentation OOM under the 20 GiB cap at step ~2300. No
                    # numerical effect; identical in every arm.
                    torch.cuda.empty_cache()
                t = time.perf_counter()
                result = super().step(iteration)
                torch.cuda.synchronize()
                loss, losses, metrics, shell = result
                row = {'iteration': iteration, 'seconds': time.perf_counter() - t,
                       'loss': float(loss.detach()),
                       'losses': {k: float(v.detach()) for k, v in losses.items()},
                       'cuda_peak_allocated': torch.cuda.max_memory_allocated()}
                telemetry.write(json.dumps(row) + '\n')
                if not np.isfinite(row['loss']):
                    raise RuntimeError('Nonfinite loss')
                if iteration % 100 == 0:
                    print(f'{args.name} step={iteration} sec={row["seconds"]:.3f} '
                          f'peak_GiB={row["cuda_peak_allocated"] / 2**30:.2f}', flush=True)
                return result

        ctx = ArmFit(FitConfig(config), scroll=scroll, paths=paths, run_dir=str(out),
                     cache_dir=str(HERE / 'cache'))
        ctx.run()
        telemetry.close()
        summary.update(status='completed', completed_iterations=args.steps,
                       cuda_peak_allocated=torch.cuda.max_memory_allocated(),
                       cuda_peak_reserved=torch.cuda.max_memory_reserved())
    except BaseException:
        summary.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        summary['elapsed_seconds'] = time.perf_counter() - started
        write_json(out / 'summary.json', summary)
        if ctx is not None:
            ctx.close()


if __name__ == '__main__':
    main()
