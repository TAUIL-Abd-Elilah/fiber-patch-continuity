"""Collect held-out evaluations and run summaries for all completed arms."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    rows = []
    for run in sorted((HERE / 'runs').iterdir()):
        ev, sm = run / 'heldout_eval.json', run / 'summary.json'
        if not ev.exists() or not sm.exists() or run.name.startswith('smoke'):
            continue
        e, s = json.loads(ev.read_text()), json.loads(sm.read_text())
        att = s.get('fiber_attached_points', {})
        rows.append({
            'run': run.name, 'arm': e['arm'], 'seed': e['seed'], 'steps': e['steps'],
            'heldout_point_satisfaction': round(e['point_weighted_satisfaction'], 4),
            'heldout_fiber_mean': round(e['fiber_mean_satisfaction'], 4),
            'fully_satisfied_fibers': e['fully_satisfied_fibers'],
            'eval_points': e['eval_points'],
            'median_error_um': round(e['median_error_um'], 1) if e['median_error_um'] is not None else None,
            'fit_fiber_attached_points': sum(att.values()) if att else 0,
            'fit_fiber_patches_touched': len({p for v in s.get('fiber_attached_patches', {}).values() for p in v}),
            'elapsed_min': round(s['elapsed_seconds'] / 60, 1),
            'peak_alloc_GiB': round(s.get('cuda_peak_allocated', 0) / 2**30, 2),
            'per_fiber': {r['fiber']: round(r['fraction'], 3) for r in e['per_fiber']},
        })
    (HERE / 'comparison.json').write_text(json.dumps(rows, indent=1))
    for r in rows:
        print({k: v for k, v in r.items() if k != 'per_fiber'})
    fibers = sorted({f for r in rows for f in r['per_fiber']})
    print('\nper held-out fiber satisfaction:')
    print('fiber'.ljust(38), ' '.join(r['run'][:12].rjust(12) for r in rows))
    for f in fibers:
        print(f[:38].ljust(38), ' '.join(str(r['per_fiber'].get(f, '-')).rjust(12) for r in rows))


if __name__ == '__main__':
    main()
