"""Render review views for chosen (fiber, patch) cases of one run.

Case spec JSON: [{"cid": int, "patches": [prefix,...], "margin_um": 1500,
                  "xsec": "contra"|"support"|float_um, "name": str}]
Candidate patches are the given prefixes plus, optionally, every other patch
that covers the fiber in the window (--alternatives).
"""
import argparse
import json
import os

import numpy as np

import evidence as ev
import native_adapter as na
import review_export as rx
from pair_evidence import pair_states, SUPPORT, CONTRA, ABSENT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--cases', required=True)
    parser.add_argument('--fiber-roots', nargs='+', default=[
        'C:/VesuviusProgressData/PHercParis4/eval_fibers',
        'C:/VesuviusProgressData/PHercParis4/eval_fibers_remote_20260923'])
    parser.add_argument('--umbilicus', default='C:/VesuviusProgressData/PHercParis4/umbilicus.json')
    parser.add_argument('--level', type=int, default=1)
    parser.add_argument('--patch-root', default='C:/VesuviusProgressData/PHercParis4/verified_patches')
    parser.add_argument('--alternatives', type=int, default=2)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dense = np.load(os.path.join(args.run, 'dense.npz'))
    cap = np.load(os.path.join(args.run, 'capture.npz'))
    report = json.load(open(os.path.join(args.run, 'replay_report.json')))
    inward = na.umbilicus_inward(args.umbilicus)
    cases = json.load(open(args.cases))
    results = []
    for case in cases:
        cid = int(case['cid'])
        fname = report['fiber_provenance'][str(cid)]['file']
        path = next(os.path.join(r, fname) for r in args.fiber_roots if os.path.exists(os.path.join(r, fname)))
        _, _, _, line, line_arc = ev.resample_fiber(path, 8.0)
        present = sorted(set(dense['patch'][dense['cid'] == cid].tolist()))
        chosen = []
        for pref in case['patches']:
            hits = [p for p in present if p == pref] or [p for p in present if p.startswith(pref)]
            if len(hits) != 1:
                raise ValueError(f'patch spec {pref!r} matches {len(hits)} patches: {hits[:4]}')
            chosen.append(hits[0])
        st0, _ = pair_states(dense, cid, chosen[0])
        arc_s = dense['arclen_%d' % cid]
        covered = np.flatnonzero(st0 != ABSENT)
        margin = case.get('margin_um', 1500)
        s_lo = max(line_arc[0], arc_s[covered].min() - margin)
        s_hi = min(line_arc[-1], arc_s[covered].max() + margin)
        # add the strongest other covering patches in the window as alternatives
        if args.alternatives:
            rows = np.flatnonzero((dense['cid'] == cid) & (dense['r_tangent'] <= 1.0))
            sa = arc_s[dense['sample'][rows]]
            rows = rows[(sa >= s_lo) & (sa <= s_hi)]
            counts = {}
            for pname in dense['patch'][rows]:
                if pname not in chosen:
                    counts[pname] = counts.get(pname, 0) + 1
            chosen += [p for p, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:args.alternatives]]
        labels = []
        for k, pname in enumerate(chosen):
            role = 'linked' if k < len(case['patches']) else 'other covering patch'
            fam = 'legacy' if not pname.startswith('band-seed') else 'band-seed'
            labels.append((pname, f'{role}: {pname[:34]} ({fam})'))
        # native link arc positions for this fiber
        w = cap['native_winner'] & (cap['collection_id'] == cid)
        orig = cap['nativeorig_%d' % cid]
        dense_orig = dense['orig_%d' % cid]
        full_arc = line_arc
        links = [(float(full_arc[orig[pos]]), str(pn)) for pos, pn in zip(cap['point_pos'][w], cap['patch'][w])]
        xs = case.get('xsec')
        if xs == 'contra' and (st0 == CONTRA).any():
            xs = float(arc_s[np.flatnonzero(st0 == CONTRA)[len(np.flatnonzero(st0 == CONTRA)) // 2]])
        elif xs == 'support' and (st0 == SUPPORT).any():
            xs = float(arc_s[np.flatnonzero(st0 == SUPPORT)[len(np.flatnonzero(st0 == SUPPORT)) // 2]])
        elif not isinstance(xs, (int, float)):
            xs = None
        hv = report['fiber_provenance'][str(cid)]['hv']
        title = (f"{case['name']}: fiber {fname} (auto tag {hv.get('automatic_tag')} "
                 f"{hv.get('automatic_certainty', 0):.2f}); PHercParis4 20260411134726, "
                 f"CT level {args.level}")
        out_png = os.path.join(args.out, f"{case['name']}.png")
        positions = case.get('positions_um')
        if positions is None:
            contra = np.flatnonzero(st0 == CONTRA)
            sup = np.flatnonzero(st0 == SUPPORT)
            positions = []
            if len(contra):
                positions.append(float(arc_s[contra[len(contra) // 2]]))
            if len(sup):
                positions += [float(arc_s[sup[0]]), float(arc_s[sup[len(sup) // 2]])]
            positions = sorted(positions)[:3] or [float(arc_s[covered[len(covered) // 2]])]
        patch_objs, _ = na.load_patches(args.patch_root, chosen, 1)
        info = rx.render_review_card(out_png, title, line.astype(np.float64), line_arc, dense, cid,
                                     patch_objs, labels, inward, positions, native_links=links,
                                     level=args.level, notes=case.get('notes'))
        info['positions_um'] = positions
        info.update(case=case, png=out_png, patches=chosen, fiber=fname, window_um=[s_lo, s_hi])
        results.append(info)
        print(json.dumps(info, default=str))
    with open(os.path.join(args.out, 'cases_rendered.json'), 'w') as handle:
        json.dump(results, handle, indent=1, default=str)


if __name__ == '__main__':
    main()
