"""How much of each native pair's contradiction is continuous with its support.

A covered sample continues the previous one when its projection foot moved no
more than the fiber did (+3 fit voxels): the patch surface is being followed
without jumping to another part of the patch. Contradiction is 'connected'
when it lies in the same continuous run as a supported sample.
"""
import argparse
import json
import os

import numpy as np

from pair_evidence import ABSENT, CONTRA, SUPPORT, DenseIndex, pair_states, sample_weights_um


def connected_contradiction(dense, cid, patch):
    st, _ = pair_states(dense, cid, patch)
    rows = dense.pair_rows(cid, patch)
    feet = np.full((len(st), 3), np.nan)
    feet[dense['sample'][rows]] = dense['foot_zyx'][rows]
    smp = dense['samples_%d' % cid].astype(float)
    w = sample_weights_um(dense['arclen_%d' % cid])
    comp = np.full(len(st), -1)
    c = 0
    for i in range(len(st)):
        if st[i] == ABSENT:
            continue
        if (i > 0 and comp[i - 1] >= 0 and
                np.linalg.norm(feet[i] - feet[i - 1]) <= np.linalg.norm(smp[i] - smp[i - 1]) + 3):
            comp[i] = comp[i - 1]
        else:
            comp[i] = c
            c += 1
    support_comps = set(comp[st == SUPPORT].tolist())
    contra = st == CONTRA
    connected = contra & np.isin(comp, list(support_comps))
    return float(w[contra].sum()), float(w[connected].sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    dense = DenseIndex(np.load(os.path.join(args.run, 'dense.npz')))
    rep = json.load(open(os.path.join(args.run, 'replay_report.json')))
    cids = {v['file']: int(k) for k, v in rep['fiber_provenance'].items()}
    pairs = [json.loads(l) for l in open(os.path.join(args.run, 'pairs.jsonl'))]
    native = [p for p in pairs if p['decisions']['A'] == 'accept']
    rows = []
    for p in native:
        total, conn = connected_contradiction(dense, cids[p['fiber_file']], p['patch'])
        rows.append({'fiber_file': p['fiber_file'], 'patch': p['patch'], 'native_links': p['native_links'],
                     'contra_um': round(total, 1), 'connected_contra_um': round(conn, 1)})
    flagged = [r for r in rows if r['contra_um'] >= 150]
    conn = [r for r in rows if r['connected_contra_um'] >= 150]
    out = {
        'run': args.run, 'native_pairs': len(rows),
        'native_links': int(sum(r['native_links'] for r in rows)),
        'pairs_contra>=150um': len(flagged),
        'pairs_connected_contra>=150um': len(conn),
        'links_on_connected_contra_pairs': int(sum(r['native_links'] for r in conn)),
        'connected_fraction_of_contra_length': (sum(r['connected_contra_um'] for r in flagged) /
                                                max(1e-9, sum(r['contra_um'] for r in flagged))),
    }
    with open(args.out, 'w') as handle:
        json.dump({'summary': out, 'pairs': rows}, handle, indent=1)
    print(json.dumps(out))


if __name__ == '__main__':
    main()
