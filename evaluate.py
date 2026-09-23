"""Arm comparison on one run: unique fiber length, pairs, links, flags.

Unique accepted length: per fiber, the union of samples where at least one
accepted pair supports the fiber (|d_n| <= 2.5 vox, covered). Duplicate
patches over the same stretch count once. Physical um from the same sample
weights as pair_evidence.

`--labels` (optional JSONL of {fiber_file, patch, label in correct/wrong/unresolved})
adds precision with Clopper-Pearson bounds, clustered counts by parent fiber,
and the conservative variant (accepted unresolved = wrong).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

from pair_evidence import pair_states, sample_weights_um, SUPPORT, CONTRA, DenseIndex


def load_pairs(run_dir):
    with open(os.path.join(run_dir, 'pairs.jsonl')) as handle:
        return [json.loads(line) for line in handle]


def cid_by_file(run_dir):
    report = json.load(open(os.path.join(run_dir, 'replay_report.json')))
    return {v['file']: int(k) for k, v in report['fiber_provenance'].items()}


def unique_lengths(run_dir, pairs, arm):
    dense = DenseIndex(np.load(os.path.join(run_dir, 'dense.npz')))
    cids = cid_by_file(run_dir)
    by_fiber = defaultdict(list)
    for p in pairs:
        if p['decisions'][arm] in ('accept', 'propose'):
            by_fiber[p['fiber_file']].append(p['patch'])
    supported_um, contra_um, total_um = 0.0, 0.0, 0.0
    per_fiber = {}
    for fiber, patches in by_fiber.items():
        cid = cids[fiber]
        w = sample_weights_um(dense['arclen_%d' % cid])
        sup = np.zeros(len(w), bool)
        con = np.zeros(len(w), bool)
        for patch in patches:
            st, _ = pair_states(dense, cid, patch)
            sup |= st == SUPPORT
            con |= st == CONTRA
        per_fiber[fiber] = float(w[sup].sum())
        supported_um += float(w[sup].sum())
        contra_um += float(w[con & ~sup].sum())
    for fiber, cid in cids.items():
        total_um += float(sample_weights_um(dense['arclen_%d' % cid]).sum())
    return supported_um, contra_um, total_um, per_fiber


def clopper_pearson(k, n, alpha=0.05):
    if n == 0:
        return (np.nan, np.nan)
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--labels', default=None)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    pairs = load_pairs(args.run)
    arms = sorted({k for p in pairs for k in p['decisions']})
    labels = {}
    if args.labels:
        with open(args.labels) as handle:
            for line in handle:
                rec = json.loads(line)
                labels[(rec['fiber_file'], rec['patch'])] = rec['label']
    result = {'run': args.run, 'pairs': len(pairs), 'arms': {}}
    for arm in arms:
        acc = [p for p in pairs if p['decisions'][arm] in ('accept', 'propose')]
        sup, con, total, per_fiber = unique_lengths(args.run, pairs, arm)
        entry = {
            'accepted_pairs': len(acc),
            'accepted_native_links': int(sum(p['native_links'] for p in acc)),
            'fibers_with_accepted_pair': len({p['fiber_file'] for p in acc}),
            'unique_supported_fiber_mm': round(sup / 1000, 1),
            'fiber_in_region_mm': round(total / 1000, 1),
            'accepted_pairs_with_contradiction>=150um': int(sum(p['contra_um'] >= 150 for p in acc)),
            'accepted_pairs_short_support<750um': int(sum(p['longest_support_um'] < 750 for p in acc)),
        }
        if labels:
            lab = [labels.get((p['fiber_file'], p['patch'])) for p in acc]
            n_correct = sum(l == 'correct' for l in lab)
            n_wrong = sum(l == 'wrong' for l in lab)
            n_unres = sum(l == 'unresolved' for l in lab)
            resolved = n_correct + n_wrong
            entry['labelled'] = {'correct': n_correct, 'wrong': n_wrong, 'unresolved': n_unres,
                                 'unlabelled': sum(l is None for l in lab)}
            entry['precision_resolved'] = n_correct / resolved if resolved else None
            entry['precision_resolved_95ci'] = clopper_pearson(n_correct, resolved)
            cons_n = n_correct + n_wrong + n_unres
            entry['precision_conservative'] = n_correct / cons_n if cons_n else None
            entry['precision_conservative_95ci'] = clopper_pearson(n_correct, cons_n)
            fibers_with_error = {p['fiber_file'] for p, l in zip(acc, lab) if l == 'wrong'}
            entry['parent_fibers_with_a_wrong_accept'] = len(fibers_with_error)
        result['arms'][arm] = entry
    text = json.dumps(result, indent=1)
    if args.out:
        with open(args.out, 'w') as handle:
            handle.write(text)
    print(text)


if __name__ == '__main__':
    main()
