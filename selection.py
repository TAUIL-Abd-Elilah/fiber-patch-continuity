"""Pair records and arm decisions on one run's identical candidate population.

Population: every (fiber, patch) pair with at least one decimated fiber point
within the native tolerance (the pairs the native linker could choose from).

Arms (all on that population):
  A        native defaults: accepted iff the unmodified linker links >= 1 point
  B_*      native variants (window gate / side filter / tighter tolerance)
  C        point-local margin: >= 2 native links, median native distance <= 1.5 vox,
           and at those points the nearest non-coincident alternative surface
           (|delta d_n| > 1 vox) is >= 4 vox away along the normal
  D        continuity: follow the patch along the fiber; propose iff the longest
           supported run >= 750 um, in-domain contradiction < 150 um, and the
           decision is unchanged under the perturbation grid; else review
Development values, frozen here before any cal/test output (rev 2 contract).
"""
from __future__ import annotations

import itertools
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

import native_adapter as na
from pair_evidence import pair_summary, DenseIndex

SCHEMA_VERSION = 'fiber-patch-assoc/0.1'
UM = na.UM_PER_VOXEL

D_POLICY = {'support_tol': 2.5, 'contra_tol': 6.0, 'cover_tol': 1.0,
            'min_longest_support_um': 750.0, 'max_contra_um': 150.0}
D_PERTURB = {'support_tol': (2.0, 3.0), 'contra_tol': (5.0, 7.0), 'max_contra_um': (100.0, 200.0)}
C_POLICY = {'min_links': 2, 'max_median_dist_vox': 1.5, 'coincident_vox': 1.0, 'min_margin_vox': 4.0}


@dataclass
class PairRecord:
    schema_version: str
    source_revision: str
    configuration_hash: str
    acquisition_id: str
    reference_frame: str
    fiber_file: str
    fiber_sha256: str
    parent_fiber_id: str
    span_id: str
    z_band: List[float]
    patch: str
    patch_family: str
    verified_equivalence_group_id: Optional[str]
    native_point_ids: List[int]
    native_links: int
    within_tol_points: int
    native_median_distance_um: Optional[float]
    support_um: float
    longest_support_um: float
    contra_um: float
    gray_um: float
    covered_um: float
    median_offset_supported_um: Optional[float]
    coincident_patches: List[str]
    alternative_patches: List[dict]
    c_margin_vox: Optional[float]
    decisions: Dict[str, str] = field(default_factory=dict)
    reason_codes: Dict[str, List[str]] = field(default_factory=dict)
    review_status: str = 'unreviewed'
    reviewer_id: Optional[str] = None


def d_decision(dense, cid, patch, policy=D_POLICY):
    s, states, _ = pair_summary(dense, cid, patch, support_tol=policy['support_tol'],
                                contra_tol=policy['contra_tol'], cover_tol=policy['cover_tol'])
    reasons = []
    if s['contra_um'] >= policy['max_contra_um']:
        reasons.append('in_domain_contradiction')
    if s['longest_support_um'] < policy['min_longest_support_um']:
        reasons.append('short_support')
    return ('propose' if not reasons else 'review'), reasons, s, states


def d_decision_stable(dense, cid, patch):
    base, reasons, s, states = d_decision(dense, cid, patch)
    flips = []
    for key, values in D_PERTURB.items():
        for value in values:
            pol = dict(D_POLICY, **{key: value})
            alt, _, _, _ = d_decision(dense, cid, patch, pol)
            if alt != base:
                flips.append(f'{key}={value}')
    if base == 'propose' and flips:
        reasons = reasons + ['unstable_under_perturbation']
        base = 'review'
    return base, reasons, s, states, flips


def c_margin(cap, dense_unused, cid, patch, point_positions):
    """Median over the pair's native-linked points of the normal-offset gap to
    the nearest alternative surface that is not coincident at that point."""
    margins = []
    for pos in point_positions:
        m = (cap['collection_id'] == cid) & (cap['point_pos'] == pos)
        rows = np.flatnonzero(m)
        this = rows[cap['patch'][rows] == patch]
        if not len(this):
            continue
        # signed offset along the umbilicus direction of each hit (side_offset);
        # fall back to distance when unknown
        def off(r):
            v = cap['side_offset'][r]
            return v if np.isfinite(v) else cap['distance'][r]
        o_this = off(this[0])
        gaps = [abs(off(r) - o_this) for r in rows if cap['patch'][r] != patch]
        gaps = [g for g in gaps if g > C_POLICY['coincident_vox']]
        margins.append(min(gaps) if gaps else 16.0)
    return float(np.median(margins)) if margins else None


def build_records(run_dir: str, splits: Optional[dict] = None, split_name: Optional[str] = None):
    cap = dict(np.load(os.path.join(run_dir, 'capture.npz')))
    dense = DenseIndex(np.load(os.path.join(run_dir, 'dense.npz')))
    report = json.load(open(os.path.join(run_dir, 'replay_report.json')))
    meta = json.load(open(os.path.join(run_dir, 'patch_meta.json')))
    variants = json.load(open(os.path.join(run_dir, 'variant_links.json')))
    prov = report['fiber_provenance']
    tol = cap['within_tol']
    pairs = sorted(set(zip(cap['collection_id'][tol].tolist(), cap['patch'][tol].tolist())))
    win = cap['native_winner']
    native_pts = defaultdict(list)
    for c, p, pos, pid in zip(cap['collection_id'][win], cap['patch'][win], cap['point_pos'][win],
                              cap['point_id'][win]):
        native_pts[(int(c), str(p))].append((int(pos), int(pid)))
    tol_count = defaultdict(int)
    for c, p in zip(cap['collection_id'][tol], cap['patch'][tol]):
        tol_count[(int(c), str(p))] += 1
    variant_pairs = {name: defaultdict(int) for name in variants}
    for name, v in variants.items():
        for c, pid, patch in v['links']:
            variant_pairs[name][(int(c), patch)] += 1

    # support states for coincidence tests, cached per pair
    states_cache = {}

    def states_of(c, p):
        if (c, p) not in states_cache:
            _, _, s, st = d_decision(dense, c, p)
            states_cache[(c, p)] = (s, st)
        return states_cache[(c, p)]

    covering = defaultdict(set)
    rt_all = dense['r_tangent']
    for (c, p), rows in dense.rows.items():
        if (rt_all[rows] <= D_POLICY['cover_tol']).any():
            covering[c].add(p)

    records = []
    for cid, patch in pairs:
        file = prov[str(cid)]['file']
        if splits is not None and splits['assignment'].get(file) != split_name:
            continue
        decision_d, reasons_d, s, states, flips = d_decision_stable(dense, cid, patch)
        sup_mask = states == 0
        coincident, alternatives = [], []
        for other in sorted(covering[cid] - {patch}):
            so, st_o = states_of(cid, other)
            both = (sup_mask & (st_o == 0)).sum()
            if sup_mask.sum() and both >= 0.5 * sup_mask.sum():
                coincident.append(other)
            elif so['covered_um'] > 0:
                alternatives.append({'patch': other, 'support_um': round(so['support_um']),
                                     'contra_um': round(so['contra_um']),
                                     'covered_um': round(so['covered_um'])})
        alternatives = sorted(alternatives, key=lambda a: -a['covered_um'])[:5]
        nat = native_pts.get((cid, patch), [])
        dists = [float(cap['distance'][np.flatnonzero((cap['collection_id'] == cid) &
                                                      (cap['point_pos'] == pos) &
                                                      (cap['patch'] == patch))[0]])
                 for pos, _ in nat]
        margin = c_margin(cap, dense, cid, patch, [pos for pos, _ in nat])
        rec = PairRecord(
            schema_version=SCHEMA_VERSION, source_revision=report['villa_revision'],
            configuration_hash=report['config_hash'],
            acquisition_id='PHercParis4/20260411134726',
            reference_frame='fit frame zyx, 9.6 um voxels (level 2 of 20260411134726)',
            fiber_file=file, fiber_sha256=prov[str(cid)]['sha256'], parent_fiber_id=file,
            span_id=f"{file}@z[{report['z_band'][0]:.0f},{report['z_band'][1]:.0f})",
            z_band=list(report['z_band']), patch=patch, patch_family=meta[patch]['family'],
            verified_equivalence_group_id=None,
            native_point_ids=[pid for _, pid in nat], native_links=len(nat),
            within_tol_points=tol_count[(cid, patch)],
            native_median_distance_um=float(np.median(dists)) * UM if dists else None,
            support_um=round(s['support_um'], 1), longest_support_um=round(s['longest_support_um'], 1),
            contra_um=round(s['contra_um'], 1), gray_um=round(s['gray_um'], 1),
            covered_um=round(s['covered_um'], 1),
            median_offset_supported_um=(float(s['median_dn_support']) * UM
                                        if np.isfinite(s['median_dn_support']) else None),
            coincident_patches=coincident, alternative_patches=alternatives, c_margin_vox=margin)
        rec.decisions['A'] = 'accept' if nat else 'not_linked'
        for name in variant_pairs:
            rec.decisions[name] = 'accept' if variant_pairs[name].get((cid, patch)) else 'not_linked'
        c_ok = (len(nat) >= C_POLICY['min_links'] and dists
                and np.median(dists) <= C_POLICY['max_median_dist_vox']
                and margin is not None and margin >= C_POLICY['min_margin_vox'])
        rec.decisions['C'] = 'accept' if c_ok else 'reject'
        rec.decisions['D'] = decision_d
        rec.reason_codes['D'] = reasons_d + ([f'flips:{",".join(flips)}'] if flips else [])
        records.append(rec)
    return records


def summarize(records: List[PairRecord]):
    arms = sorted({k for r in records for k in r.decisions})
    out = {'pairs': len(records)}
    for arm in arms:
        acc = [r for r in records if r.decisions[arm] in ('accept', 'propose')]
        out[arm] = {'accepted_pairs': len(acc),
                    'accepted_native_links': int(sum(r.native_links for r in acc)),
                    'accepted_support_mm': round(sum(r.support_um for r in acc) / 1000, 1),
                    'accepted_with_contradiction>=150um': int(sum(r.contra_um >= 150 for r in acc))}
    return out


def save(records, path):
    with open(path, 'w') as handle:
        for r in records:
            handle.write(json.dumps(asdict(r)) + '\n')
