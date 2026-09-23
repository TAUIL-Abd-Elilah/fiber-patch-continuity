"""Aggregate dense hits into (fiber, patch) pair evidence in physical length.

For every fiber sample the pair is in one state:
  support       patch covers the spot (r_t <= cover_tol) and |d_n| <= support_tol
  contradiction patch covers the spot and |d_n| >= contra_tol
  gray          patch covers the spot, support_tol < |d_n| < contra_tol
  absent        no hit within the envelope, or the nearest point is on the
                patch edge (r_t > cover_tol): the patch says nothing here
Each sample carries the arc length halfway to its neighbours (um), so the
lengths do not depend on the sampling density.
"""
from __future__ import annotations

import numpy as np

SUPPORT, GRAY, CONTRA, ABSENT = 0, 1, 2, 3


def sample_weights_um(arclen_um: np.ndarray) -> np.ndarray:
    if len(arclen_um) == 1:
        return np.zeros(1)
    mids = np.concatenate([[arclen_um[0]], (arclen_um[1:] + arclen_um[:-1]) / 2, [arclen_um[-1]]])
    return np.diff(mids)


class DenseIndex:
    """Wraps a loaded dense.npz: arrays materialised once and a
    (cid, patch) -> row index built once, so per-pair lookups are O(rows)."""

    def __init__(self, npz):
        self._npz = npz
        self._arrays = {}
        cid = self['cid']
        patch = self['patch']
        order = np.lexsort((patch, cid))
        keys = list(zip(cid[order].tolist(), patch[order].tolist()))
        self.rows = {}
        start = 0
        for i in range(1, len(keys) + 1):
            if i == len(keys) or keys[i] != keys[start]:
                self.rows[keys[start]] = order[start:i]
                start = i

    def __getitem__(self, key):
        if key not in self._arrays:
            self._arrays[key] = self._npz[key]
        return self._arrays[key]

    def pair_rows(self, cid, patch):
        return self.rows.get((int(cid), str(patch)), np.zeros(0, dtype=np.int64))


def _rows(dense, cid, patch):
    if isinstance(dense, DenseIndex):
        return dense.pair_rows(cid, patch)
    return np.flatnonzero((dense['cid'] == cid) & (dense['patch'] == patch))


def pair_states(dense, cid: int, patch: str, *, support_tol=2.5, contra_tol=6.0, cover_tol=1.0):
    """(states[S], d_n[S]) for one fiber/patch pair over the fiber's samples."""
    n = len(dense['samples_%d' % cid])
    states = np.full(n, ABSENT, dtype=np.int8)
    dn = np.full(n, np.nan)
    m = _rows(dense, cid, patch)
    idx = dense['sample'][m]
    covered = dense['r_tangent'][m] <= cover_tol
    d = dense['d_normal'][m]
    dn[idx] = d
    ad = np.abs(d)
    st = np.where(~covered, ABSENT, np.where(ad <= support_tol, SUPPORT,
                                             np.where(ad >= contra_tol, CONTRA, GRAY)))
    states[idx] = st
    return states, dn


def runs(mask: np.ndarray):
    """[(start, stop)) runs of True."""
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    diff = np.diff(padded)
    return list(zip(np.flatnonzero(diff == 1), np.flatnonzero(diff == -1)))


def pair_summary(dense, cid, patch, **tols):
    states, dn = pair_states(dense, cid, patch, **tols)
    w = sample_weights_um(dense['arclen_%d' % cid])
    out = {s: float(w[states == code].sum()) for s, code in
           (('support_um', SUPPORT), ('gray_um', GRAY), ('contra_um', CONTRA), ('absent_um', ABSENT))}
    sup = runs(states == SUPPORT)
    out['longest_support_um'] = max((float(w[a:b].sum()) for a, b in sup), default=0.0)
    covered = states != ABSENT
    out['covered_um'] = float(w[covered].sum())
    out['median_dn_support'] = float(np.nanmedian(dn[states == SUPPORT])) if (states == SUPPORT).any() else np.nan
    return out, states, dn
