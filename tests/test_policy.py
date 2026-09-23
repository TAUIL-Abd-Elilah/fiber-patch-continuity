"""Code-behaviour tests on synthetic arrays. They check the contract of the
evidence and decision code; they say nothing about scroll accuracy."""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from pair_evidence import (ABSENT, CONTRA, GRAY, SUPPORT, DenseIndex, pair_states,  # noqa: E402
                           pair_summary, runs, sample_weights_um)
import selection  # noqa: E402


def make_dense(offsets_by_patch, spacing_vox=8.0, n=40, r_t=None):
    """One fiber (cid 1) along x, samples every `spacing_vox`; each patch
    gives a normal offset per sample (NaN = no hit)."""
    samples = np.stack([np.zeros(n), np.zeros(n), np.arange(n) * spacing_vox], 1).astype(np.float32)
    arclen = np.arange(n) * spacing_vox * 9.6
    rows = {k: [] for k in ('cid', 'sample', 'patch', 'distance', 'd_normal', 'r_tangent', 'foot_zyx')}
    for patch, offs in offsets_by_patch.items():
        for i, d in enumerate(offs):
            if np.isnan(d):
                continue
            rows['cid'].append(1)
            rows['sample'].append(i)
            rows['patch'].append(patch)
            rows['distance'].append(abs(d))
            rows['d_normal'].append(d)
            rows['r_tangent'].append(0.0 if r_t is None else r_t[patch][i])
            rows['foot_zyx'].append(samples[i] + np.array([0, d, 0]))
    out = {k: np.asarray(v) for k, v in rows.items()}
    out['samples_1'] = samples
    out['arclen_1'] = arclen
    return out


def test_states_and_lengths_are_physical():
    offs = np.r_[np.zeros(10), np.full(10, 3.5), np.full(10, 9.0), np.full(10, np.nan)]
    dense = make_dense({'p': offs})
    st, _ = pair_states(dense, 1, 'p')
    assert (st[:10] == SUPPORT).all() and (st[10:20] == GRAY).all()
    assert (st[20:30] == CONTRA).all() and (st[30:] == ABSENT).all()
    s, _, _ = pair_summary(dense, 1, 'p')
    # 10 samples x 76.8 um, minus half a sample at the fiber start
    assert s['support_um'] == pytest.approx(9.5 * 76.8, rel=1e-6)


def test_density_does_not_inflate_support():
    coarse = make_dense({'p': np.zeros(20)}, spacing_vox=8.0, n=20)
    fine = make_dense({'p': np.zeros(80)}, spacing_vox=2.0, n=80)
    a, _, _ = pair_summary(coarse, 1, 'p')
    b, _, _ = pair_summary(fine, 1, 'p')
    assert a['support_um'] == pytest.approx(b['support_um'], rel=0.05)


def test_edge_clipped_hits_are_absent_not_contradiction():
    offs = np.r_[np.zeros(20), np.full(20, 9.0)]
    rt = {'p': np.r_[np.zeros(20), np.full(20, 5.0)]}  # patch ends: foot clamped to its edge
    dense = make_dense({'p': offs}, r_t=rt)
    st, _ = pair_states(dense, 1, 'p')
    assert (st[20:] == ABSENT).all()
    decision, reasons, *_ = selection.d_decision_stable(dense, 1, 'p')
    assert decision == 'propose', reasons


def test_contradiction_routes_to_review():
    offs = np.r_[np.zeros(20), np.full(20, 10.0)]
    dense = make_dense({'p': offs})
    decision, reasons, *_ = selection.d_decision_stable(dense, 1, 'p')
    assert decision == 'review' and 'in_domain_contradiction' in reasons


def test_single_touch_routes_to_review():
    offs = np.full(40, np.nan)
    offs[18:21] = [3.0, 0.5, 3.0]
    dense = make_dense({'p': offs})
    decision, reasons, *_ = selection.d_decision_stable(dense, 1, 'p')
    assert decision == 'review' and 'short_support' in reasons


def test_threshold_edge_is_unstable():
    # contradiction just under the 150 um cutoff: one perturbation flips it
    offs = np.r_[np.zeros(38), np.full(2, 6.5)]  # 1.5 samples = 115 um
    dense = make_dense({'p': offs})
    decision, reasons, s, _, flips = selection.d_decision_stable(dense, 1, 'p')
    assert s['contra_um'] < 150
    assert decision == 'review' and 'unstable_under_perturbation' in reasons and flips


def test_nan_geometry_is_absent():
    offs = np.r_[np.zeros(20), np.full(20, np.nan)]
    dense = make_dense({'p': offs})
    st, _ = pair_states(dense, 1, 'p')
    assert (st[20:] == ABSENT).all()


def test_equal_candidates_are_decided_independently():
    dense = make_dense({'a': np.zeros(40), 'b': np.zeros(40)})
    da, *_ = selection.d_decision_stable(dense, 1, 'a')
    db, *_ = selection.d_decision_stable(dense, 1, 'b')
    assert da == db == 'propose'


def test_dense_index_matches_mask_lookup():
    dense = make_dense({'a': np.zeros(10), 'b': np.r_[np.zeros(5), np.full(5, 9.0)]}, n=10)

    class Npz(dict):
        pass
    idx = DenseIndex(Npz(dense))
    for patch in ('a', 'b'):
        s1, _ = pair_states(dense, 1, patch)
        s2, _ = pair_states(idx, 1, patch)
        assert (s1 == s2).all()


def test_runs():
    assert runs(np.array([0, 1, 1, 0, 1], bool)) == [(1, 3), (4, 5)]
    assert runs(np.zeros(3, bool)) == []


def test_weights_sum_to_span():
    arc = np.array([0.0, 10.0, 30.0, 60.0])
    assert sample_weights_um(arc).sum() == pytest.approx(60.0)


def test_frame_check_rejects_undeclared(tmp_path):
    import native_adapter as na
    path = tmp_path / 'f.json'
    path.write_text(json.dumps({'control_points': [], 'line_points': []}))
    with pytest.raises(na.FrameError):
        na.check_fiber_frame(str(path))
    path.write_text(json.dumps({'coordinate_base_shape_zyx': [100, 100, 100]}))
    with pytest.raises(na.FrameError):
        na.check_fiber_frame(str(path))
    path.write_text(json.dumps({'coordinate_base_shape_zyx': [75784, 32694, 32694]}))
    assert na.check_fiber_frame(str(path))['_frame_rule'] == 'declared_l0_shape'
