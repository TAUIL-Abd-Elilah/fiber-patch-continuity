"""Raw-CT review views for fiber-to-patch associations.

Straightened view: x = arc length along the fiber (its control-span-trimmed
polyline, the one villa fits), y = offset along the umbilicus-radial
direction made orthogonal to the fiber tangent (+ = inward). The direction
is patch-independent, so the view does not favour any candidate. Each
candidate patch is drawn where it covers the fiber (in-plane residual <= 1
voxel) at its offset along that direction. Native links are ticks on top.

Cross-section: the plane through one fiber sample spanned by the same radial
direction and the binormal (fiber tangent x radial), so sheets run across.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import ct
import native_adapter as na

COLORS = ['#e8590c', '#1c7ed6', '#2f9e44', '#ae3ec9', '#f59f00', '#0ca678']


def _smooth(a: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(a, sigma, axis=0, mode='nearest')


def _frames(line_zyx: np.ndarray, inward, level: int, step_vox: float):
    """Unit tangent (from a smoothed path) and display direction: the CT
    structure-tensor sheet normal (patch-independent), oriented inward,
    orthogonal to the tangent and smoothed; radial where CT gives none."""
    sig = max(1.0, 25.0 / step_vox)  # ~25 fit voxels of smoothing
    t = np.gradient(_smooth(line_zyx, sig), axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    r = np.asarray(inward(line_zyx), dtype=np.float64)
    every = max(1, int(round(30.0 / step_vox)))
    idx = np.arange(0, len(line_zyx), every)
    scale = 2 ** (2 - level)
    n = ct.sheet_normals(level, line_zyx[idx] * scale, half=12 * scale // 1 if scale > 1 else 12,
                         sigma_tensor=4.0 * scale)
    ok = np.isfinite(n).all(1)
    n[ok] *= np.sign((n[ok] * r[idx][ok]).sum(1))[:, None]
    n[~ok] = r[idx][~ok]
    u = np.stack([np.interp(np.arange(len(line_zyx)), idx, n[:, i]) for i in range(3)], 1)
    u = _smooth(u, max(1.0, 40.0 / step_vox))
    u = u - (u * t).sum(1, keepdims=True) * t
    u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
    return t, u


def straightened(line_zyx, arclen_um, s_lo_um, s_hi_um, inward, half_width=40.0,
                 level=2, step_um=None):
    """CT image (V, S) along the fiber between arc lengths s_lo..s_hi."""
    vox_um = na.UM_PER_VOXEL / (2 ** (2 - level))
    step_um = step_um or vox_um
    s = np.arange(s_lo_um, s_hi_um, step_um)
    p = np.stack([np.interp(s, arclen_um, line_zyx[:, i]) for i in range(3)], 1)
    t, u = _frames(p, inward, level, step_um / na.UM_PER_VOXEL)
    v = np.arange(-half_width, half_width + 1e-6, vox_um / na.UM_PER_VOXEL)
    coords = p[None, :, :] + v[:, None, None] * u[None, :, :]
    img = ct.sample(level, coords * (2 ** (2 - level)))
    return img, s, v, p, t, u


def patch_offsets_on_view(p, u, dense, cid, patch, arclen_samples_um, s):
    """Offset (fit voxels, along u) of a patch at the view columns, from the
    dense evidence feet; NaN where the patch does not cover the fiber."""
    m = (dense['cid'] == cid) & (dense['patch'] == patch) & (dense['r_tangent'] <= 1.0)
    if not m.any():
        return np.full(len(s), np.nan)
    smp = dense['sample'][m]
    feet = dense['foot_zyx'][m]
    s_smp = arclen_samples_um[smp]
    out = np.full(len(s), np.nan)
    order = np.argsort(s_smp)
    s_smp, feet = s_smp[order], feet[order]
    spacing = np.median(np.diff(arclen_samples_um)) if len(arclen_samples_um) > 1 else 77.0
    for j, sj in enumerate(s):
        k = np.searchsorted(s_smp, sj)
        cand = [c for c in (k - 1, k) if 0 <= c < len(s_smp) and abs(s_smp[c] - sj) <= spacing * 0.6]
        if not cand:
            continue
        c = min(cand, key=lambda c: abs(s_smp[c] - sj))
        out[j] = float((feet[c] - p[j]) @ u[j])
    return out


def patch_plane_curve(patch, foot_ij, origin, e_u, e_b, e_t, radius_cells=4.0, step=0.05,
                      slab=0.6):
    """Points of a patch surface lying within `slab` fit voxels of the plane
    through `origin` with normal e_t, in (b, u) plane coordinates (fit voxels).
    Samples the bilinear surface on a fine ij grid around foot_ij."""
    import torch
    h, w = patch.zyxs.shape[:2]
    i0, j0 = float(foot_ij[0]), float(foot_ij[1])
    ii = np.arange(max(0.0, i0 - radius_cells), min(h - 1 - 1e-3, i0 + radius_cells), step)
    jj = np.arange(max(0.0, j0 - radius_cells), min(w - 1 - 1e-3, j0 + radius_cells), step)
    if not len(ii) or not len(jj):
        return np.zeros((0, 2))
    grid = np.stack(np.meshgrid(ii, jj, indexing='ij'), -1).reshape(-1, 2).astype(np.float32)
    zyx, valid = patch.ij_to_zyx(torch.as_tensor(grid))
    zyx = zyx.numpy()[valid.numpy()].astype(np.float64)
    rel = zyx - origin
    near = np.abs(rel @ e_t) <= slab
    rel = rel[near]
    return np.stack([rel @ e_b, rel @ e_u], 1)


def render_review_card(out_png, title, line_zyx, line_arclen_um, dense, cid, patch_objs, labels,
                       inward, positions_um, native_links=None, level=1, half_um=420.0,
                       notes=None):
    """Overview of each candidate's normal offset along the fiber + true-aspect
    CT cross-sections (perpendicular to the fiber) at `positions_um`, with the
    patches' actual intersection curves. patch_objs: {name: tifxyz Patch}."""
    arc_samples = dense['arclen_%d' % cid]
    n_pos = len(positions_um)
    fig = plt.figure(figsize=(3.6 * max(n_pos, 3), 7.4))
    grid = fig.add_gridspec(2, max(n_pos, 3), height_ratios=[1, 1.55])
    ax = fig.add_subplot(grid[0, :])
    lo_s = min(positions_um) - 2500
    hi_s = max(positions_um) + 2500
    for k, (name, label) in enumerate(labels):
        m = (dense['cid'] == cid) & (dense['patch'] == name)
        a = arc_samples[dense['sample'][m]]
        dn = dense['d_normal'][m]
        cov = dense['r_tangent'][m] <= 1.0
        # plotted as patch position relative to the fiber: -d_n (+ = patch inward)
        ax.plot(a[cov] / 1000, -dn[cov] * na.UM_PER_VOXEL, '.', ms=3, color=COLORS[k % len(COLORS)],
                label=label)
        ax.plot(a[~cov] / 1000, -dn[~cov] * na.UM_PER_VOXEL, '.', ms=1, alpha=0.25,
                color=COLORS[k % len(COLORS)])
        lo_s = min(lo_s, a[cov].min() - 500) if cov.any() else lo_s
        hi_s = max(hi_s, a[cov].max() + 500) if cov.any() else hi_s
    ax.axhspan(-2.5 * na.UM_PER_VOXEL, 2.5 * na.UM_PER_VOXEL, color='#ffd43b', alpha=0.25,
               label='native tolerance (2.5 vox)')
    ax.axhline(0, color='k', lw=0.6)
    names = [n for n, _ in labels]
    for arc, name in (native_links or []):
        k = names.index(name) if name in names else None
        ax.plot([arc / 1000], [150], 'v', ms=5,
                color=COLORS[k % len(COLORS)] if k is not None else 'gray')
    for j, pos in enumerate(positions_um):
        ax.axvline(pos / 1000, color='gray', lw=0.8, ls=':')
        ax.text(pos / 1000, -165, 'abc'[j] if j < 3 else str(j), ha='center', fontsize=9)
    ax.set_xlim(max(lo_s, line_arclen_um[0]) / 1000, min(hi_s, line_arclen_um[-1]) / 1000)
    ax.set_ylim(-180, 180)
    ax.set_xlabel('arc length along fiber (mm)   [triangles: native links, coloured by chosen patch]')
    ax.set_ylabel('patch offset from fiber\nalong patch normal (um)')
    ax.legend(fontsize=7, loc='upper left', ncol=2, framealpha=0.7, markerscale=3)
    ax.set_title(title, fontsize=9)

    scale = 2 ** (2 - level)
    step = 1.0 / scale  # fit voxels per CT voxel
    for j, pos in enumerate(positions_um):
        axc = fig.add_subplot(grid[1, j])
        idx = int(np.argmin(np.abs(line_arclen_um - pos)))
        lo_i, hi_i = max(0, idx - 12), min(len(line_zyx) - 1, idx + 12)
        e_t = line_zyx[hi_i] - line_zyx[lo_i]
        e_t /= np.linalg.norm(e_t)
        origin = line_zyx[idx]
        n_ct = ct.sheet_normals(level, origin[None] * scale, half=12 * scale, sigma_tensor=4.0 * scale)[0]
        r = np.asarray(inward(origin[None]), dtype=np.float64)[0]
        e_u = n_ct if np.isfinite(n_ct).all() else r
        e_u = e_u * np.sign(e_u @ r if (e_u @ r) != 0 else 1.0)
        e_u = e_u - (e_u @ e_t) * e_t
        e_u /= np.linalg.norm(e_u)
        e_b = np.cross(e_t, e_u)
        half = half_um / na.UM_PER_VOXEL
        g = np.arange(-half, half + 1e-6, step)
        coords = origin + g[:, None, None] * e_u + g[None, :, None] * e_b
        sec = ct.sample(level, coords * scale)
        vals = sec[sec > 0]
        vmin, vmax = (np.percentile(vals, [1, 99.5]) if len(vals) else (0, 255))
        ext = [-half_um, half_um, -half_um, half_um]
        axc.imshow(sec, cmap='gray', vmin=vmin, vmax=vmax, origin='lower', extent=ext)
        for k, (name, label) in enumerate(labels):
            m = np.flatnonzero((dense['cid'] == cid) & (dense['patch'] == name))
            if not len(m):
                continue
            a = arc_samples[dense['sample'][m]]
            near = m[np.argmin(np.abs(a - pos))]
            if abs(arc_samples[dense['sample'][near]] - pos) > 200:
                continue
            patch = patch_objs[name]
            ij = _foot_ij(patch, dense['foot_zyx'][near])
            curve = patch_plane_curve(patch, ij, origin, e_u, e_b, e_t)
            if len(curve):
                axc.plot(curve[:, 0] * na.UM_PER_VOXEL, curve[:, 1] * na.UM_PER_VOXEL, '.',
                         ms=1.2, color=COLORS[k % len(COLORS)])
        axc.plot([0], [0], 'o', mfc='none', mec='#ffd43b', ms=9, mew=1.5)
        axc.set_xlim(-half_um, half_um)
        axc.set_ylim(-half_um, half_um)
        axc.set_title(f"{'abc'[j] if j < 3 else j}: {pos / 1000:.2f} mm, z={origin[0]:.0f}", fontsize=8)
        axc.tick_params(labelsize=7)
        if j == 0:
            axc.set_ylabel('toward umbilicus (um)', fontsize=8)
    if notes:
        fig.text(0.01, 0.005, notes, fontsize=7, va='bottom')
    fig.tight_layout(rect=(0, 0.03 if notes else 0, 1, 1))
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return {'ct_bytes_downloaded': ct.downloaded_bytes()}


def _foot_ij(patch, foot_zyx):
    """ij of the patch vertex nearest to a foot point (coarse; the curve
    sampler searches a few cells around it)."""
    import torch
    z = patch.zyxs.numpy() if hasattr(patch.zyxs, 'numpy') else np.asarray(patch.zyxs)
    valid = (z != -1).any(-1)
    d = np.linalg.norm(z - np.asarray(foot_zyx)[None, None, :], axis=-1)
    d[~valid] = np.inf
    i, j = np.unravel_index(np.argmin(d), d.shape)
    return float(i), float(j)


def render_pair_view(out_png, title, line_zyx, line_arclen_um, dense, cid, patches, inward,
                     s_lo_um, s_hi_um, native_links=None, xsec_at_um=None, level=2,
                     half_width=40.0, notes=None):
    """patches: list of (patch_name, label). native_links: list of (arc_um, patch_name)."""
    img, s, v, p, t, u = straightened(line_zyx, line_arclen_um, s_lo_um, s_hi_um, inward,
                                      half_width, level)
    arc_samples = dense['arclen_%d' % cid]
    ncols = 2 if xsec_at_um is not None else 1
    fig = plt.figure(figsize=(14 if ncols == 2 else 11, 5.2))
    grid = fig.add_gridspec(1, ncols, width_ratios=[3, 1.25][:ncols])
    ax = fig.add_subplot(grid[0])
    lo, hi = np.percentile(img[img > 0], [1, 99.5]) if (img > 0).any() else (0, 255)
    extent = [s[0] / 1000, s[-1] / 1000, v[0] * na.UM_PER_VOXEL, v[-1] * na.UM_PER_VOXEL]
    ax.imshow(img, cmap='gray', vmin=lo, vmax=hi, aspect='auto', origin='lower', extent=extent,
              interpolation='nearest')
    ax.axhline(0, color='#ffd43b', lw=0.9, ls='--', label='fiber (traced)')
    for k, (name, label) in enumerate(patches):
        off = patch_offsets_on_view(p, u, dense, cid, name, arc_samples, s)
        ax.plot(s / 1000, off * na.UM_PER_VOXEL, '.', ms=2.2, color=COLORS[k % len(COLORS)],
                label=label)
    for arc, name in (native_links or []):
        if s[0] <= arc <= s[-1]:
            k = [n for n, _ in patches].index(name) if name in [n for n, _ in patches] else None
            ax.plot([arc / 1000], [v[-1] * na.UM_PER_VOXEL * 0.93], 'v', ms=5,
                    color=COLORS[k % len(COLORS)] if k is not None else 'white')
    if xsec_at_um is not None:
        ax.axvline(xsec_at_um / 1000, color='white', lw=0.6, alpha=0.6)
    ax.set_xlabel('arc length along fiber (mm)')
    ax.set_ylabel('radial offset from fiber (um, + = toward umbilicus)')
    ax.set_title(title, fontsize=9)
    ax.legend(loc='lower left', fontsize=7, framealpha=0.6, markerscale=4)
    if xsec_at_um is not None:
        ax2 = fig.add_subplot(grid[1])
        j = int(np.argmin(np.abs(s - xsec_at_um)))
        b = np.cross(t[j], u[j])
        w = np.arange(-64, 64.01, 0.5 if level < 2 else 1.0)
        vv = np.arange(-half_width, half_width + 1e-6, 0.5 if level < 2 else 1.0)
        coords = p[j][None, None, :] + vv[:, None, None] * u[j] + w[None, :, None] * b
        sec = ct.sample(level, coords * (2 ** (2 - level)))
        ax2.imshow(sec, cmap='gray', vmin=lo, vmax=hi, origin='lower', aspect='equal',
                   extent=[w[0] * na.UM_PER_VOXEL, w[-1] * na.UM_PER_VOXEL,
                           vv[0] * na.UM_PER_VOXEL, vv[-1] * na.UM_PER_VOXEL])
        ax2.plot([0], [0], 'o', mfc='none', mec='#ffd43b', ms=8)
        for k, (name, label) in enumerate(patches):
            off = patch_offsets_on_view(p[j:j + 1], u[j:j + 1], dense, cid, name, arc_samples, s[j:j + 1])
            if np.isfinite(off[0]):
                ax2.plot([0], [off[0] * na.UM_PER_VOXEL], 'x', color=COLORS[k % len(COLORS)], ms=8)
        ax2.set_title(f'cross-section at {xsec_at_um / 1000:.2f} mm (um)', fontsize=9)
        ax2.set_xlabel('along binormal (um)')
    if notes:
        fig.text(0.01, 0.01, notes, fontsize=7, va='bottom')
    fig.tight_layout(rect=(0, 0.04 if notes else 0, 1, 1))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    zc = p[:, 0]
    return {'z_range_fit': [float(zc.min()), float(zc.max())], 'ct_bytes_downloaded': ct.downloaded_bytes()}
