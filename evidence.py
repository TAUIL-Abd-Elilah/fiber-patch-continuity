"""Dense per-(fiber, patch) geometric evidence along each fiber.

The native linker looks at one decimated fiber point (40 fit voxels apart by
default) and one distance at a time. Here every fiber is re-sampled along
the SAME control-span-trimmed polyline villa uses (villa's own loader, a
finer decimation) and every patch surface within the envelope is recorded
with:
  * normal offset  d_n = (p - foot) . n   (n = patch normal at the foot,
    oriented to the umbilicus inward direction, so + means the fiber lies
    inward of the patch)
  * in-plane residual r_t = |(p - foot) - d_n n|: ~0 when p projects onto
    the patch interior, large when the nearest point is clamped to the patch
    edge (the patch does not cover this spot).

Evidence is reported in physical arc length (micrometres), never in sample
counts, so sampling density cannot inflate support.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch

import native_adapter as na


@dataclass
class DenseEvidence:
    cid: np.ndarray          # per hit
    sample: np.ndarray       # index into the fiber's dense samples
    patch: np.ndarray
    distance: np.ndarray     # fit voxels
    d_normal: np.ndarray     # signed, fit voxels (+ = fiber inward of patch)
    r_tangent: np.ndarray    # fit voxels
    foot_zyx: np.ndarray
    normal_zyx: np.ndarray
    # per fiber
    samples: Dict[int, np.ndarray]     # cid -> (S,3) zyx
    arclen_um: Dict[int, np.ndarray]   # cid -> (S,) arc length along trimmed line
    orig_index: Dict[int, np.ndarray]  # cid -> (S,) index into trimmed dense line


def resample_fiber(path: str, spacing: float):
    """(zyx samples, trimmed-line indices, arc length um of each sample) via
    villa's loader: same frame check, same control-span trim."""
    _, spiral_helpers, _, _ = na.import_villa()
    _, full = na.villa_load_fiber(path, 0, 0.0)
    pts = np.stack([full['points'][k]['p'] for k in sorted(full['points'])])[:, ::-1].astype(np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arclen = np.concatenate([[0.0], np.cumsum(seg)]) * na.UM_PER_VOXEL
    _, keep = spiral_helpers._decimate_ordered_points_min_spacing(pts, spacing, return_indices=True)
    return pts[keep].astype(np.float32), keep, arclen[keep], pts, arclen


def patch_normals(patch, ij: np.ndarray) -> np.ndarray:
    """Unit normal (zyx) of a tifxyz patch at fractional ij via central
    differences of its bilinear map; NaN where not evaluable."""
    h, w = patch.zyxs.shape[:2]
    ij = torch.as_tensor(np.asarray(ij, dtype=np.float32))
    eps = 0.25
    out = np.full((len(ij), 3), np.nan, dtype=np.float64)
    di = torch.tensor([eps, 0.0])
    dj = torch.tensor([0.0, eps])

    def at(q):
        q = q.clone()
        q[:, 0] = q[:, 0].clamp(0.0, h - 1 - 1e-3)
        q[:, 1] = q[:, 1].clamp(0.0, w - 1 - 1e-3)
        z, v = patch.ij_to_zyx(q)
        return z.numpy().astype(np.float64), v.numpy()

    a, va = at(ij + di)
    b, vb = at(ij - di)
    c, vc = at(ij + dj)
    d, vd = at(ij - dj)
    ok = va & vb & vc & vd
    n = np.cross(a - b, c - d)
    norm = np.linalg.norm(n, axis=1)
    ok &= norm > 1e-9
    out[ok] = n[ok] / norm[ok, None]
    return out


def dense_capture(fiber_paths: Dict[int, str], patches, envelope: float, inward,
                  spacing: float = 8.0, z_band=None, index=None) -> DenseEvidence:
    """Envelope hits for dense samples of each fiber (cid -> source path)."""
    if index is None:
        index, surface_ids = na.build_index(patches, envelope)
    else:
        index, surface_ids = index
    pc, _, _, _ = na.import_villa()
    rows = {k: [] for k in ('cid', 'sample', 'surf', 'dist', 'ij')}
    samples, arcs, origs = {}, {}, {}
    for cid, path in sorted(fiber_paths.items()):
        zyx, orig, arc, _, _ = resample_fiber(path, spacing)
        if z_band is not None:
            keep = (zyx[:, 0] >= z_band[0]) & (zyx[:, 0] < z_band[1])
            zyx, orig, arc = zyx[keep], orig[keep], arc[keep]
        samples[cid], arcs[cid], origs[cid] = zyx, arc, orig
        if not len(zyx):
            continue
        off, s, d, ij = index.locate_all_xyz_batch(np.ascontiguousarray(zyx[:, ::-1]), envelope)
        p = pc._hits_from_offsets(off)
        rows['cid'].extend([cid] * len(p))
        rows['sample'].extend(p.tolist())
        rows['surf'].extend(s.tolist())
        rows['dist'].extend(d.tolist())
        rows['ij'].extend(ij.tolist() if hasattr(ij, 'tolist') else list(ij))
    cid = np.asarray(rows['cid'], dtype=np.int64)
    sample = np.asarray(rows['sample'], dtype=np.int64)
    surf = np.asarray(rows['surf'], dtype=np.int64)
    ij = np.asarray(rows['ij'], dtype=np.float32).reshape(-1, 2)
    feet = np.full((len(surf), 3), np.nan)
    normals = np.full((len(surf), 3), np.nan)
    for s_idx in np.unique(surf):
        m = np.flatnonzero(surf == s_idx)
        patch = patches[surface_ids[s_idx]]
        z, v = patch.ij_to_zyx(torch.as_tensor(ij[m]))
        if not bool(v.all()):
            h, w = patch.zyxs.shape[:2]
            q = torch.as_tensor(ij[m]).clone()
            q[:, 0] = q[:, 0].clamp(0.0, h - 1 - 1e-3)
            q[:, 1] = q[:, 1].clamp(0.0, w - 1 - 1e-3)
            z2, v2 = patch.ij_to_zyx(q)
            fill = (~v) & v2
            z[fill] = z2[fill]
            v = v | v2
        vz = v.numpy()
        feet[m[vz]] = z.numpy()[vz]
        normals[m] = patch_normals(patch, ij[m])
    pts = np.stack([samples[c][k] for c, k in zip(cid, sample)]).astype(np.float64) if len(cid) else np.zeros((0, 3))
    vec = pts - feet
    inward_dir = np.asarray(inward(pts), dtype=np.float64).reshape(-1, 3) if len(pts) else np.zeros((0, 3))
    orient = np.sign((normals * inward_dir).sum(1))
    orient[orient == 0] = 1.0
    n_or = normals * orient[:, None]
    d_n = (vec * n_or).sum(1)
    r_t = np.linalg.norm(vec - d_n[:, None] * n_or, axis=1)
    return DenseEvidence(
        cid=cid, sample=sample, patch=np.asarray([surface_ids[s] for s in surf]),
        distance=np.asarray(rows['dist'], dtype=np.float32), d_normal=d_n, r_tangent=r_t,
        foot_zyx=feet, normal_zyx=n_or, samples=samples, arclen_um=arcs, orig_index=origs)
