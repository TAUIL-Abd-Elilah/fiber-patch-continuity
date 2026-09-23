"""Capture villa's native fiber-to-patch candidates without changing its output.

Everything that decides an association is villa's own code, imported from a
pinned checkout: fiber loading (spiral_helpers.load_fiber_point_collection),
patch loading/erosion (spiral_helpers.load_patch_payload_chunk), the native
surface index (vc_spiral.surface_index) and the selection
(point_collection._link_from_hits / link_points_to_patches).

This module adds two things only:
  * an envelope query (a second index, padded wider than the link
    tolerance) so the competing surfaces just outside tolerance are recorded;
  * per-hit evidence the native path computes but discards: projection foot,
    signed offset along the umbilicus inward direction, window hit counts.

`replay_matches_native` runs the unmodified linker on a deep copy of the same
fibers and checks that the instrumented replay chose the same patch, distance
and ij for every point.

Units: every coordinate here is in the PHercParis4 fit frame (9.6 um voxels,
zyx for points, as villa uses them). `UM_PER_VOXEL` converts to micrometres.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

UM_PER_VOXEL = 9.6
EXPECTED_FIBER_FRAME = {
    'vc_open_data_coordinate_space': 'PHercParis4/20260411134726@L0',
    'vc_open_data_source_coordinate_level': 0,
    'vc_open_data_source_coordinate_scale_factor': 1,
    'vc_open_data_source_original_resolution': 2.4,
}
FIBER_TO_FIT_SCALE = 0.25  # 2.4 um L0 -> 9.6 um fit frame


def villa_spiral_dir() -> str:
    path = os.environ.get('VILLA_SPIRAL_FITTING')
    if not path:
        raise RuntimeError('set VILLA_SPIRAL_FITTING to villa/spiral-fitting')
    return path


def import_villa():
    path = villa_spiral_dir()
    if path not in sys.path:
        sys.path.insert(0, path)
    import point_collection as pc  # noqa: F401
    import spiral_helpers  # noqa: F401
    import tifxyz  # noqa: F401
    import umbilicus  # noqa: F401
    return pc, spiral_helpers, tifxyz, umbilicus


def villa_revision() -> str:
    path = villa_spiral_dir()
    head = subprocess.run(['git', '-C', path, 'rev-parse', 'HEAD'],
                          capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', path, 'status', '--porcelain', '--', '.'],
                           capture_output=True, text=True, check=True).stdout
    tracked_dirty = [l for l in dirty.splitlines() if not l.startswith('??')]
    return head + ('+dirty' if tracked_dirty else '')


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

class FrameError(ValueError):
    pass


# Published CT for the scan (checked 2026-09-23 from the zarr's .zarray):
# level 0 [75784, 32693, 32693] (2.4 um), level 2 [18946, 8174, 8174] = the
# 9.6 um fit frame. Newer VC3D fibers declare the L0 domain with an inclusive
# maximum (32694); villa's loader accepts that convention and derives 0.25.
DECLARED_L0_SHAPES = ([75784, 32694, 32694], [75784, 32693, 32693])
FIT_BASE_SHAPE_ZYX = [18946, 8174, 8174]


def check_fiber_frame(path: str) -> dict:
    """Refuse fibers whose frame is not explicitly declared as the expected one.
    A missing declaration is never treated as a known scale.

    Accepted declarations: (a) VC3D open-data fields naming
    PHercParis4/20260411134726@L0 at 2.4 um; (b) coordinate_base_shape_zyx
    equal to that scan's published L0 shape. Returns the parsed JSON with
    '_frame_rule' set to 'open_data_fields' or 'declared_l0_shape'."""
    with open(path, 'rb') as handle:
        data = json.loads(handle.read())
    declared = data.get('coordinate_base_shape_zyx')
    if declared is not None:
        if list(declared) not in DECLARED_L0_SHAPES:
            raise FrameError(f'{os.path.basename(path)}: coordinate_base_shape_zyx={declared} '
                             'is not the published 20260411134726 L0 shape')
        space = data.get('vc_open_data_coordinate_space')
        if space is not None and space != EXPECTED_FIBER_FRAME['vc_open_data_coordinate_space']:
            raise FrameError(f'{os.path.basename(path)}: coordinate space {space!r}')
        data['_frame_rule'] = 'declared_l0_shape'
        return data
    for key, expected in EXPECTED_FIBER_FRAME.items():
        if key not in data:
            raise FrameError(f'{os.path.basename(path)}: missing {key}')
        value = data[key]
        if isinstance(expected, float):
            ok = isinstance(value, (int, float)) and abs(float(value) - expected) < 1e-9
        else:
            ok = value == expected
        if not ok:
            raise FrameError(f'{os.path.basename(path)}: {key}={value!r}, expected {expected!r}')
    data['_frame_rule'] = 'open_data_fields'
    return data


def villa_load_fiber(path: str, collection_id: int, min_point_spacing: float):
    """villa's loader with the frame argument matching the declaration; the
    loader's own dyadic check must then produce exactly 0.25."""
    _, spiral_helpers, _, _ = import_villa()
    raw = check_fiber_frame(path)
    base = FIT_BASE_SHAPE_ZYX if raw['_frame_rule'] == 'declared_l0_shape' else None
    collection = spiral_helpers.load_fiber_point_collection(
        path, collection_id, coordinate_scale=FIBER_TO_FIT_SCALE,
        min_point_spacing=min_point_spacing, base_shape_zyx=base)
    if collection is not None and \
            collection['metadata'].get('input_coordinate_scale') != FIBER_TO_FIT_SCALE:
        raise FrameError(f'{path}: loader used scale '
                         f'{collection["metadata"].get("input_coordinate_scale")}')
    return raw, collection


def load_fibers(paths: List[str], min_point_spacing: float = 40.0, first_id: int = 1_000_000):
    """villa's own fiber loader (control-span trim + decimation) after an
    explicit frame check. Returns (collections, provenance, rejected)."""
    collections, provenance, rejected = {}, {}, []
    next_id = first_id
    for path in sorted(paths):
        try:
            raw, collection = villa_load_fiber(path, next_id, min_point_spacing)
        except FrameError as error:
            rejected.append((os.path.basename(path), str(error)))
            continue
        if collection is None or not collection['points']:
            rejected.append((os.path.basename(path), 'villa loader returned no points'))
            continue
        for point in collection['points'].values():
            point['zyx'] = np.array([point['p'][2], point['p'][1], point['p'][0]], dtype=np.float32)
        collection['source_file'] = os.path.basename(path)
        # fit_spiral stamps these right after loading (fit_spiral.py, fiber
        # catalog setup); villa's H/V tagger returns None without them, which
        # would silently disable the side rules.
        collection['sampling_group'] = 'fibers'
        collection.setdefault('metadata', {}).update({
            'logical_input_kind': 'fiber',
            'logical_input_id': os.path.splitext(os.path.basename(path))[0],
            'logical_input_revision': sha256_file(path),
        })
        collections[next_id] = collection
        provenance[next_id] = {
            'file': os.path.basename(path),
            'sha256': sha256_file(path),
            'hv': raw.get('hv_classification', {}),
            'frame_rule': raw['_frame_rule'],
            'branches': [{k: b.get(k) for k in ('branch_file', 'control_point_index',
                                                'branch_control_point_index')}
                         for b in (raw.get('branches') or [])],
            'kept_orig_indices': [int(i) for i in collection.get('kept_orig_indices', [])],
        }
        next_id += 1
    return collections, provenance, rejected


def select_patch_names(meta_index_npz: str, points_zyx: np.ndarray, pad: float) -> List[str]:
    """Every patch whose meta bbox (xyz), padded by `pad`, contains a point.
    Conservative prefilter only: the native index decides actual hits."""
    index = np.load(meta_index_npz)
    lo = index['bbox_lo_xyz'] - pad
    hi = index['bbox_hi_xyz'] + pad
    xyz = points_zyx[:, ::-1].astype(np.float64)
    keep = np.zeros(len(lo), dtype=bool)
    for start in range(0, len(xyz), 256):
        chunk = xyz[start:start + 256]
        inside = ((chunk[:, None, :] >= lo[None]) & (chunk[:, None, :] <= hi[None])).all(-1)
        keep |= inside.any(0)
    return sorted(index['names'][keep].tolist())


def load_patches(patch_root: str, names: List[str], erode_cells: int = 1,
                 z_range=(-1e9, 1e9)) -> Dict[str, object]:
    """villa's patch payload loader (same erosion and z filtering)."""
    _, spiral_helpers, tifxyz, _ = import_villa()
    patches, dropped = {}, {}
    results = spiral_helpers.load_patch_payload_chunk(
        patch_root, names, z_range[0], z_range[1], erode_cells, io_threads=8)
    for entry, payload, error, reason in results:
        if payload is not None:
            patches[entry] = tifxyz.patch_from_payload(payload)
        else:
            dropped[entry] = error or reason
    return patches, dropped


def umbilicus_inward(umbilicus_json: str):
    pc, _, _, umb = import_villa()
    return pc.umbilicus_inward_direction(umb.json_umbilicus_z_to_yx(umbilicus_json, 1.0))


# --------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------

@dataclass
class NativeConfig:
    tolerance: float = 2.5            # pcl_link_distance_tolerance (fit voxels)
    window_points: int = 1            # pcl_link_window_points
    window_min_points: int = 1        # pcl_link_window_min_points
    side_filter: bool = False         # pcl_fiber_link_side_filter
    side_margin: float = 0.5          # pcl_fiber_link_side_margin_voxels
    hit_policy: str = 'largest_area'  # fit_spiral general policy
    min_point_spacing: float = 40.0   # pcl_fiber_min_point_spacing
    erode_cells: int = 1              # patch_erode_patches

    def as_dict(self):
        return dict(self.__dict__)

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()[:16]


def native_options(config: NativeConfig, collections, inward):
    """PatchLinkOptions exactly as fit_spiral._patch_link_options builds them
    (side rules from villa's own H/V classifier)."""
    pc, spiral_helpers, _, _ = import_villa()
    rules = {}
    if config.side_filter:
        for cid, pcl in collections.items():
            tag = spiral_helpers.fiber_collection_hv_tag(pcl, min_z_fraction=0.8, min_auto_certainty=0.5)
            if tag == 'V':
                rules[cid] = pc.SIDE_FRONT
            elif tag == 'H':
                rules[cid] = pc.SIDE_BEHIND
    return pc.PatchLinkOptions(
        window_points=config.window_points, window_min_points=config.window_min_points,
        side_rules=rules, inward_direction=inward if rules else None,
        side_margin=config.side_margin)


def build_index(patches, padding: float):
    pc, _, _, _ = import_villa()
    built = pc._build_surface_patch_index(patches, padding)
    if built is None:
        raise RuntimeError('vc_spiral.surface_index native backend is unavailable')
    return built


@dataclass
class Capture:
    """Flat hit table: one row per (fiber point, patch surface) within the envelope."""
    collection_id: np.ndarray
    point_pos: np.ndarray        # position in id-ordered point list
    point_id: np.ndarray
    patch: np.ndarray            # patch name
    distance: np.ndarray         # fit voxels
    ij: np.ndarray               # (N, 2)
    foot_zyx: np.ndarray         # (N, 3), NaN when not evaluable
    side_offset: np.ndarray      # foot offset along umbilicus inward dir, NaN if unknown
    within_tol: np.ndarray       # distance <= native tolerance (from the native-tolerance query)
    native_winner: np.ndarray    # the unmodified linker chose this row
    points: Dict[int, np.ndarray] = field(default_factory=dict)  # cid -> (P,3) zyx in id order
    point_ids: Dict[int, np.ndarray] = field(default_factory=dict)


def capture(collections, patches, config: NativeConfig, envelope: float, inward) -> Capture:
    """Envelope + native-tolerance queries for every point, the native
    selection replayed on the tolerance hits, all hits recorded."""
    pc, _, _, _ = import_villa()
    if envelope < config.tolerance:
        raise ValueError('envelope must be >= native tolerance')
    native_index, surface_ids = build_index(patches, config.tolerance)
    wide_index, wide_ids = build_index(patches, envelope)
    assert list(surface_ids) == list(wide_ids)
    options = native_options(config, collections, inward)
    areas = None
    if config.hit_policy == 'largest_area':
        areas = np.asarray([float(patches[p].area) for p in surface_ids], dtype=np.float64)

    def patch_for_surface(surface):
        return patches.get(surface_ids[surface])

    rows = {k: [] for k in ('cid', 'pos', 'pid', 'surf', 'dist', 'ij', 'tol', 'win')}
    points_out, ids_out = {}, {}
    for cid in sorted(collections):
        collection = collections[cid]
        items = pc._ordered_point_items(collection)
        zyxs = pc._point_zyxs(items)
        points_out[cid] = zyxs.copy()
        ids_out[cid] = np.asarray([int(k) for k, _ in items], dtype=np.int64)
        xyz = np.ascontiguousarray(zyxs[:, ::-1])

        # native-tolerance query, exactly the linker's call
        off_n, s_n, d_n, ij_n = native_index.locate_all_xyz_batch(xyz, config.tolerance)
        p_n = pc._hits_from_offsets(off_n)
        # replay the native selection on a private copy of the collection
        private = {'name': collection.get('name', ''), 'metadata': collection.get('metadata', {}),
                   'points': {k: {kk: vv for kk, vv in v.items() if kk != 'on_patch'}
                              for k, v in collection['points'].items()}}
        private_items = pc._ordered_point_items(private)
        pc._link_from_hits(private, cid, private_items, zyxs, p_n, s_n, d_n, ij_n,
                           distance_scale=1.0, surface_ids=surface_ids,
                           patch_for_surface=patch_for_surface, patch_areas=areas,
                           options=options)
        winners = {}
        for pos, (_, point) in enumerate(private_items):
            if 'on_patch' in point:
                winners[pos] = point['on_patch']['id']
        native_pairs = {(int(p), int(s)) for p, s in zip(p_n, s_n)}

        # envelope query: superset of candidates
        off_w, s_w, d_w, ij_w = wide_index.locate_all_xyz_batch(xyz, envelope)
        p_w = pc._hits_from_offsets(off_w)
        wide_pairs = {(int(p), int(s)) for p, s in zip(p_w, s_w)}
        missing = native_pairs - wide_pairs
        if missing:
            raise AssertionError(f'collection {cid}: envelope query lost {len(missing)} '
                                 'native-tolerance hits (candidate search incomplete)')
        for k in range(len(p_w)):
            pos, surf = int(p_w[k]), int(s_w[k])
            rows['cid'].append(cid)
            rows['pos'].append(pos)
            rows['pid'].append(int(ids_out[cid][pos]))
            rows['surf'].append(surf)
            rows['dist'].append(float(d_w[k]))
            rows['ij'].append(ij_w[k])
            rows['tol'].append((pos, surf) in native_pairs)
            rows['win'].append(winners.get(pos) == surface_ids[surf])

    surf = np.asarray(rows['surf'], dtype=np.int64)
    pos = np.asarray(rows['pos'], dtype=np.int64)
    cids = np.asarray(rows['cid'], dtype=np.int64)
    ij = np.asarray(rows['ij'], dtype=np.float32).reshape(-1, 2)
    # projection feet and umbilicus side offsets for every envelope hit
    feet = np.full((len(surf), 3), np.nan, dtype=np.float32)
    side = np.full(len(surf), np.nan, dtype=np.float64)
    for cid in np.unique(cids):
        mask = np.flatnonzero(cids == cid)
        feet[mask] = pc._projection_feet(pos[mask], surf[mask], ij[mask], patch_for_surface)
    known = np.isfinite(feet).all(1)
    if known.any():
        pts = np.stack([points_out[int(c)][int(p)] for c, p in zip(cids[known], pos[known])]).astype(np.float64)
        inward_dir = np.asarray(inward(pts), dtype=np.float64).reshape(-1, 3)
        defined = np.linalg.norm(inward_dir, axis=1) > 0
        offs = ((feet[known].astype(np.float64) - pts) * inward_dir).sum(1)
        offs[~defined] = np.nan
        side[known] = offs
    return Capture(
        collection_id=cids, point_pos=pos, point_id=np.asarray(rows['pid'], dtype=np.int64),
        patch=np.asarray([surface_ids[s] for s in surf]),
        distance=np.asarray(rows['dist'], dtype=np.float32), ij=ij, foot_zyx=feet,
        side_offset=side, within_tol=np.asarray(rows['tol'], dtype=bool),
        native_winner=np.asarray(rows['win'], dtype=bool), points=points_out, point_ids=ids_out)


def run_unmodified(collections, patches, config: NativeConfig, inward) -> Dict[tuple, tuple]:
    """villa's link_points_to_patches on a deep copy: {(cid, point_id): (patch, distance, ij)}."""
    pc, _, _, _ = import_villa()
    private = copy.deepcopy(collections)
    options = native_options(config, private, inward)
    pc.link_points_to_patches(patches, private, tolerance=config.tolerance,
                              surface_index_tolerance=config.tolerance, distance_scale=1.0,
                              general_hit_policy=config.hit_policy, options=options)
    out = {}
    for cid, collection in private.items():
        for pid, point in collection['points'].items():
            if 'on_patch' in point:
                link = point['on_patch']
                out[(int(cid), int(pid))] = (link['id'], float(link['distance']), tuple(link['ij']))
    return out


def replay_matches_native(cap: Capture, unmodified: Dict[tuple, tuple]) -> dict:
    replay = {}
    for k in np.flatnonzero(cap.native_winner):
        key = (int(cap.collection_id[k]), int(cap.point_id[k]))
        if key in replay:
            raise AssertionError(f'two winners for {key}')
        replay[key] = (str(cap.patch[k]), float(cap.distance[k]))
    mismatches = []
    for key in sorted(set(replay) | set(unmodified)):
        a, b = replay.get(key), unmodified.get(key)
        if a is None or b is None or a[0] != b[0] or abs(a[1] - b[1]) > 1e-4:
            mismatches.append((key, a, b and b[:2]))
    return {'replay_links': len(replay), 'native_links': len(unmodified),
            'mismatches': len(mismatches), 'examples': mismatches[:10]}
