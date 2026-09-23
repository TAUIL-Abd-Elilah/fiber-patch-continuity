"""Bounded raw-CT access for PHercParis4 20260411134726 (public, OME-Zarr v2).

Only chunks intersecting a requested box are fetched; each is cached on disk
as raw bytes (the store is uncompressed uint8, 128^3 chunks). A running byte
counter enforces a download cap. Coordinates are zyx in the level's voxels;
level 2 is the 9.6 um fit frame.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

import numpy as np
from scipy import ndimage

BASE = ('https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/'
        'volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr')
CACHE = os.environ.get('PARIS4_CT_CACHE', 'C:/VesuviusProgressData/ct_cache/PHercParis4_20260411134726')
DOWNLOAD_CAP_BYTES = int(float(os.environ.get('PARIS4_CT_CAP_GB', '4')) * 1e9)
_downloaded = {'bytes': 0}


def level_meta(level: int) -> dict:
    path = os.path.join(CACHE, f'L{level}', '.zarray')
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with urllib.request.urlopen(f'{BASE}/{level}/.zarray', timeout=60) as r:
            raw = r.read()
        with open(path, 'wb') as handle:
            handle.write(raw)
    with open(path) as handle:
        meta = json.load(handle)
    assert meta['dtype'] == '|u1' and meta['compressor'] is None and meta['order'] == 'C'
    assert meta.get('dimension_separator') == '/'
    return meta


def _chunk(level: int, cz: int, cy: int, cx: int, meta: dict) -> np.ndarray:
    shape = tuple(meta['chunks'])
    path = os.path.join(CACHE, f'L{level}', f'{cz}_{cy}_{cx}.bin')
    if os.path.exists(path):
        raw = open(path, 'rb').read()
    else:
        if _downloaded['bytes'] >= DOWNLOAD_CAP_BYTES:
            raise RuntimeError('CT download cap reached')
        url = f'{BASE}/{level}/{cz}/{cy}/{cx}'
        raw = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=120) as r:
                    raw = r.read()
                break
            except urllib.error.HTTPError as error:
                if error.code in (403, 404):  # missing chunk == fill value
                    raw = b''
                    break
                if attempt == 4:
                    raise
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                if attempt == 4:
                    raise
            time.sleep(2.0 * (attempt + 1))
        _downloaded['bytes'] += len(raw)
        with open(path + '.part', 'wb') as handle:
            handle.write(raw)
        os.replace(path + '.part', path)
    if not raw:
        return np.full(shape, meta['fill_value'], dtype=np.uint8)
    arr = np.frombuffer(raw, dtype=np.uint8)
    if arr.size != int(np.prod(shape)):
        raise ValueError(f'chunk {level}/{cz}/{cy}/{cx} has {arr.size} bytes')
    return arr.reshape(shape)


def read_box(level: int, lo_zyx, hi_zyx) -> np.ndarray:
    """uint8 block for [lo, hi) (clipped to the volume; outside = 0)."""
    meta = level_meta(level)
    shape = np.asarray(meta['shape'])
    ch = np.asarray(meta['chunks'])
    lo = np.asarray(lo_zyx, dtype=np.int64)
    hi = np.asarray(hi_zyx, dtype=np.int64)
    out = np.zeros(tuple(hi - lo), dtype=np.uint8)
    clo = np.maximum(lo, 0)
    chi = np.minimum(hi, shape)
    if (chi <= clo).any():
        return out
    for cz in range(clo[0] // ch[0], (chi[0] - 1) // ch[0] + 1):
        for cy in range(clo[1] // ch[1], (chi[1] - 1) // ch[1] + 1):
            for cx in range(clo[2] // ch[2], (chi[2] - 1) // ch[2] + 1):
                block = _chunk(level, cz, cy, cx, meta)
                c0 = np.array([cz, cy, cx]) * ch
                a = np.maximum(clo, c0)
                b = np.minimum(chi, c0 + ch)
                out[tuple(slice(a[k] - lo[k], b[k] - lo[k]) for k in range(3))] = \
                    block[tuple(slice(a[k] - c0[k], b[k] - c0[k]) for k in range(3))]
    return out


def sample(level: int, coords_zyx: np.ndarray, order: int = 1) -> np.ndarray:
    """Interpolated CT at arbitrary level-voxel coordinates (..., 3).

    Points are grouped by the chunk they fall in and read per chunk-neighbourhood,
    so a long curved fiber does not pull its whole bounding box."""
    coords = np.asarray(coords_zyx, dtype=np.float64)
    flat = coords.reshape(-1, 3)
    meta = level_meta(level)
    ch = np.asarray(meta['chunks'])
    key = np.floor(flat / ch).astype(np.int64)
    out = np.zeros(len(flat), dtype=np.float32)
    uniq, inverse = np.unique(key, axis=0, return_inverse=True)
    for g in range(len(uniq)):
        rows = np.flatnonzero(inverse.reshape(-1) == g)
        pts = flat[rows]
        lo = np.floor(pts.min(0)).astype(np.int64) - 2
        hi = np.ceil(pts.max(0)).astype(np.int64) + 3
        block = read_box(level, lo, hi).astype(np.float32)
        out[rows] = ndimage.map_coordinates(block, (pts - lo).T, order=order, mode='nearest')
    return out.reshape(coords.shape[:-1])


def sheet_normals(level: int, centers_zyx: np.ndarray, half: int = 12, sigma_grad: float = 1.0,
                  sigma_tensor: float = 4.0) -> np.ndarray:
    """CT structure-tensor sheet normal (unit, zyx, sign arbitrary) at each centre."""
    out = np.full((len(centers_zyx), 3), np.nan)
    for k, c in enumerate(np.asarray(centers_zyx, dtype=np.float64)):
        lo = np.round(c).astype(np.int64) - half
        block = read_box(level, lo, lo + 2 * half + 1).astype(np.float32)
        if (block > 0).mean() < 0.5:
            continue
        g = [ndimage.gaussian_filter(block, sigma_grad, order=o) for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
        w = np.zeros(block.shape, np.float32)
        w[half, half, half] = 1.0
        w = ndimage.gaussian_filter(w, sigma_tensor)
        tensor = np.array([[(g[i] * g[j] * w).sum() for j in range(3)] for i in range(3)])
        vals, vecs = np.linalg.eigh(tensor)
        out[k] = vecs[:, -1]
    return out


def prefetch(level: int, coords_zyx: np.ndarray, margin: int = 16, workers: int = 8) -> int:
    """Fetch (in parallel) every chunk within `margin` voxels of the given
    level-voxel coordinates, so later reads hit the disk cache. Returns the
    number of chunks requested."""
    from concurrent.futures import ThreadPoolExecutor
    meta = level_meta(level)
    ch = np.asarray(meta['chunks'])
    shape = np.asarray(meta['shape'])
    flat = np.asarray(coords_zyx, dtype=np.float64).reshape(-1, 3)
    keys = set()
    stencil = [(0, 0, 0)] + [tuple(margin * s * (np.arange(3) == a)) for a in range(3) for s in (-1, 1)]
    for delta in stencil:
        pts = np.clip(flat + np.asarray(delta), 0, shape - 1)
        keys |= set(map(tuple, np.unique(np.floor(pts / ch).astype(np.int64), axis=0).tolist()))
    todo = [k for k in sorted(keys)
            if not os.path.exists(os.path.join(CACHE, f'L{level}', f'{k[0]}_{k[1]}_{k[2]}.bin'))]
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda k: _chunk(level, k[0], k[1], k[2], meta), todo))
    return len(todo)


def downloaded_bytes() -> int:
    return _downloaded['bytes']
