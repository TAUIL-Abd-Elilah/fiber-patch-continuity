"""Index every verified patch's meta.json (bbox, area, provenance) once.

Read-only over the patch cache. Output: patch_meta_index.npz + a sidecar
JSON with the per-source counts and the input directory listing hash.
Coordinates stay exactly as stored: meta bbox is [[x,y,z],[x,y,z]] in the
PHercParis4 9.6 um fit frame (tifxyz convention).
"""
import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def read_meta(root, entry):
    path = os.path.join(root, entry, 'meta.json')
    try:
        with open(path, 'rb') as handle:
            raw = handle.read()
        meta = json.loads(raw)
    except Exception as error:  # recorded, never silently dropped
        return entry, None, repr(error)
    bbox = meta.get('bbox')
    ok = (isinstance(bbox, list) and len(bbox) == 2
          and all(isinstance(c, list) and len(c) == 3 for c in bbox))
    return entry, {
        'bbox': bbox if ok else None,
        'area_vx2': float(meta.get('area_vx2', float('nan'))),
        'area_cm2': float(meta.get('area_cm2', float('nan'))),
        'scale': meta.get('scale'),
        'source': str(meta.get('source', '')),
        'uuid': str(meta.get('uuid', entry)),
        'erode_override': meta.get('spiral_patch_erode_cells'),
        'meta_sha256': hashlib.sha256(raw).hexdigest(),
    }, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patches', default='C:/VesuviusProgressData/PHercParis4/verified_patches')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    entries = sorted(e for e in os.listdir(args.patches)
                     if os.path.isdir(os.path.join(args.patches, e)))
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda e: read_meta(args.patches, e), entries))

    names, lo, hi, area, family, errors, sha = [], [], [], [], [], [], []
    sources = {}
    for entry, meta, error in results:
        if meta is None or meta['bbox'] is None:
            errors.append((entry, error or 'no bbox'))
            continue
        names.append(entry)
        lo.append(meta['bbox'][0])
        hi.append(meta['bbox'][1])
        area.append(meta['area_vx2'])
        fam = 'band_seed' if entry.startswith('band-seed') else 'legacy'
        family.append(fam)
        sha.append(meta['meta_sha256'])
        sources[meta['source']] = sources.get(meta['source'], 0) + 1

    listing_hash = hashlib.sha256('\n'.join(entries).encode()).hexdigest()
    np.savez_compressed(
        args.out + '.npz',
        names=np.asarray(names), bbox_lo_xyz=np.asarray(lo, np.float64),
        bbox_hi_xyz=np.asarray(hi, np.float64), area_vx2=np.asarray(area),
        family=np.asarray(family), meta_sha256=np.asarray(sha))
    with open(args.out + '.json', 'w') as handle:
        json.dump({
            'patch_root': args.patches, 'entries': len(entries),
            'indexed': len(names), 'errors': errors[:50], 'num_errors': len(errors),
            'listing_sha256': listing_hash, 'sources': sources,
            'frame': 'PHercParis4 9.6um fit frame, bbox xyz',
        }, handle, indent=1)
    print(f'indexed {len(names)}/{len(entries)} errors {len(errors)} listing {listing_hash[:16]}')


if __name__ == '__main__':
    main()
