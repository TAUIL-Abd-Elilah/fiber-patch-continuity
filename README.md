# fiber-patch-continuity

A review aid for PHercParis4 fiber-to-patch placements. It replays villa's
spiral-fitting linker exactly. It then follows each linked patch along the
fiber, and routes a placement to review when the patch covers the fiber's path
but sits about a sheet away from it. Clean placements come out as proposals.
Every flag comes with a raw-CT card.

**Status (23 Sep 2026): experimental. No human labels yet.** The numbers below
are geometric measurements on real data. They show where the current linker and
the patch surfaces disagree. They do not show which side is wrong, and they do
not show that a proposal is correct. That needs a knowledgeable reviewer's
verdict on raw CT; the blinded labeling kit for it is in `label_kit/`.

## The problem

Sean described ~85,000 Paris4 patches and fiber placements whose review is the
bottleneck ([Discord, #unrolling-vc3d, 21 Sep](https://discord.com/channels/1079907749569237093/1243576621722767412/1551574883950137407)).
villa's linker (`spiral-fitting/point_collection.py`, called from `fit_spiral.py`) attaches
each decimated fiber point (every 40 fit voxels = 384 um) to the largest-area
patch whose surface is within 2.5 voxels (24 um). It judges each point
on its own, so it never checks what the same patch does elsewhere along the
fiber.

## What was measured

Data: PHercParis4 scan 20260411134726, fibers from the public `eval_fibers`
(frame checked against the published zarr; undeclared frames rejected), the
89,237 public `verified_patches`, villa `d285029ab` (22 Sep main; `spiral-fitting/`
identical to `a91475ba`). Regions are fit-frame z bands (9.6 um voxels) with
500-voxel buffers, fixed before any policy ran on them (`results/splits_20260923.json`).

**Replay:** the instrumented capture reproduces the unmodified linker link for link.

| region (z) | fibers | patches loaded | fiber points | native links | mismatches |
|---|---:|---:|---:|---:|---:|
| dev 10500-11500 (used before) | 36 | 6,438 | 1,843 | 324 | 0 |
| cal 12500-14500 (new fibers) | 31 | 13,344 | 4,723 | 1,179 | 0 |
| test 15000-17500 (new fibers) | 14 | 11,473 | 4,711 | 2,338 | 0 |

The tests also check equality under villa's own options: window 5/3, side
filter, tolerance 1.5.

**Continuity.** Each fiber is re-sampled every 8 voxels along the same trimmed
polyline villa fits. For every patch within 16 voxels, the offset is split into
a normal part and an in-plane part; an in-plane part under 1 voxel means the
patch really covers that spot. A placement is *contradicted* when, over
>= 150 um of fiber, the linked patch covers the path but sits >= 6 voxels
(58 um) off it, continuously connected to the stretch where it does support
the fiber.

| region | native (fiber, patch) pairs | pairs with connected contradiction | native links on those pairs |
|---|---:|---:|---:|
| dev | 82 | 49 | 176 of 324 |
| cal | 280 | 75 | 340 of 1,179 |
| test | 347 | 179 | 1,722 of 2,338 |

**Case A (dev).** Fiber `anon_20260815T133724240_000070`, legacy patch
`auto_grown_20260521190621512_sel_20260521_190859_17`, 25 native links. The patch
follows the fiber for ~10 mm. For the 7 mm before that, it lies 90-150 um (about
one sheet) outward, and a second legacy patch coincides with it there. The native
linker's own options (window gate, side filter, tighter tolerance, all combined)
all still link it. Card: `results/cases/dev_A_diverging_legacy_patch.png`;
control: `results/cases/dev_D_control_long_support.png`.

**Arms on the same candidates** (unique supported fiber length, duplicate
patches counted once; mm):

| arm | dev | cal | test |
|---|---:|---:|---:|
| A native defaults | 117.9 | 433.7 | 877.2 |
| B window 5/3 | 95.8 | 392.2 | 844.2 |
| B window 5/3 + side + tol 2 | 64.8 | 269.7 | 732.6 |
| C point-local margin | 80.1 | 308.1 | 685.6 |
| D continuity (proposals) | 33.0 | 202.1 | 93.9 |

D keeps 28% / 47% / 11% of A's supported length. **In the test region that is
below the 25% retention target set before the run.** D's thresholds were frozen
before the cal/test runs and have not been retuned on test. Whether D's
proposals are more often correct than A's links is not known until labels exist.

## What this does not show

- Accuracy of any arm: no human labels yet.
- Review time saved: no timed review yet.
- A better spiral fit: not run.
- The flags are geometric disagreements between a human-directed fiber trace and
  an automatic patch. Either can be the wrong one; a flag says a reviewer should
  look.

## Run it

Python 3.14 environment with villa's spiral-fitting dependencies and its native
`vc_spiral` modules; set `VILLA_SPIRAL_FITTING` to villa's `spiral-fitting`
directory at the revision above.

```
python build_patch_meta_index.py --out patch_meta_index
python fiber_inventory.py
python make_splits.py
python run_evidence.py --fibers <fiber jsons> --z-min 12500 --z-max 14500 --out runs/cal_region
python -c "import selection as s; r=s.build_records('runs/cal_region'); s.save(r,'runs/cal_region/pairs.jsonl')"
python evaluate.py --run runs/cal_region --out runs/cal_region/arm_summary.json
python continuity.py --run runs/cal_region --out runs/cal_region/continuity.json
python render_cases.py --run runs/dev_band_10500_dense --cases cases/dev_first3.json --out views/cards
python -m pytest tests
```

Runtimes on one Windows desktop (CPU): 1-2 min per region to load patches,
under 40 s for capture, native replay and dense evidence. CT is read from the
public zarr (level 1, 4.8 um) chunk by chunk with a disk cache.

## Files

- `native_adapter.py`: frame checks, villa loaders, envelope capture, replay equality.
- `evidence.py`, `pair_evidence.py`, `continuity.py`: dense per-pair evidence in physical length.
- `selection.py`: pair records (schema `fiber-patch-assoc/0.1`) and the arms. `evaluate.py`: arm summaries, and precision with bounds once labels exist.
- `review_export.py`, `render_cases.py`, `ct.py`: CT review cards.
- `blind_cards.py`, `build_label_page.py`, `label_page_template.html`: blinded labeling kit.
  `label_kit/` holds the 64 v1 cards (calibration and test regions, four strata, shuffled).
  Its `index.html` also works from a local copy, but there answers stay in your browser;
  ask the author for the hosted copy, which saves them. The card-to-pair key is withheld
  until labeling ends.
- `results/`: replay reports, arm and continuity summaries, split manifest, dated protocol.

Per-pair decisions for the cal and test regions are withheld until their labels
are collected, so the blinded cards cannot be matched to decisions.

## villa change

`spiral-fiber-link-review-filter` (local branch, not yet proposed upstream):
`PatchLinkOptions.allowed_patches` / `rejected_patches`. This lets reviewed
placements constrain a fit. It is off unless passed; the tests cover both link
backends.

## Provenance and licenses

Code MIT (see LICENSE). It imports villa (MIT, Vesuvius Challenge) and does not
copy it. The CT images and derived records come from Vesuvius Challenge open
data (PHercParis4); follow the Vesuvius Challenge data terms when reusing them.
Developed with AI assistance under the author's direction; every number above
comes from a command in this repository run on the data named here.
