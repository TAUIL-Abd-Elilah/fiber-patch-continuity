# Fiber-to-patch association: baseline and first real failure (PHercParis4)

23 September 2026. Development data only. No human labels yet; every case
below is **unresolved pending human review**, not a verified error.

## Exact baseline

- villa `main` at `d285029ab6b62bfacf12cdb41bd4653867db73c0` (22 Sep). `spiral-fitting/`
  is byte-identical to the 21 Sep pin `a91475ba` (only CRLF differs; checked
  with `git diff a91475ba -- spiral-fitting` = empty). Native modules
  (`vc_spiral/*.pyd`) built from `f07d33be`, which has the same spiral-fitting
  source.
- Matcher: `point_collection.link_points_to_patches`, called the way
  `fit_spiral._derive_point_inputs` / `_relink_fibers_to_patches` call it:
  `surface_index_tolerance = pcl_link_distance_tolerance`, `general_hit_policy='largest_area'`.
- Config (villa defaults, `config.py`): `pcl_link_distance_tolerance 2.5` fit voxels (24 um),
  `pcl_link_window_points 1`, `pcl_link_window_min_points 1`, `pcl_fiber_link_side_filter False`,
  `pcl_fiber_min_point_spacing 40` (fit voxels, 384 um), `patch_erode_patches 1`. No shipped
  config under `spiral-fitting/configs/` changes these.
- Intended behaviour (README "Point-to-patch linking"): every fiber point attaches to the
  largest-area patch whose surface is within 2.5 voxels; nearest distance breaks ties. The
  optional window gate and side rule exist but are off by default.

## Inputs

- Fibers: 36 development fibers in fit-frame z [10500, 11500) (the 19 Sep pilot's 27+9
  files; already used in development). Loaded with villa's own
  `load_fiber_point_collection` after an explicit frame check (PHercParis4/20260411134726@L0,
  2.4 um, scale 0.25 to the 9.6 um fit frame). 1,843 decimated points in the band.
- Patches: `C:/VesuviusProgressData/PHercParis4/verified_patches` (89,237 dirs: 84,316
  band-seed + 4,921 legacy; listing sha256 `88483e87...`). Loaded with villa's
  `load_patch_payload_chunk` (same erosion). 6,438 patches whose padded meta bbox meets
  the band's fiber points.
- CT: public `PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr`, level 1
  (4.8 um) for figures; bounded chunk reads, disk cache.

## Replay (no behaviour change)

`native_adapter.py` imports villa's loaders, its surface index and `_link_from_hits`, and
records every (point, patch) hit within a 16-voxel envelope plus the projection foot,
patch normal and umbilicus-side offset. The unmodified `link_points_to_patches` is run
on a deep copy of the same fibers. **Result: 324 of 324 links identical (patch, distance),
0 mismatches**; every native-tolerance hit also appears in the envelope query. Runtime:
26 s load, 2.3 s capture + native link, 2.9 s dense evidence (this machine, CPU).

What the replay shows about the baseline on this band:
- 324 of 1,843 points link (17.6%); 192 of those 324 have 2+ patches within tolerance;
  117 winners are not the nearest candidate (median gap 0.29 voxels).
- Nearest-patch distances form a continuum from 0 to 12+ voxels (vertical fibers almost
  flat); there is no clean gap a tighter tolerance could exploit.

## The gap: the matcher never follows a patch along the fiber

Each point is judged alone, at 2.5 voxels. A patch that touches the fiber within
tolerance at some points but, elsewhere along the same fiber, **covers the fiber's path
while lying a full sheet away** is still linked. `evidence.py` re-samples each fiber
every 8 fit voxels along the same trimmed polyline and, for each patch, splits the
offset into a normal part and an in-plane part (in-plane <= 1 voxel = the patch really
covers that spot). Lengths are physical arc length, not sample counts.

On the band, 43 of the 82 native (fiber, patch) pairs (154 of 324 links) have >= 200 um
of fiber where the linked patch covers the path but sits >= 6 voxels (58 um) away.
These thresholds are uncalibrated development values.

## Cases (CT cards in `views/`)

| Case | Fiber | Linked patch | Native links | What the card shows |
|---|---|---|---|---|
| A (unresolved) | `anon_20260815T133724240_000070` (H 0.78) | `auto_grown_20260521190621512_sel_20260521_190859_17` (legacy) | 25 | Patch follows the fiber for ~10 mm (12-22 mm), but for the preceding 7 mm the same patch, continuously, lies 90-150 um (about one sheet) outward; a second legacy patch coincides with it there. Ramp onto the fiber over ~0.5 mm at 11.5-12 mm. Either the patch or the fiber changes sheet. |
| B (unresolved, likely wrong) | `anon_20260815T131202278_000059` (H 0.98) | `band-seed755037-20260730-001305-027` | 1 | Patch wanders 20-110 um inward of the fiber and enters tolerance at one point; two coinciding patches sit ~120 um outward. |
| C (unresolved) | `anon_20260815T021612104_000016` (H 0.90) | `auto_grown_20260521153551996_sel_20260521_154722_4` | 4 | Like A: ~100 um inward for 7 mm, then converges where the links are. |
| D (control) | `anon_20260815T030050140_000047` (H 0.69) | `auto_grown_20260526195309752_sel_20260526_210343_58` | 11 | Patch and an independent band-seed patch coincide with the fiber for ~17 mm; fiber on the same CT sheet; a third patch one sheet below never enters tolerance. |

An LLM reading these images is not ground truth; each needs a knowledgeable human's
raw-CT verdict.

## What the proposed method changes

An optional, default-off decision layer between candidate capture and constraint
export. It does not change the projector, the index or the native link when off.
Per (fiber span, patch) it reports support length, in-domain contradiction length,
competing covering surfaces and native-link count, and returns `propose` only for
long, uncontradicted support; contradicted, single-touch or ambiguous pairs go to
`review` with the card above. The window gate (`pcl_link_window_min_points`) already
drops some single touches (B) but cannot see A/C, whose linked points are consecutive.

## Overlap check (23 Sep)

- villa main has no spiral-fitting change since the pin. Open PRs: #1864/#1855 (winding
  audit false contradictions, sergeievland windaudit) audit annotation-graph cycles, not
  fiber-to-patch placement; #1829 (VC3D adjacent fiber link candidates) links fibers to
  fibers. No open PR/issue found for continuity checks on fiber-to-patch links (GitHub
  search; not proof no private work exists).
- Discord, #unrolling-vc3d, 21 Sep 13:48: Sean replied to our question: he has no
  bundle yet, will upload reviewed placements to ash2txt `datasets/` after his review and
  post in the channel. Checked 23 Sep: not posted; `datasets/spiral_datasets/PHercParis4/`
  has no new review folder. Not treated as endorsement or ownership.
- 22 Sep: Qual is producing spline fibers from a model; Sean wants them as VC3D fiber
  JSONs so `fit_spiral.py` uses them unchanged. More auto fibers make placement review
  the bottleneck, not a competing solution.

## Remaining unknowns / dependencies

1. Human labels. Exact need: a knowledgeable reviewer's correct / wrong / unresolved
   verdict on ~100-200 (fiber, patch) pairs drawn from untouched regions, blinded to
   the method's decision, plus timing. Candidate sources: Sean's promised reviewed bundle;
   another reviewer the user can recruit; the user. None confirmed.
2. Whether Sean's association step is `fit_spiral`'s linker or a VC3D workflow.
3. Sheet spacing and the H/V layer offset vary; 2.5 / 6 voxel thresholds are
   development values to be calibrated only on calibration data.
