# Arm F — written 2026-09-24, after arms A/D/P (3 seeds) and BEFORE any F fit

Why a new arm: D (reject flagged fiber-patch pairs) lowered held-out satisfaction in all 3 seeds.
In the flagged cases the fiber links usually sit on the stretch where the patch is right; the
problem is the patch itself, which lies a sheet off elsewhere along the fiber. The fitter treats
each patch as one winding, so such a patch misleads the fit with or without fiber links.

F: same dataset, config, 3,000 steps and seeds (20260923, 20260925, 20260926) as A, with fibers and
native linking, but the 34 verified patches that a FIT fiber contradicts (connected contradiction
>= 150 um in runs/cal_region/continuity.json; held-out fibers not used) removed via villa's
patch_uuid_filter_regex. List: F_removed_patches.json (sha256 7ea0de88cfbddd6fbb1eb958f31cb77e061293d0469cb1cff446031f4b17555d).

Primary: held-out point-weighted satisfaction, F vs A paired by seed.
Reading, fixed now: F better than A in >= 2 of 3 seeds AND mean gain > 0.027 (the per-arm seed sd)
-> "promising on development"; confirm on the untouched test band with a fresh fit/held-out
split before any claim. Otherwise -> no benefit; stop the fit line and report it.
These 9 held-out fibers have now been used to evaluate A/D/P, so this is development evidence,
not a final test.
