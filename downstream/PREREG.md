# Downstream fit comparison — written 2026-09-23 before any full-length fit

Dataset: manifest.json (sha256 e3c4fa358e261277b4cb7aebd41ed72183ebe98f62e9de35ff1ca5a781e4aae5):
PHercParis4 fit-frame z [12500, 13500), 14,780 verified patches loaded by the fitter, 18 fit
fibers, 9 held-out fibers (whole files, sha256-ordered split; never in any arm).

Arms (identical config, seed 20260923, 3,000 steps, villa d285029ab + local filter commit dce8f8353):
- P: verified patches only.
- A: patches + fit fibers, villa's native linking (defaults).
- D: as A, but the 299 fit-fiber pairs the frozen continuity policy routed to review are passed
  as rejected patches.

Primary: point-weighted native strip satisfaction of held-out fibers (villa
satisfaction_metrics), D vs A. Secondary: per-fiber mean, fully satisfied fibers, median
projection error, fit-fiber attached points per arm, wall time, peak GPU memory.

Reading: D > A by more than the seed-to-seed spread (if extra seeds run) supports "the flagged
links hurt the fit". D <= A: no downstream benefit shown; report it. One seed first (single-run
limitation stated); seeds 2-3 for A and D only if GPU time allows. GPU shared with the user's
PRPLL job; timings are not clean benchmarks. This is a bounded development fit, not the
production recipe, and satisfaction is a metric, not a verified surface.
