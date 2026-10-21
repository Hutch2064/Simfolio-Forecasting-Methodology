# Internal score alignment experiment, research only

New isolated infrastructure; the sequential-learning module and all prior qualified
sources/plans remain frozen. No production edit, publication, outer panel run, OOS
win or runtime-saving claim is part of this implementation.

## Controlled axes

`configurations(inner_paths=2, schedule='half_doubling')` independently enumerates
36 unique configurations: three lead criteria, four scales and three learners.
The same existing 167 flattened recipes, requested-horizon adaptation, completed
dated fold schedule, represented-origin counts, prefix fits and inner seeds apply
to every configuration. `prepare` generates one joint-asset path-cube stream per
fold, computes exact raw empirical CRPS coefficients at each native lead with the
unchanged `energy.coefficients`, then releases the lead's raw A/B after adding it
to the12 score streams. It retains no per-lead path or pair-matrix tensor.

Each lead's CRPS is L_h(w)=w'A_h-.5*w'B_h*w. The coefficient scaling is division
by the selected denominator **before** criterion weighting:

| Scale | Denominator at native lead h |
|---|---|
| raw | 1 |
| train_SD | std(fold-prefix portfolio daily returns, ddof=1) |
| train_SD_sqrt_lead | same training SD times sqrt(h) |
| sqrt_lead | sqrt(h) |

Zero/nonfinite training SD fails the entire experiment, including raw modes,
because all12 streams are deliberately admitted as one controlled comparison.
No validation return enters the SD, and no outer-heldout argument is accepted.

| Criterion | Lead weights within one fold |
|---|---|
| endpoint | 1 at the final native lead, 0 elsewhere |
| uniform_days | 1 at every native lead; **sum**, without dividing by lead count |
| log_calendar | positive log-calendar interval weights normalized to sum1 |

Log-calendar weights reuse the frozen function in
`asset_selector_fold_scale/canonical_scoring.py`. With elapsed calendar days d_h
from the fold's last training date to the actual historical validation dates,
weight h is [log(1+d_h)-log(1+d_(h-1))]/log(1+d_H), d_0=0. Weekend and holiday
gaps are represented by actual dates; no extra target observations are invented.
All of these dates/observations are already inside the outer request's history.

Each stream supplies one fold loss multiplied by its represented-origin count.
Batch accumulates the represented-count-weighted losses chronologically and divides
by total represented origins. Uniform-days therefore pools native daily losses as
a sum for each represented origin; it does not average each fold first. BOA also
receives one count-weighted fold loss (variance uses squared count-weighted gradient),
not artificial independent repeated observations. Other normalization is exactly as
specified above. With a constant internal horizon within a request, dividing every
uniform-days fold by that common horizon is mathematically a global scale change,
but is intentionally not done here.

## Learners and preservation

For each criterion/scale, `replay(cache)` returns batch, BOA and progressive weights
without resimulating paths. Batch uses the unchanged parent certified solver;
BOA/progressive reuse the frozen sequential-learning implementation, including its
literature-based gradient sign, adaptive-range initialization, numerical guards and
explicit absence of transferred quantile-pooling or stochastic fast-rate guarantees.
Refer to that module's existing DESIGN.md; it was not changed for this experiment.
Weights are request-specific. No prior panel score, previous request's weight, or
outer evaluation outcome is accepted by the learner.

Endpoint/sqrt_lead/batch preserves exact parent arithmetic, including the endpoint
observation's `.sum()` reduction instead of cumsum's potentially different rounding.
The full real-bank fixture verifies byte-exact A/B, weights, allocations and final
paths against `asset_selector_endpoint_flat`, at chunk widths1,2,3. Changing
criteria/scales deliberately defines alternative internal losses, not different
outer evaluation scores.

`ScorePool` delegates final current-prefix component drawing and dated-request
binding to the frozen sequential path adapter, which preserves original joint
asset/portfolio accounting, antithetic even allocation, component ordering and
context-seed calls. Official outer experiments must request1000 paths uniformly.
The bounded unit fixture uses20 final paths and inner2; this is contract verification,
not an accuracy comparison.

## Cache ownership and numerical limits

The request cache stores float64 A and float64 symmetric upper-triangle B for all12
streams/folds, valid recipe IDs, counts, native horizons, prefix lengths and prefix
SDs. Dates, seeds, bank hash and exact history/portfolio bindings travel in JSON
provenance. Its coefficient storage is bounded by
8*12*F*(K+K*(K+1)/2) bytes for F represented folds and K valid recipes, independent
of horizon/path count. Temporary chunk memory is released after each chunk.
There is one caller-owned request cache, no hidden global or persistent cache.
`save_cache` writes lossless compressed NPZ; `load_cache` checks request identity,
recipe-bank identity, schema and axis ordering. The caller must release each
completed request's in-memory bundle/cache before advancing; retained files require
an explicit owning experiment/storage budget. Unit tests delete their NPZ fixtures.

Training-SD failure, nonfinite cubes, incomplete cursors, changing valid recipe
identities, nonfinite coefficients and numerical overflow/underflow in scaling
fail visibly. The unchanged BOA also rejects extreme float64 states rather than
silently clipping, eliminating experts, or selecting a fixed winner. This version
has no state-rescaling extension. No recipe/origin/path/iteration caps were added.

The shared final prefix fit covers the union of36 supports, commonly every recipe;
shared preparation timing cannot be attributed to any individual variant. A later
experiment must separate coefficient generation, learner replay, refit and path
costs if it wants to discuss runtime. No speed claim follows from this work.

## Recommended compact smoke

Start with the already selected compact real-data requests and **inner2 and inner8**,
all36 variants, identical outer1000 paths, frozen schedules/seeds/recipes. Reuse each
request's compact coefficient cache, verify exact parent endpoint control first,
and record errors at variant/request level. Treat tiny-inner results as numerical
and scientific admission only; they cannot establish production accuracy. Do not
select a variant using outer results and then report that same sample as fresh
confirmation. Any broader predeclared gate and storage growth remains root-owned.
