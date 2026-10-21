# Equal-day aligned CRPS research kernel and protocol v7

This research snapshot preserves a request-specific mixture over the same 167
flattened joint-asset recipes. Its objective is daily cumulative-log-return CRPS
divided by training-prefix portfolio SD (sample SD, ddof=1) and square root of
native lead, with equal daily weights. Historical validation uses each fold's
training SD. The external score is the mean of scaled daily losses. Internally,
config18 retains the reference lead SUM exactly; the common request-specific
1/H normalization does not change its batch minimizer. The prior scorer and
aggregation hierarchy remain available unchanged.

The first specialization computes only the chosen coefficient stream and batch
fit instead of twelve streams and thirty-six learners. The compiled variant
compiles the full outer B-pair loop, retaining the existing sorted absolute-distance
kernel, arithmetic order, float64, every pair, and no fastmath. Neither variant
reduces recipes, fits, historical folds, lead work, simulation counts or final paths.
Request caches are transient. No production web or engine code is changed.

Qualification compared A/B coefficients, weights, invalid options, trace, every
final path value and complete daily loss vectors against the all-objective aligned
research reference: 36 original-context/seed pairs for V1, and 48 for V2, including
all twelve actual one-week requests at seed0. These overlapping comparisons are
implementation parity evidence, not independent predictive observations. The
original long-context test uses three internal seeds. All final forecasts have
1,000 paths and two inner paths. Private raw losses remain outside this snapshot.

Serial cProfile measurements include complete request selection/refit and all
1,000 output paths. Warm V2 wall times were 0.950s (week), 0.560s (historical slow
case) and 1.210s (portfolio), versus 4.777s, 5.045s and 11.250s for the
all-objective aligned research reference: approximately 5.0x, 9.0x and 9.3x.
Cold V2 times were 1.173s, 0.883s and 1.463s including the new JIT compilation.
These are local instrumented kernel measurements, not deployed-browser latency,
not comparisons with the old endpoint leader, and not a claim of production readiness.

At preparation, baseline rescoring under v7 is complete and independently
reduced on the same 64,049 requests per baseline: replacement Q=0.7691497993249988
and history-partition Q=0.7529818660401508. Replacement is the fixed reference
selected from full-panel coverage; partial reports keep that reference even if
subset rankings flip. The aligned full-panel comparison has not launched. The best broader development diagnostic reported
by the research owner is 32.98% versus the fixed replacement baseline over 355
contexts and three seeds: that candidate learned the internal endpoint objective
and was evaluated externally under v7. It is not the result of the aligned
internal-objective full-panel comparison, which remains pending. No model
promotion, catalogue increment or benchmark winner is established here.

`aligned_v7_public_inventory.json` (local reviewed inventory) pins every published
file and original restore location. The helper defaults to preparation only.
After operator inventory review, its explicit `--publish FINGERPRINT` stages only
listed files and PUBLIC_FILES.json, preserves prior public files, commits/pushes,
then downloads each scoped file from the immutable remote commit with HTTP200,
verifies SHA256 and restored modes, and runs only the 4 scorer, 7 focused V1/V2,
11 runner contract tests, and 2 baseline-rescore contract tests. `--recover COMMIT` repeats recovery into a fresh tree.
The library source and dependency lockfile are versioned under `_execution_closure`;
recovery reconstructs their original relative layout. Full inputs and array
payloads require the separately retained private archives; source recovery alone
does not claim to reproduce market-data forecasts.
