# Mapped representative ratio geometry v1

Source/synthetic prototype: eight focused qualification tests passed; independent review and real-data admission remain separate. No historical numerical fits or real request forecasts. No scientific launch or production approval. The broader source-design rationale is O/validation_horizon_geometry_design_v1.md; ONLY its final mapped-representative variant is implemented, not its all-origin alternative.

## Exactly one scientific change

Finite41 catalogue remains `[1,2,3,4..15,22..33,112..123,166,167]`. N2, financial recipes, fitted state equations, original path RNG namespaces, whole-joint paths, daily target rebalancing, endpoint CRPS/sqrt(native h), unique-v3 solver and final forecast generation are unchanged. There is no reference reliability multiplier, new mean, law, outer-score change or source routing.

First obtain the FROZEN old eligible set, half-doubling representative folds and integer represented counts. Let N be outer known rows, H requested native horizon, and m the unchanged eligible catalogue's minimum prefix support. For N>m:

- If H*m<=N*(N-m): use target rational(H,N), maximum prefix U=N²//(N+H).
- Otherwise: use maximal feasible rational(N-m,m), U=m. Record requested_ratio_unattainable/support_projection_applied and the requested/used exact rational fields. This is an explicitly disclosed APPROXIMATION when the requested ratio cannot be evaluated, not a model-winner fallback or evidence at the requested ratio.
- Map each old(n,h,mass) to n'=min(n,U), h'=ceil(n'*r_num/r_den) using exact integer arithmetic. Merge ONLY identical(n',h') by adding mass; preserve chronology. Retain the old full represented-count denominator and total mass. The count of actual fits never increases.
- N<=m has no positive completed interval and fails unsupported. A collapsed single episode remains ONE episode even if it carries many represented origins. Metadata does not reinterpret represented mass as independent validation observations.

`requested_ratio_preserved=True` means the REQUESTED TARGET RATIONAL is retained. It does not mean rounded h'/n' equals it: realized ratios are in[r,r+1/n'). Prefix+horizon completion is guaranteed by integer feasibility; all labels end inside known N. No date, ticker or return outcome influences this map.

The pushforward preserves the old weighted REPRESENTATIVE measure, not an exact pushforward of every unexecuted original origin. A/B still approximate an historical selection risk; no claim that matching h/n yields unbiased actual(N,H) risk or optimally handles volatility/PPW/regime time scales. For MedEqN=H12106, old(24,6041,1),(6065,6041,1) becomes(24,24,1),(6053,6053,1). At H1 the old schedule is exactly retained, including mass. At very large H with infeasible requested ratio all old mass maps to(m,N-m), explicitly projected/one-episode.

## Request-local cache and interface

New GroupWork subclasses only the frozen request-local cache. It computes registered_folds for current/mapped geometry and replaces ONLY its OWN expected-use counters with exact new prefix counts. prefix_fits accepts only registered stops/currentN and finite41 IDs. Inherited bit-exact prefix and portfolio checks remain in force. No imported schedule, BANK, MENUS or core is patched. Current and mapped modes can reuse immutable fits at identical prefixes; final prefix is always the FULL currentN, never the mapped prefix.

prepare(...mode='mapped_ratio',seed_replication=0,fit_cache=None,solver=None,width=64) returns one explicit source-bound bundle; a single-mode context registers only that mode. prepare_all defaults to current/mapped modes with one shared context; an error returns an explicit per-mode unsupported receipt, with no forecast weights fabricated. build_model rejects unsupported or mismatched mode/source bundles. No failed-current recipe is substituted.

The seed formula is unchanged and uses each ACTUAL new prefix signature and h. H>1 geometry changes should change draws appropriately; do not relabel old A/B as new. H1 exact schedule and source kernels create an implementation BYTE gate. Production captured-asset/signed-floor API remains outside this research adapter.

## Qualification gates

Focused tests cover exact integer feasibility vs brute-force tiny cases, support projection, H1 schedule identity, huge H, N<=m, duplicated-fold mass collapse, unchanged current finite41 coefficient/weight/count bytes, H1 objective/seed/path bytes on prescribed states, chronological hashes/no future input, exact mapped-prefix cache registration and forbidden IDs, current/mapped failure isolation and source guards. Prescribed states estimate NO parameters; no historical request executes. A real source-bound old/current and H1 gate is still needed after admission, followed by metadata-defined matched short/medium/long cases and explicit overlap/support/tail/failure/cost reporting. No all-origin grid or automatic larger launch.
