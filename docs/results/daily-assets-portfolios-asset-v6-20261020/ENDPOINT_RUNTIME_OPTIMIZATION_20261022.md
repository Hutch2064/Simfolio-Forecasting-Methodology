# Endpoint runtime optimization and lifecycle evidence

**Evidence date:** 2026-10-22
**Status:** completed bounded diagnostics; no model promotion and no full-panel completion claim.

This note records a speed-focused implementation comparison for the frozen forecast research protocol. It does not change the model catalogue, historical schedule, requested horizons, two internal paths, 1,000 final paths, joint asset paths, portfolio accounting, or fixed with-replacement reporting reference. The full expanded-panel run is separate; this note reports no score for it.

## Numerical checks

An independently verified, matched 355-group broad comparison covered 1,065 endpoint forecasts. Training traces and invalid masks matched throughout, with no forecast or scoring errors. Raw-loss vectors were byte-identical for 1,064 of 1,065 forecasts. The candidate's mean Q was 0.9173317358408478 versus 0.9173318111047232 for the reference (candidate minus reference: −7.526387535428114 × 10⁻⁸). The one changed result crossed an integer final-path allocation boundary after a small optimizer-weight drift; it is recorded as numerical micro-drift, not exact parity.

A separate saved-short replay audited 1,293 groups across horizons 1, 5, 10, and 21 for three models (15,516 forecast records). All 10,344 baseline loss vectors were byte-identical; 5,170 of 5,172 endpoint loss vectors were byte-identical. The supported canonical subset contains 1,292 asset groups (5,168 requests per model). One portfolio group in 2012-Q3 has no matched asset support in this partial replay, so the original all-1,293 aggregate is not a canonical aggregate. All 12 excluded raw losses were still audited and byte-identical. On the supported asset-only subset, endpoint Q changed by +1.2862842450056533 × 10⁻⁶. This is a diagnostic subset result, not a portfolio-weighted or full-panel result.

## Runtime measurement

The whole-group serial comparison measured four groups, each with 12 complete model/horizon forecasts, over three interleaved repetitions with fresh training, two internal paths, and 1,000 final paths. All 12 raw-loss vectors were exact within each group. The observed CPU speed ratios were 1.4013×, 4.7221×, 1.7893×, and 4.3740×. These are local research measurements for the selected groups; they are not population-wide, production, website, or browser-latency claims.

The evidence-led stop review found no remaining measured, non-speculative implementation improvement in the investigated bottlenecks. It is not proof of globally minimal runtime or unchanged outputs on every unseen request. Reducing recipes, folds, paths, or optimization accuracy was outside the accepted parity scope.

## Pause and recovery check

A real controller-cycle check observed four workers running, drained to zero on pause, then resumed with twelve workers. It verified preservation of 51,734 pre-existing checkpoint files and observed 16 new checkpoint files during the cycle. This verifies the tested research lifecycle path only; it is not a scientific score or a production claim.

## Recovery archive

The frozen code, control/recovery tools, plans, closed qualification outputs, and compact verification receipts are preserved in a private release. The release has 4,182 inventoried files (105,589,967 source bytes) across five parts. Every part was freshly downloaded and disk-restored with file hashes and modes checked; local originals were retained. The archive excludes active checkpoint/output trees, baseline/provider inputs, production state, and mutable goal state.

[Private recovery release](https://github.com/Hutch2064/simfolio-engine/releases/tag/research-speed-lifecycle-v1-20261022T045103Z) (repository access required).

The release receipt SHA-256 is `ab842ba6e760cf5175a66ef30a5702bc6deb54817ea2b5aef99d20e937ac8e69`; the file inventory SHA-256 is `397686b3054c085783a78736fecf3d7e538bf282a2a45fc5d4f9523459002330`. The private archive includes the full file-level manifest and restore instructions. The [machine-readable evidence record](endpoint_runtime_optimization_20261022.json) also records the frozen runner, lifecycle helper, source, and verification-receipt hashes.
