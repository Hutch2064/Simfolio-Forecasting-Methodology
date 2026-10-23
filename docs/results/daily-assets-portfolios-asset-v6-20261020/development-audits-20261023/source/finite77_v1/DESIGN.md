# Finite77 source-only prototype

This adapter adds no financial law. It restricts the qualified filtered-residual PPW backend to the existing finite41 core and36 empirical residual-block derivatives. Both modes are prepared from one shared historical path bank, with separate submatrix QPs. Scientific execution is not admitted by this source receipt.

## Exact identities and eligibility

finite41 core: `[1,2,3,4..15,22..33,112..123,166,167]`.
finite77 adds exactly `[170..193,242..253]`: constant/EWMA_reverting/SV_RMS × empirical innovations × sample/shrink/zero means × mean-uncertainty false/true × stationary/circular residual blocks. Every derivative's base IID ID is in the core. No Student, recursive EWMA, stochastic SV_path, GARCH/GJR/HARCH/AR1_GARCH, raw-return PPW168/169, EGARCH278..283 or new mean law enters. Sample/shrink/zero retain their qualified semantics.

The schedule's original eligible set is restricted before every historical prefix fit, cube and QP. Full-current fitting receives only the union of actually selected allowed recipes. At requested H1, all36 derivatives are excluded before fitting and QP, leaving exactly the same effective core catalogue. The identical H1 solver result is copied with the same170-slot vector, rather than allowing duplicate columns or allocating a separate MC advantage. At H>1 a77 vector retains the qualified284-slot backing bank (all other slots exactly0); core vectors stay170 slots for byte comparison with the frozen control.

At current selected block length1, each derivative's learned mass transfers to its base IID recipe BEFORE even allocation, using the unchanged PPW alias arithmetic. Historical QP coefficients/weights remain separately retained because historical length fits need not equal current lengths. A failed current derivative is not silently aliased away: it remains unsupported for the selecting mode. One mode's solver or selected-current failure is explicit without dropping a valid other mode or substituting/renormalizing weights.

## Unchanged numerical backend and law

GroupWork, all_chunks and Pool.terminal_chunks are direct qualified PPW backend references. PPW estimates corrected lengths from each family's SAME-prefix standardized residual pool, with the same target portfolio weights; the original common-length residual/residual-square heuristic is unchanged. Existing row RNGs, paired starts, forward chronological continuation, joint cross-asset rows, restart decisions, chunk behavior and family recursion are untouched. Existing grouped coefficient helper, unique-v3 solver, endpoint CRPS/sqrt(native validation horizon), N2, half_doubling folds, represented masses, seeds and final daily target-weight portfolio accounting remain unchanged.

No imported globals are assigned. Returned bundles and model IDs bind mode and canonical transitive local source hashes; dated prefix, portfolio and horizon checks remain in the final qualified parent sampler. The PPW module imports an EGARCH helper as an existing dependency, but this adapter never schedules its IDs or requests its fit. All such source dependencies are bound; third-party runtime versions belong to the runner. This research API does not replace production's captured-asset/signed-weight generator.

## Distribution and moment scope

This is conditional on finite fitted states and a fixed finite H, with long-only target weights. The added family variance forecasts are deterministic; empirical joint residuals and block indices have finite support. Existing Gaussian mean-uncertainty has finite exponential moments at fixed H. Hence finite positive wealth moments hold within that conditional fixed-H scope. This does not supply uniform-in-H bounds, a reasonable-tail guarantee, a martingale mean claim, or a general signed/leveraged-wealth theorem. A zero offset under block-resampled predictable residuals is not zero expected return. Corrected PPW is motivated by long-run-variance estimation; this common multivariate length is not proven CRPS-optimal. Previous development evidence motivates a controlled test only.

## Source/mock qualification and next gate

Seven mock tests cover exact semantic IDs and direct backend binding; H1 pre-fit/pre-QP exclusion and identical core weights; A/B/weights/240 allocation BYTE parity with frozen finite41; restrictions in every historical stage and no parent mutation; length1 exact mass/count alias before allocation while retaining training fits; cross-mode/source rejection and isolated QP failure; failed current derivative retained as unsupported without invalidating the core. Tests use no historical financial fit or forecast.

A meaningful regression found and fixed mixed170/284-vector failure-index handling before freeze. No frozen parent was changed. Before any accuracy claim, root must admit real-prefix old/new finite41 BYTE control gates, forced added-family reference paths at H5/H21 even when selected mass0, and explicit availability/failure/tail diagnostics on metadata-defined asset and portfolio requests. No cap, clipping or path exclusions may repair failures. Broader scoring remains separate and must use the unchanged external score with complete matched support.

API: prepare_all(history,weights,H,dates,seed_replication=0,fit_cache=None,solver=None,width=64) returns finite41 and finite77 bundles. build_model(mode,dates,bundle) rejects source/menu mismatch and unsupported mode receipts. Fields unsupported_solver and unsupported_selected_current follow the already reviewed mean-reliability-v2 convention. No scientific launch or production approval is implied.
