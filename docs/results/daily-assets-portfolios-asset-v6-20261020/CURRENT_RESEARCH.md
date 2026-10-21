# Current asset-level accuracy research

The goal remains active. Baseline qualification, both complete baseline evaluations, the first full candidate and its independent audit are complete. This is not the evidence-based research stopping point.

## Official retained set: one main score

All methods below use the identical64,049 successful forecasts across1,264 streams at1,000simulations. Lower main Q is better. The fixed reporting reference is the worse completed TestFolio-style asset configuration, With Replacement.

| Method | Main Q | Better vs fixed reference | Recorded worker compute seconds |
|---|---:|---:|---:|
| Asset Blocks — With Replacement |0.6320715457043177|reference|13316.0431|
| Asset Blocks — Without Replacement |0.6310508866260718|0.1614784%|12956.4817|
| Asset Equal Mix / Smooth Gaussian Mean Uncertainty |0.6150071135191089|2.6997628%|11721.9583|

Every retained method completed64,049forecasts with zero native failures and zero scoring integrity errors. Source-bound checkpoint identities and unchanged score reduction were independently verified. Recorded worker compute includes input preparation, fitting, joint simulations and empirical CRPS, including identity-verified reused pilot forecasts. It excludes IO/setup/queue and is not a controlled speed comparison. No independence or significance claim is made; these are adaptively reused benchmark cases.

The current best is one fixed cohesive asset-level law: equal raw joint historical-block paths and per-asset GARCH(1,1) filtered historical simulations, joint empirical innovation rows, smooth training-only mean shrinkage, and jointly correlated Gaussian uncertainty around the mean. It evolves holdings and rebalancing inside each simulated path. There are no test-score-selected class or horizon routes.

## Further research

A100-path355-case screen favored the equal mixture over its filtered-only endpoint. A second same-count screen rejected ordinary nonwrapping replacement for the raw half. A distinct circular-start contrast improved the100-path screen, both355-case1,000-path development/reserved gates and the7,418-case broad gate. It is now running the full64,049-forecast evaluation with12workers, reusing8,128identity/hash-verified pilot checkpoints. The verified current best above is retained until complete independently checked full results justify promotion. An IID raw-component simplification passed2unit tests and is queued for a100-path screen after the shared forecast slots become free; it retains joint asset rows and the unchanged filtered component. All screens are exploratory, separate from official scores and full-panel catalogue counts. Plans freeze sources/cases/seeds before field forecasts. Twelve forecast slots are shared and bulk archival IO is coordinated.

The additive full-score diagnostic reconciles every dimension to canonical Q. It shows broad gains but some remaining family regressions; diagnostics receive zero additional score or selection weight. They guide hypotheses, not weights, routes or model promotion.

## Preservation and recovery

The private baseline archive preserves134,653files. The complete hybrid archive preserves72,258files, including all64,049full checkpoints plus pilots, source, controls/registry, seeds, admission and verification evidence. Every archived file was fresh-downloaded, physically restored and SHA256verified. These private releases and exact per-part manifest digests are recorded in closed-baseline-archive-receipt.json and closed-hybrid-archive-receipt.json. No active scientific originals were removed. The full-panel catalogue retains every completed implementation version, including losers; identical reruns count once.

Restore baseline shared/input archives and sealed numerical parents referenced by execution identities, then restore candidate archives at the research owner root. Verify every manifest path, size and SHA256 before use. Public code/evidence is separately fresh-download verified after publication. No production code or deployment was changed.
