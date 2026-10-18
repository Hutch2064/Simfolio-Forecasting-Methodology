# Unchanged canonical controls

Set `SIMFOLIO_EXPANDED_CONTROL` to exactly one of the IDs below, then pass
`--adapter tools/expanded_controls/models.py --model-id "$SIMFOLIO_EXPANDED_CONTROL"`
to `tools/expanded_panel/run_panel.py`. Each worker imports only its selected
registry model. The candidate is the actual registered class, with no wrapper
around fitting, simulation, seeds, dates, or terminal semantics. No new model
IDs or statistical laws are introduced.

| Exact ID | Distinct control | Historical original80 CRPS |
| --- | --- | --- |
| `naive_iid_historical_portfolio_bootstrap` | Raw IID historical portfolio returns | 0.3385054919574415 |
| `constant_mean_gaussian` | Historical sample mean and constant Gaussian volatility, direct horizon ensembles | 0.33541860381061706 |
| `bayesian_sbb_overlay_ewma_absolute` | Bayesian stationary bootstrap with absolute-return EWMA volatility overlay | 0.25801165086966715 |
| `factor_ff6_ridge_residual_sbb_none` | FF6 ridge factor and residual joint stationary bootstrap | 0.3320269111881849 |

The score source is `docs/results/combined-176-score-ranking.json`; canonical
membership and rounded scores also appear in
`src/simfolio_forecasting_methodology/resources/catalogs/canonical_175_part1.csv`.
The ranking explicitly marks all four `historical_score_verified=false`.
Gaussian and IID rows have `bounded_source_parity_verified_historical_linkage_blocked`;
EWMA and factor rows have
`source_parity_verified_bounded_historical_score_linkage_qualified`.
No validated original80 cell-loss file for these four is supplied by this
adapter. The historical scalars must not be used to claim a newly verified
equal-block result: execute both original80 and single52 and retain their
complete checkpoint/cell evidence. The historical ranking motivates controls;
it does not establish present runtime or expanded-panel performance.

The Gaussian source implementation samples direct terminal distributions at
each horizon; its rows need not describe coherent daily paths. Its terminal
semantics remain declared by the original class. The other implementations'
original path and seed laws are equally unchanged. In particular these source
classes can derive RNG seeds from the dated origin, dense horizon grid, model
ID, and simulations rather than consuming `context.seed`. The original context
is passed by the canonical runner without substitutions.

These are portfolio-return models on both blocks. A single-asset task supplies
that asset's portfolio return series. The factor model retains its original
dated offline FF6 alignment and fails closed when its minimum aligned history
is unavailable. It is selected over its nearly identical EWMA suffix variant
to avoid duplicating a family; the raw descriptors of both variants omit
`residual_overlay`, so the public implementation resolves both to `none`.

There is no exact website-naive raw stationary-bootstrap candidate registered
in this public catalogue. IID historical bootstrap is not relabeled as that
model. A future website-naive extraction would require a distinct new ID,
source verification, and fresh runs of both blocks; it is outside this adapter.

`TIMINGS` and `FIT_DIAGNOSTICS` intentionally remain empty. The panel records
total task duration; no decomposition or inferred speed claim is added here.
