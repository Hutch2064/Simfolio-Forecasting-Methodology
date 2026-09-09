# Canonical Dense Out-of-Sample Methodology

## Scope

This document describes the evaluation contract represented by `canonical_dense_daily_2026_08_23`. It defines how a forecasting specification is evaluated. Catalogue membership is governed separately so the same evaluation contract can be applied to the canonical publication catalogue or a larger research catalogue without changing the test itself.

## Portfolio panel

The study evaluates 80 deterministic multi-asset portfolios on one common daily return window from 1979-12-31 through 2026-05-13. Each portfolio contains two equity series, two fixed-income series, and two alternative-asset series. The three sleeves receive equal one-third target weights, so each security begins with one-sixth portfolio weight. Portfolios use one of four rebalancing conventions: buy-and-hold, monthly, quarterly, or annually.

Portfolio selection is deterministic. Eligible ticker pairs are enumerated by asset class. Candidate portfolio/rebalance combinations receive seeded tie-breaking values from NumPy's default random generator with seed `20260528`. A greedy balancing rule chooses 80 unique portfolios by minimizing, in order, cross-ticker usage imbalance, usage pressure, squared usage concentration, seeded jitter, and the deterministic portfolio key. The resulting panel uses 38 distinct simulated series and has rebalance counts of 24 buy-and-hold, 20 monthly, 18 quarterly, and 18 annual portfolios.

The standalone implementation is in `src/simfolio_forecasting_methodology/panel.py`.

## Forecast origins

Each portfolio contributes 48 rolling origins selected across the full eligible historical sequence using the `full_history_even` policy. Eligibility must be established before subsampling and must respect the forecast-horizon validity/no-look-ahead rule.

The evenly-spaced selector uses rounded indices from a linearly spaced grid over the ordered eligible-origin sequence. If rounding creates fewer distinct indices than requested, unused indices are added deterministically in ascending order.

Three temporal holdout origins are evaluated in addition to the 48 rolling origins. They correspond to training on the first 25%, 50%, and 75% of the common history. The split position is `floor(n * fraction) - 1`, clipped to the retained minimum-history and final-observation bounds. Therefore each portfolio contributes 51 origin tasks and the 80-portfolio panel contributes 4,080 tasks.

The standalone implementation is in `src/simfolio_forecasting_methodology/origins.py`.

## Horizons

The dense protocol scores every daily forecast horizon that is eligible under the preserved origin rule. A fixed horizon cap is not applied. The longer fixed list used by earlier research generations is retained only as origin-selection context and must not be substituted for the dense daily scoring grid.

This distinction is important: an earlier experiment generation used 33 origins, 300 simulations, and 15 fixed horizons. That is a separate historical protocol, not an alternate description of this study.

## Simulation count

Each origin/model evaluation uses 240 simulated forecast paths. Random-number construction is treated as part of the reproducibility contract. Model-specific seed derivation is being promoted into the standalone repository only where it is recoverable from retained source evidence; unverified seed logic must not be silently replaced by a new policy while calling a result a verbatim reproduction.

## Score

For realized terminal log return `y` and equally weighted forecast samples `x_1, ..., x_n`, the exact empirical continuous ranked probability score is

`CRPS = mean_i |x_i - y| - (1 / (2 n^2)) * sum_i sum_j |x_i - x_j|`.

The implementation evaluates the pairwise term using sorted-sample rank algebra, avoiding an `n x n` allocation while preserving the exact empirical score.

No weighted-interval-score approximation is used for the canonical dense ranking.

## Aggregation

Aggregation is cell-first:

1. For each portfolio and daily horizon, average the CRPS losses across forecast origins that contribute to that cell.
2. Give each resulting portfolio-horizon cell equal weight in the final model score.

The canonical retained result has 701,280 scored portfolio-horizon cells per comparable model.

Pooling all origin rows directly would assign greater weight to cells with more contributing origins and would therefore implement a different estimand.

## Catalogue independence

The canonical publication catalogue contains 175 retained specifications. A broader master research catalogue is a superset and is intended to use this same dense protocol by default. Expanding the set of models being evaluated does not change portfolio construction, origins, horizons, simulation count, score, aggregation, or failure policy.

Historical protocols remain reproducible only through explicit protocol selection; they are never implicit defaults for master-catalogue runs.

## Reproducibility boundary

A complete reproduced result should identify at least the catalogue digest, protocol digest, portfolio-panel digest, data fingerprint, random-seed schedule, software environment, and source revision. A result missing any of those identities may be useful research evidence but should not be represented as a byte-for-byte or path-for-path reproduction of the canonical study.
