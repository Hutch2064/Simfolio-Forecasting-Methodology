# White-paper dense OOS methodology

The public white-paper provenance PDF has SHA-256
`0fb61f974e5c035833564eaf8b47d247a720ab136fd8bfa17eeac9198051d86e`. The
executable protocol identity is
`canonical-whitepaper-dense-daily-v1` with protocol fingerprint
`24bace549583831ccfa016ce62d3f604020fa7b2e0b36574cb697aa02baa4ba3`. The
canonical scored panel is the 80-row, 52-ticker resource
`resources/panels/scored52_portfolio_panel.csv`, whose byte hash is
`fc42d3a5b7b0272cab1b4fc5db998d059a3f8fed93f6275295331fdc744f2f36` and
semantic fingerprint is
`c017963c9eb772e9bac09bb7a84ae975443cbb99cd127d666a9b0ac8103933d3`.
Its rebalance counts are annual 19, monthly 26, none 18, and quarterly 17.
The scored portfolio return history is constructed from the full source price
history before the common-window trim, with initial capital 10,000, equal six-way
weights, no cash flows, and the source 15 bps turnover cost.

Each portfolio has 48 full-history-even rolling origins and three temporal
origins, giving 4,080 origin tasks at 240 simulations per task. Rolling
candidates are calendar quarter ends at positions at least 504. The preserved
selection horizons are 21, 42, 63, 126, 189, 252, 378, 504, 756, 1008,
1260, 1512, 1764, 2016, 2268, 2520, 3780, 5040, and 7560 trading days. A
rolling task scores every daily horizon from one through
`min(forward_days, floor(training_observations / 4))`. Temporal positions use
`floor(n * train_fraction) - 1`, clipped to 79 through `n - 2`, and score every
future daily horizon. With 11,687 dates, the largest temporal future is 8,766
horizons, so the fixed portfolio-horizon denominator is 80 × 8,766 = 701,280
cells.

Forecast models return daily log-return increments. The evaluator cumulatively
sums those increments to terminal log returns, computes exact empirical CRPS
with the `n²` pairwise denominator, averages origins within each
portfolio-horizon cell, and then weights cells equally. Nonfinite forecasts,
missing tasks, failed origins, and missing cells fail the fixed denominator
rather than changing it.

Seeds retain the source call contexts. The BLAKE2b helper uses null-delimited
string parts, an eight-byte digest, little-endian decoding, and reduction
modulo `2**32 - 1`. Daily coherent benchmark calls use
`asset_level_naive`, `forecast_oos_all_daily_coherent`, origin date, maximum
horizon, and simulation count. Asset dependence calls use
`copula_alternatives`, candidate ID, origin date, maximum horizon, and
simulation count. Fixed-horizon candidate calls use
`forecast_oos_candidate`, origin date, the available-horizon tuple, candidate
ID, and simulation count.

The historical appendix remains provenance only. Its normalized LF resource
hash is `6691eb1daf38cc2aae6708fe1d155aab7975f29ebc6ada197f46bd8d9a8ecf70`;
the original CRLF representation is
`847946335fea3473e987af23f528483d66ab265e85596f5d226216091fb476b4`. It has
38 tickers and rebalance counts annual 18, monthly 20, none 24, quarterly 18,
so it is not the executable scored panel.

Retained historical wrappers include a branch that indexed daily path columns
directly instead of cumulatively summing increments. The executable evaluator
uses the daily-increment contract above, while parity of that historical
branch and its stored scores remains unresolved. No score is silently
relabelled as a reproduced result until that adapter is reconciled.
