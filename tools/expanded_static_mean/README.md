# M261 raw historical asset log mean

Candidate: `asset_m261_raw_historical_log_mean`.

This standalone research ablation retains the exact M261
`asset_m256_data_timescales_existing_sv_rate` simulation, including its
conventional and rough fits, return standard deviations, finite innovation
nodes, Student corrected-DCC dependence, seeds, draw order and path count.
It changes only each asset's forecast location from the existing
volatility-scaled mean to its raw training daily log-return sample mean.
No mean shrinkage, horizon taper, portfolio mean fit or mixture is introduced.

For each asset and day, after the complete original simulation and before
the existing portfolio rejoin, the adapter applies

`new_asset_path = old_asset_path + training_asset_log_mean - old_mean_curve`.

The old mean curve is obtained from the same unchanged
`moment_return_curves(learned.student.original_fit(data), horizon)` used by
M261's actual asset mapper. Its SD output is discarded; no SD is recomputed
using the replacement mean. Every realized centered asset residual is
preserved to floating-point translation precision. The original full asset
tensor is translated in place by NumPy's native addition, without a second tensor.
Portfolio rebalancing and buy-and-hold compounding then operate on the
translated assets, rather than translating an aggregate portfolio path.

The research runner must select this adapter alone in its process, as for
the retained M261 adapter. Its temporary synchronous rejoin callback is
restored even if fitting, simulation or rejoin fails. No production source
or global moment function is edited. Unit and synthetic mechanism checks
are not OOS performance evidence. No equal pool is registered here.
