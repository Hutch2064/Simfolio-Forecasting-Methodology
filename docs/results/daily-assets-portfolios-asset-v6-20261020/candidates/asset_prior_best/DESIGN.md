# Prior-best approach at asset level: unadmitted candidate

This is the user-requested asset-level counterpart of the previous Equal Mix /
Smooth Mean / Gaussian Mean Uncertainty winner. It is not yet a field winner or
an admitted full-panel model. Its raw component must inherit the better of the
two completed new asset configurations; the worse remains the sole reporting
reference. No parent is selected from partial scores.

Half the simulated paths use that raw joint block sampler; half use filtered
historical simulation. Each constituent is fitted with the previous best's exact
two-start variance-targeted GARCH(1,1) estimator. Historical standardized residual
rows are sampled jointly across assets, then each asset's conditional variance is
advanced along its own path. Actual constituent paths and portfolio holdings are
calculated inside every scenario at the panel's fixed rebalance endpoints.

The mean-uncertainty extension uses all training observations split into
floor(sqrt(n)) contiguous batches. Weighted batch-mean outer products estimate a
joint long-run covariance, divided by n and multiplied by df/(df-2) to preserve
the old matched-Gaussian variance convention. Smooth per-asset mean regularization
and antithetic stratified latent Gaussian draws preserve correlated uncertainty;
independent per-asset mean draws would discard this dependence. This is a modeling
approximation, not a claim of an exact Bayesian posterior or optimal estimator.

The old scalar fit functions are copied verbatim and checked by AST equality.
New asset and portfolio wealth are accumulated in log arithmetic. The old scalar
simple-return clipping/floor convention is not imposed on the new joint model;
this difference is declared explicitly. Every path and variance must remain
finite. Failed fits or invalid paths are disclosed, never silently excluded.

Seven training/numerical tests pass. A real-input preflight checked710prefixes
from the already-frozen development/reserved roster across70families:1190asset
fits,zero failures. These are training-only checks,not held-out forecasts or
accuracy evidence. The reused roster is not independent validation. Both field
phases remain355cases each and use1000simulations with12workers when the baseline
queue releases its slots. Promotion uses only the unchanged main score on exact
matched coverage, followed by broader/full testing. Diagnostics have zero score
weight and no separate promotion objective.

Before any field outcomes, a broad follow-through roster was frozen:7418cases
cover all1264series and70families. It uses the first,middle,last remaining rolling
origins and remaining temporal splits,excluding the development/reserved cases.
Additional individual-asset calendar support is included where required by the
unchanged main score. No earlier pilot cases were needed as support replays.
This phase is eligible only if development and reserved both improve the main
score relative to the better completed asset parent. It is not a claim of fresh
independent validation or a substitute for a successful full-panel run.

## Academic context and limits

The FHS authors describe filtering historical shocks by conditional volatility
and preserving cross-series dependence by parallel resampling. This supports the
construction, not any claim that this hybrid improves CRPS:
https://www.filteredhistoricalsimulation.com/ .

Multivariate batch means are studied as covariance estimators under explicit
regularity conditions in Vats, Flegal and Jones, *Multivariate Output Analysis for
Markov chain Monte Carlo*: https://arxiv.org/abs/1512.07713 . Their results are not
a proof of stationarity, calibration or predictive optimality for financial
returns. The present weighted-batch extension and Gaussian uncertainty law must
be judged by the frozen forecast tests.

Vats and Flegal also study negative finite-sample bias in time-average covariance
estimates: https://arxiv.org/abs/1809.04541 . No lugsail correction is included in
this candidate. It is only a future mechanism to investigate if actual forecast
evidence supports underestimating mean uncertainty.
