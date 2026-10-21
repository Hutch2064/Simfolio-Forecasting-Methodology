# Broad shared-fit feature screen

The current verified GARCH-fade hybrid remains the anchor, not the boundary of the search. This first factorial batch freezes216 coherent global model configurations:9 filter specifications ×3 innovation laws ×2 raw-fade states ×2 point-mean treatments ×2 mean-uncertainty states. This is exploratory feature/interaction testing, not216 official full-panel scores.

## Frozen mechanisms

The equal raw-circular/FHS mixture, joint asset simulation and in-path portfolio accounting remain. All assets in each configuration follow the same method, with parameters fitted from their training prefixes. The filter choices are constant historical variance; deterministic mean-reverting EWMA expected variance; variance-targeted GARCH; GJR; nonnegative variance-targeted HARCH1/5/22; moment-fitted joint log-OU stochastic volatility; its analytic conditional-RMS counterpart; training-shrunk AR1-GARCH innovations; and recursively simulated EWMA. Thus the batch covers non-GARCH laws as well as local changes to the incumbent.

Innovation choices are shared empirical residual rows, correlated Gaussian shocks and common-radial multivariate Student shocks. The latter fits a common degrees-of-freedom parameter by training marginal composite likelihood with the Gaussian endpoint. Gaussian and Student GJR variants target their symmetric negative second moment; stationary support is enforced before forecast recursion. No path is clipped, excluded or selectively scored.

Point-mean shrinkage and Gaussian joint batch-mean uncertainty are independently on/off. Raw conditioning is independently absent or the verified training-GARCH variance fade. The fade flag applies to the raw half; each filtered variance law has its separately documented future recursion. No fixed horizon cutoff or outcome-fitted decay is introduced.

The log-OU proxy uses log(standardized squared return+1/n), so zero observations remain finite without deleting data. A measurement-noise-corrected moment fit and Kalman filter yield stationary logvariance state parameters. Joint future logvariance innovations use training proxy-change correlations; RMS uses the conditional Gaussian second moment of the same fitted state law. This is a cheap new moment-SV approximation, not a claim to reproduce the historical MAP/Laplace rough model. HARCH uses training NNLS followed by explicit stationary variance targeting. Estimator constraints are distinct from prohibited clipping of simulated returns.

## Earlier-panel evidence and limits

Read the saved full52 refinement, full80 ranking and expanded132 source-bound score records. The full80 leading recorded methods include the M261/EWMA complete-path pool, data-selected rough-volatility timescales, Student dynamic dependence and OU pooling. The 52 refinement favored deterministic variance integration and moment-corrected shock nodes. These motivate the Student, SV/RMS and mean-law contrasts here; old scores and percentages are not numerically comparable to the expanded panel's main Q. Dynamic-correlation and faithful expensive-state adapters remain additional research directions, not completed tests in this first batch. FAST is not restored to the official retained set.

## Efficient bounded execution

Select the training-position-median case from each of the70 existing families, then add only the asset-quarter support required for the unchanged main reducer. Selection uses no loss values. Every configuration receives identical dates, assets, horizons and100-path context seeds. Each case prepares raw fits and mean treatment once, then its filter fits; all24 combinations for each filter reuse the verified bundle. Twelve shared workers pipeline fit preparation and forecast jobs, with a bounded four-bundle per-worker memory cache. Cached histories and sources are hash-bound.

An anchor configuration must reproduce the saved incumbent loss vector byte for byte on every selected real case. Exact tests also check synthetic anchor equivalence, all216 path ensembles, chunk invariance, covariance/variance targets and SV second moments. Candidate failures remain visible and incomplete configurations are excluded from the complete-coverage screening ranking. Native fitting/generation failures are reported separately from integrity errors.

Only the existing canonical overall score ranks complete matched screens. Family/horizon diagnostics have zero extra weight. Low-path results are preliminary and adaptively reused, not independent validation or significance evidence. Credible survivors require broader1000-path matched development/reserved/broad checks before any full-panel admission. Only both baseline configurations and the current verified best appear in official scores. All historical full-panel versions remain in the remote-preserved catalogue; screens do not increment its full-panel count.
