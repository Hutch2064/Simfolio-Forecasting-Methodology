# Bounded public stationary-bootstrap refinements, 2026-10-19

Neither tested refinement wins. This is a small exploratory pilot, separate from the running fifteen-model French551 campaign. No campaign registry, original forecast source, checkpoint, worker allocation, production code, or historical score was changed.

## Frozen reference and selection

The reference is exactly `TestingNaiveControl` / `model_testing_joint_stationary_bootstrap_v1` from methodology revision `b958ed46cc2b341a024abbd8a8dea70ff569c01f`. It is not website Default. Inputs are the frozen equity551 panel fingerprint `e45e1b5bec4ad011f280a236bd9901c5ea75fd4282ab26072be2dd289e41e4c0`, immutable daily log returns and actual provider dates. Each French portfolio is one unit-weight asset. Training contains the entire prefix through the selected origin, inclusive; targets are never available to the parameter fit.

Before scoring, `plan.json` selected the lower-median series ID in every family-by-weighting stratum: five portfolio families with both weightings and the value-weighted broad market, giving eleven series. The representative is selected only from metadata ordering, not from returns or scores. Four origins are `rolling_01`, `rolling_24`, `rolling_48`, and `train_first_quarter_test_remaining`; their complete eligible horizons are 131, 3,402, 43 and 19,738 observations. Every integer daily horizon is scored, with 240 paths. The long temporal forecast has 6,579 training observations, about twenty years, and includes the full remaining history. Full scope is 44 origins per variant, with dependent overlapping horizons and related equity portfolios.

Seeds are the unchanged `coherent_daily_seed(origin_date, full_max_horizon, 240)`. Baseline and refinements share exactly the reference's row indices, Politis–White block rule, draw order and seed. Public replay is byte-identical at every daily increment on all 44 tasks and its saved empirical CRPS vectors are byte-identical to all 44 corresponding official public checkpoints.

## First refinement

Keep the public native simple/log round trips and fitted raw sample mean. Fit training Gaussian predictive likelihood EWMA using the existing `fit_ewma_qmle` implementation: training sample variance initialization, decay searched over [0,1], bounded optimum plus both endpoints, no fixed fallback. Predictive variance is computed before each observed residual. Standardize each training residual by its predictive standard deviation, center this empirical pool, and scale its empirical second moment to one. Bootstrap its entries with the unchanged public stationary indices. Forecast variance evolves recursively from the last fitted variance as `v_next=lambda*v+(1-lambda)*shock_squared`.

Generated daily returns use native simple-return clipping and multiplicative one-asset wealth accounting, including the native wealth floor. Shape and finite-value checks fail closed. Unit-RMS residuals and a raw-mean parameter do not guarantee the same expected forecast drift when blocked innovations interact with endogenous scale.

## One incremental alternate

After the first result failed, `alternate/plan.json` declared exactly one added fitted parameter, a variance-restoring persistence rho:

`v_next=(1-rho)*training_sample_variance + rho*(lambda*v+(1-lambda)*shock_squared)`.

The variance target is the same sample variance. Fit lambda and rho on [0,1] using training predictive Gaussian likelihood, two declared deterministic L-BFGS-B starts, and explicit nested EWMA rho=1 and constant variance rho=0 alternatives. Failed optimization raises. All means, blocks, seeds, empirical standardized residual construction, horizons, paths and accounting remain as above. Original public arrays are reused, not refitted. This second method is exploratory on the same already-inspected pilot and has no untouched validation set.

## Existing-family relation

These are implementations of an existing conditional-volatility stationary-bootstrap family, not a claim to invent EWMA residual SBB or GARCH residual SBB. The retained historical `bayesian_sbb_overlay_ewma_qmle` family already exists in the research catalogue. The exact frozen historical helper is copied to `reference/historical_ewma_reference.py.txt`.

Its implementation is distinct: bounded decay [0.80,0.995], update-before-current-residual likelihood/scales, fallback decay and variance floors; a Bayesian/constrained SBB base mean; and a deterministic mean-reverting future volatility multiplier with bounds. The present first refinement instead uses predictive variances, a full [0,1] decay search with endpoints, the public raw mean, a centered unit-RMS empirical residual pool, and stochastic recursive variance. The existing official `FRENCH1926-EWMA-QMLE` control uses independent Gaussian innovations and recursive EWMA. No historical results or model identities are inherited by either pilot.

## Scores and results

Use exact empirical CRPS of cumulative log-return distributions with the original self-inclusive squared ensemble denominator. Normalize every origin loss at h by `s_train*sqrt(h+h*h/n_train)`, with training sample standard deviation ddof=1; average origins within each series-by-horizon cell; weight horizons by `log1p(1/h)` and divide by `log(H+1)`; average the eleven series equally. The declared sensitivity removes h²/n from the reference scale. Legacy raw CRPS is retained separately. Missing origin tasks may not silently reduce a variant's denominator.

| Method | Primary v2 | Sensitivity | Legacy raw | Skill versus public |
| --- | ---: | ---: | ---: | ---: |
| Exact public stationary bootstrap | 0.8087918052044514 | 1.0220602751078658 | 5.448137148136759 | 0% |
| Recursive EWMA empirical residual SBB | 0.9237505484893547 | 1.1733344621897042 | 6.272173308587093 | -14.2136% |
| Mean-reverting empirical residual SBB | 0.9601634738190478 | 1.2348548191661435 | 7.137811516458586 | -18.7158% |

All variants have 44/44 finite tasks. The first refinement's long-horizon terminal log standard deviations collapse to roughly 0.12–0.24, versus 2.3–6.4 for public. Adding training-fitted mean reversion restores dispersion but worsens aggregate loss. It can also change drift through dependent innovations interacting with scale: for the equal-weighted size representative FR0027, quarter-origin public mean CAGR is 1.519 while the alternate gives 2.355; retaining the mean parameter does not isolate distributional dispersion. These failures reject these particular implementations, not all volatility conditioning.

`official_matched_comparison.json` compares both pilots against every official model using only existing exact case-origin-horizon checkpoint intersections. Each row reports its available task set and recomputed public loss on that same set. Different rows may have different coverage; they are not a common complete-panel ranking. `constant_mean_student_t` had no available execution manifest at the comparison snapshot and remains unknown. No official method was refitted.

## Validation, runtime and storage

`validation.json` records eight independent checks: scalar training recurrence; scalar forecast/accounting recurrence; the nested EWMA boundary; admissible training-only optimization and empirical residual normalization; three independent scalar v2 score oracles; and all 44 byte-exact public replays. The original launch encountered a JSON serialization error after the first task's NPZ had been saved. The array and error log remain under `incomplete-attempt`; only serialization was repaired, original plan retained, and the identical case was rerun. No score-based selection change occurred.

One nice-15 process ran at a time, with numerical thread counts fixed at one and a private Numba cache. Kernel timings are local diagnostic adapter/fit times, excluding exact CRPS/checkpoint IO unless separately specified; initial compilation and warm process state differ. They do not establish a general speedup, production click-to-visible performance, or a runtime comparison against the full catalogue.

All per-task original daily loss vectors, input context, seeds, fitted parameters, native metadata, mean and median CAGR, and selected raw terminal ensembles are retained. `raw_forecasts/` additionally contains every complete 240-by-H daily log-return array for all three variants: 1,477,212,704 bytes in 44 files. They were reconstructed from saved parameters with zero fitting calls, under a passed 1.5-GiB-growth/3-GiB-budget storage preflight. Their selected terminal ensembles match the original stored arrays byte-for-byte; `raw_forecasts_manifest.json` binds file and array SHA-256 hashes. No active campaign checkpoint was regenerated.

Run `implementation/run_pilot.py` or `alternate/run_alternate.py` with the frozen code `src` on PYTHONPATH and the retained catalogue52 virtualenv. The original source hashes are bound in both plans. Run `validate_candidates.py` for independent compact checks. `retain_raw_forecasts.py` reproduces the full daily ensembles solely from saved parameters and refuses to overwrite existing raw files; move restored raw files out of the destination before explicitly invoking a second replay. The input bytes, executable closure and environment lockfiles belong to the parent French551 archive; preserve them together for restoration.

Research expansion stops here. Any new design should follow the separate public-specific mean-versus-dispersion diagnosis and a fresh, predeclared validation subset.
