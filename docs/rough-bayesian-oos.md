# Joint Bayesian rough volatility candidates

Eight new asset-level candidates are implemented in `tools/rough_bayesian` and
included in both public catalogues. The canonical catalogue now has 191 entries;
the broader active research catalogue has 383. Their official scores are blank
until the complete canonical panel finishes. The existing production Frontier
and the completed eight-factor rough model are included as unchanged controls.

| Candidate | Question | Volatility inference |
| --- | --- | --- |
| `asset_rough_volterra_sv_eight_factor_bayesian` | Does parameter uncertainty improve the winning rough overlay? | Unchanged eight-factor kernel; collapsed Gaussian filtering + parameter MCMC |
| `asset_rough_volterra_sv_accuracy_lift_bayesian` | Does an accuracy-controlled kernel improve that winner further? | Tempered fractional Volterra covariance; adaptive lift + parameter MCMC |
| `asset_bayesian_lifted_rfsv_ess` | Does jointly inferred fractional OU log volatility improve forecasts? | Full latent Gaussian innovations and parameters; ESS/HMC + MH |
| `asset_bayesian_lifted_rfsv_tight_ess` | Is the nominal kernel resolution sufficient? | Same target family, tenfold tighter kernel tolerance |
| `asset_bayesian_lifted_rfsv_ess_terminal_pgas` | Does particle ancestor sampling improve inference efficiency? | Full-path ESS/HMC plus exact terminal-block PGAS, same posterior as nominal arm |
| `asset_bayesian_lifted_rfsv_leverage_ess` | Does return/volatility leverage improve forecasts? | Jointly inferred within-asset rho |
| `asset_bayesian_lifted_rough_heston_hmc` | Does square-root fractional variance outperform log volatility? | Compiled analytic-adjoint HMC + parameter MH |
| `asset_bayesian_lifted_rfsv_student_ess` | Does a heavier-tailed return law help? | Student-t degrees of freedom inferred jointly |

The mean remains the production historical fixed-mean forecast. Dependence
fitting, Gaussian copula uniforms, data, origin dates, trading calendar, policy
rebalancing, costs, daily return clipping and the public scorer remain unchanged.
The six canonical alternatives replace the production variance engine. Their return
likelihood and predictive innovation law match: Gaussian, or variance-standardized
Student-t for the separately identified Student candidate. These six use their inferred volatility process directly in predictive returns.

## Incremental upgrades of the winning rough model

Both overlay upgrades retain the production daily mean and variance curve,
centered empirical return nodes, and the existing dependence/rebalancing shell.
They infer H, kappa and log-volatility scale per asset with two parameter MCMC
chains. Latent Gaussian states are marginalized by the Kalman filter and their
terminal posterior uncertainty is sampled for each predictive path. The training
observation equation deliberately retains the winner's winsorized log-square
proxy, its fixed training mean, and Gaussian measurement variance pi squared / 2.
This is a Gaussian quasi-likelihood comparison, distinct from the six raw-return
likelihood alternatives below.

The eight-factor arm keeps the original kernel and priors, replacing its MAP
parameter estimate with actual posterior draws. The accuracy-controlled arm
instead targets the continuous tempered fractional Volterra kernel
`exp(-kappa*t)*t**(H-.5)/Gamma(H+.5)`. Its exact stationary autocovariance is
computed with Tricomi's U function. A three-point log-rate quadrature is selected
against that covariance before fitting, with absolute autocorrelation error below
0.001 on a mesh covering 17 H values, three kappa values and daily lags through
25,200. Dense independent checks cover additional H/kappa points. The selected
16-bin grid yields 33 states after negligible-memory factors are aggregated and
an independent white factor retains the analytically known unresolved zero-lag
variance. This is a daily-grid approximation with a tested accuracy budget.

For either upgrade the multiplier is
`exp(.5*(latent-predictive_mean)-.25*predictive_variance)`, preserving conditional
expected squared multiplier equal to one. The first forecast state is advanced
one day from the terminal filtering posterior. The two changes therefore test
parameter uncertainty and kernel accuracy while preserving the winning model's
production variance anchor. The old rough winner remains an unchanged control.

## Fractional OU / RFSV model

Let alpha = H + 1/2. The stationary Gaussian fractional OU driving kernel is

```
g(t) = t^(alpha-1) / Gamma(alpha)
k(t) = g(t) - kappa * integral_0^t exp(-kappa*(t-s)) * g(s) ds
```

Its spectral transfer is `(i*w)^(1-alpha)/(kappa+i*w)`. This signed kernel is
essential: a positive mixture of ordinary OU covariance functions is a different
model. The stationary log variance is `level + scale * X`, with X normalized to
unit stationary variance. Scale therefore parametrizes stationary log-variance
SD; physical kernel amplitude is its corresponding reparameterization.

We discretize the Brownian driver on the observed daily grid using cell integrals
of k. For the independent direct reference, those integrals are differences of
`t^alpha/Gamma(alpha+1) * hyp1f1(1,alpha+1,-kappa*t)`. The production candidate
implementation approximates g's Laplace measure by three-point Gauss-Legendre
quadrature in log-rate intervals, then applies the fractional OU transformation
and exact daily cell integration to each exponential. All factors share the same
daily Gaussian innovation. Initial factors have their joint stationary Gaussian
prior, including their correlation with the mean-reversion factor.

The grid is selected before inference from successively finer grids against exact
fractional cell integrals, at 17 H values over the entire prior range and daily
lags through 25,200. The tolerances are 0.01 and 0.001. Independently tested signed
fractional OU kernel errors also meet the respective budgets on the specified
H/kappa test grid. Machine-precision white contributions are aggregated, with
first-nonzero-lag impulse error bounded by `1e-15 * sum(weights)`. The current
selection yields 32 fractional factors plus one OU factor, or 47 plus one for the
tighter arm. These counts are outcomes of the error checks, not fixed choices
such as the old eight factors. The tighter arm is an approximation comparison
within the same model family. The budgets concern the daily projected kernel;
they are not a claim of exact continuous-time or intraday simulation.

We infer H, log kappa, log scale and level, together with all historical latent
innovations and stationary initial uncertainty. Priors and proposal bounds are
fully retained in each catalogue definition. Leverage additionally infers
`atanh(rho)`; the Student candidate infers `log(nu-2)`. In the leverage model,

```
r_t - production_mean = exp(h_t/2) * (rho*W_t + sqrt(1-rho^2)*epsilon_t) / 100
```

h_t depends on earlier W innovations. Thus leverage uses the same conditioning
in estimation and prediction. The Gaussian return shock retains the production
cross-asset copula; an independent within-asset innovation completes the joint
return/volatility driver. The Student candidate has zero leverage and applies
the variance-standardized Student inverse CDF to the same dependence uniforms.

## Lifted rough Heston model

Variance is `theta + sum_i U_i`, where U_i is the contribution of a fractional
factor, with phi_i its exponential decay and a_i its daily integrated kernel
weight. All factors share Brownian W. Starting variance is theta. We use the
published drift-implicit, full-truncation approach on the daily grid:

```
v_plus = max(theta + sum(U), 1e-10)
common = eta * sqrt(v_plus) * W
v_next = (theta + sum(phi*U) + sum(a)*(kappa*theta + common)) / (1+kappa*sum(a))
U_next = phi*U + a*(common-kappa*(v_next-theta))
```

H, kappa, theta, eta/sqrt(theta), rho and the complete historical Gaussian driver
are inferred jointly. This is a stated numerical approximation to rough Heston,
with fractional kernel error checked before inference. A native analytic
adjoint differentiates this exact implemented scheme, including truncation, and
is verified against finite differences. Eight leapfrog steps update the full
innovation vector; the MH correction preserves the posterior. HMC step-size
adaptation occurs only during warmup. The log-return mean remains the production
mean, rather than introducing a risk-neutral price-drift convention.

## Inference and speed

Each candidate uses two chains, each with 2,048 warmup iterations and at least
8,192 retained draws, without thinning. If training-only split R-hat / bulk ESS
diagnostics fail, sampling continues from the exact saved RNG and chain state,
doubling retained draws up to 65,536 per chain. Adaptation remains frozen during
extensions; no sampling restarts, score-based stopping or future-data decisions
are used. Every predictive path uniformly selects one joint parameter/terminal
state draw from all retained draws. The cache retains the exact 240 joint draws
chosen for the configured ensemble and the complete diagnostic trace, avoiding
storage of unused terminal factors without changing the forecast.

Parameter updates combine symmetric component MH and a joint covariance MH
proposal learned only during warmup. ESS changes stationary initial uncertainty
and every historical innovation. A four-step native analytic-adjoint HMC update
also rejuvenates the complete Gaussian latent vector; its Gaussian and Student
likelihood gradients are independently checked against finite differences. The
PGAS arm additionally rejuvenates the last 32 driving innovations with eight
particles. Ancestor weights evaluate every remaining return in that block
exactly. It is an ESS/HMC/PGAS hybrid, rather than scalar AR(1) PGAS or an
ancestor-weight approximation that truncates fractional dependence.

Every asset fit retains rank-normalized and folded split R-hat, bulk ESS, and
ESS per fit second. Fits failing R-hat < 1.05 or bulk ESS >= 100 are explicitly
flagged and counted in the completed run; a finite-budget result does not assert
convergence. The minimum, maximum, diagnostic gates and continuation policy are part of the candidate specification.
The numerical and sampler comparison arms should be assessed with these
inference diagnostics as well as predictive scores. The initial fixed-budget
portfolio pilot exposed poor mixing. Native gradient rejuvenation, joint
parameter proposals and diagnostic continuation were added before the official
panel launch; [the full-budget pilot receipt](results/rough-bayesian/pilot-validation.json) records the final implementation. In this pilot, both control loss vectors are byte-identical to their saved full-panel references. Both overlay upgrades and the nominal, tight, PGAS and Student alternatives pass the inference gates for all six assets. One leverage fit and four rough-Heston fits remain flagged at the maximum retained-draw budget; their diagnostics remain explicit in the official run.

Expensive recurrences, likelihoods, both HMC adjoints and PGAS loops compile to native
code with Numba. Gaussian ESS uses linearity to avoid re-filtering rejected
ellipses. Stationary fractional factors and their eigensystems are cached;
mean-reversion initialization uses a conditional covariance construction rather
than repeated full eigendecomposition. Unchanged configurations are reused in
parameter updates. Posterior fits are cached by training bytes, source digest,
model and inference settings, with file locks to share them across portfolios.
Outer workers are bounded to 12, with one numerical thread each.

`docs/results/rough-bayesian/speed-parity.json` retains independent reference
recurrences and complete 240-path smoke loss-vector comparisons for all eight
candidates. These are numerical parity checks with a short smoke inference
budget, not official OOS scores. A full-budget canonical-origin pilot additionally
checks the unchanged production loss vector against the saved production run.

### Exact-output optimization of the direct upgrades

The ARM research backend uses a C++ Kalman likelihood loop with strict floating
point operations (`-ffp-contract=off`) and the same SciPy BLAS dot product as the
Numba reference. Covariance matrices use alternating buffers and exact equality
checks for the existing steady-state condition; no convergence tolerance or
approximate division is introduced. Other platforms retain the Numba filter.
The native module is built once into a content-addressed temporary cache using
Clang and `pybind11==3.1.0`; its source and binary hashes enter the run manifest.

Prediction retains the original BLAS reductions, terminal covariance and
eigensystem rounding. A separate native terminal-state filter omits unused
likelihood and history-path outputs while preserving those bits. Its
results are cached by asset history, selected posterior draws and implementation
contract, with individual matrix memory layouts preserved. Conditional means
and variances are likewise computed once per asset fit and forecast horizon.
Repeated posterior parameter draws share preparation without removing any
predictive paths. Constant kernel grids are cached once. The stochastic path loop
reuses exact scalars and a matrix-vector output buffer, preserving all random
calls, their order, the multiplier's arithmetic, and every posterior draw.
Both candidates share one twelve-worker queue rather than separate fixed queues.
No history, chain budget, kernel accuracy, path count or scoring rule is reduced.
The [optimization receipt](results/rough-bayesian/upgrade-speed-parity.json)
records the repeated profile review, complete retained-checkpoint parity,
full-budget chain parity and separate cold/warm prediction timings.

## Official panel

The runner reuses `tools/rough_jump_vine/run_panel.py` and its public evaluator.
It runs all 80 portfolios, 48 rolling origins plus three temporal holdouts per
portfolio: 4,080 origins per candidate, 240 paths per origin and 701,280 cells per
model, with the common history beginning 1979-12-31. It averages origin CRPS
inside each portfolio-horizon cell and then averages all cells equally. The
pairwise CRPS denominator remains n squared. No score-based early stopping or
selective origins are used.

The immutable run manifest binds source hashes, data identity, every task and
runtime versions. The restarted run selects only the two direct overlay upgrades. The six full
latent-state alternatives remain stopped and unscored. Saved production and old
rough control panels remain the comparison references. Each origin saves both
upgrade loss vectors, timings and asset inference diagnostics. Progress prints at 25%, 50%, 75% and 100%. Final results
require all denominator gates and exact production-reference parity.

```sh
OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python tools/rough_bayesian/run_panel.py \
  --output outputs/rough-bayesian-full \
  --reference outputs/parameter-mcmc-tuned-full \
  --workers 12 \
  --model-ids asset_rough_volterra_sv_eight_factor_bayesian \
    asset_rough_volterra_sv_accuracy_lift_bayesian
```

The saved full-panel controls are production CRPS **0.25246784959071183** and
old eight-factor rough CRPS **0.2511998559613309**. Their existing public evidence
and source implementations remain preserved. New full-panel scores will replace
the blank catalogue fields only after completion and validation.

## Sources

- [Gatheral, Jaisson and Rosenbaum, Volatility is rough](https://arxiv.org/abs/1410.3394).
- [Abi Jaber and El Euch, Multifactor approximation of rough volatility models](https://arxiv.org/abs/1801.10359).
- [Abi Jaber, Lifting the Heston model](https://arxiv.org/abs/1810.04868).
- [Bayer and Breneis, Weak Markovian approximations of rough Heston](https://arxiv.org/abs/2309.07023).
- [Murray, Adams and MacKay, Elliptical slice sampling](https://proceedings.mlr.press/v9/murray10a.html).
- [Lindsten, Jordan and Schön, Particle Gibbs with ancestor sampling](https://www.jmlr.org/beta/papers/v15/lindsten14a.html).
