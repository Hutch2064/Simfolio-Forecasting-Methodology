# Joint Bayesian rough volatility candidates

Nine new asset-level candidates are implemented in `tools/rough_bayesian` and
included in both public catalogues. The canonical catalogue now has 208 entries;
the broader active research catalogue has 397. The eight-factor Bayesian upgrade
and dynamic-resolution candidate completed their full panels; the fixed-resolution lift retains its stopped partial
results. Other aggregate scores remain blank until their complete panels finish.
The existing production Frontier
and the completed eight-factor rough model are included as unchanged controls.

| Candidate | Question | Volatility inference |
| --- | --- | --- |
| `asset_rough_volterra_sv_eight_factor_bayesian` | Does parameter uncertainty improve the winning rough overlay? | Unchanged eight-factor kernel; collapsed Gaussian filtering + parameter MCMC |
| `asset_rough_volterra_sv_accuracy_lift_bayesian` | Does an accuracy-controlled kernel improve that winner further? | Tempered fractional Volterra covariance; adaptive lift + parameter MCMC |
| `asset_rough_volterra_sv_dynamic_lift_bayesian` | Can parameter-specific resolution approximate rough covariance efficiently? | Positive Gaussian spectral quadrature; every positive integer order is eligible; complete daily-lag error control + parameter MCMC |
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
runtime versions. The eight-factor upgrade and the separate dynamic-resolution candidate have
completed their full panels. The fixed-resolution lift was stopped and retains
only its completed portfolio scores. The six full latent-state alternatives
remain stopped and unscored. Saved control panels use matching portfolios,
origins, horizons and scoring weights.

```sh
OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python tools/rough_bayesian/run_panel.py \
  --output outputs/dynamic-rough-full \
  --reference outputs/parameter-mcmc-tuned-full \
  --workers 12 \
  --model-ids asset_rough_volterra_sv_dynamic_lift_bayesian
```

| Full-panel model | Exact empirical CRPS |
|---|---:|
| Dynamic-resolution rough | 0.25074679212444156 |
| Previous eight-factor rough (MAP rough parameters) | 0.2511998559613309 |
| Eight-factor Bayesian upgrade | 0.2522081419138852 |
| Production Frontier | 0.25246784959071183 |
| Production Default, matching saved cell replay | 0.292821259593872 |

The [dynamic results](results/rough-bayesian/dynamic-panel.json) and
[independent audit](results/rough-bayesian/dynamic-full-audit.json) preserve all
80 portfolio scores, exact reconstruction of 4,080 checkpoint vectors and
701,280 cells, unchanged source closure and daily-lag approximation certificates.
Dynamic CRPS is 0.18036% lower than previous rough and 0.68169% lower than
production Frontier. It wins 40 of 80 portfolios against previous rough and
43 against production. The descriptive paired-portfolio bootstrap 90% interval
for its mean difference versus production is [-0.00337581, 0.000024319].
Six of 2,549 unique asset fits missed the sampler convergence targets at the
maximum draw budget; their diagnostics remain visible in the results and audit.
The score validation does not establish complete posterior convergence or
production suitability. The archived Default full-panel score is
0.2928212617494334; its matching replay differs by 2.16e-9.

## Sources

- [Gatheral, Jaisson and Rosenbaum, Volatility is rough](https://arxiv.org/abs/1410.3394).
- [Abi Jaber and El Euch, Multifactor approximation of rough volatility models](https://arxiv.org/abs/1801.10359).
- [Abi Jaber, Lifting the Heston model](https://arxiv.org/abs/1810.04868).
- [Bayer and Breneis, Weak Markovian approximations of rough Heston](https://arxiv.org/abs/2309.07023).
- [Murray, Adams and MacKay, Elliptical slice sampling](https://proceedings.mlr.press/v9/murray10a.html).
- [Lindsten, Jordan and Schön, Particle Gibbs with ancestor sampling](https://www.jmlr.org/beta/papers/v15/lindsten14a.html).

## Dynamic resolution of the tempered rough covariance

The dynamic candidate preserves the production mean, variance anchor, empirical
return innovations, cross-asset dependence, calendar, costs and CRPS scorer. It
changes the finite-dimensional approximation of the asset's stationary Gaussian
rough-volatility covariance. It remains a three-parameter collapsed Bayesian
model with the original priors and two-chain sampling budgets.

For `alpha = H + 1/2`, the normalized covariance of the tempered Volterra kernel
`exp(-kappa*t)*t**(alpha-1)/Gamma(alpha)` has the exact representation

```text
C(t) = E[exp(-kappa*(1+Z)/(1-Z)*t)], Z ~ Beta(1-alpha, 2*H).
```

This follows by integrating one of the two Laplace measures in the stationary
kernel convolution, then substituting `Z = r/(r+2*kappa)`. Positive spectral
quadrature weights produce independent OU factors with that approximate scalar
Gaussian covariance. This is a numerical representation of the same stationary
Gaussian process, rather than a new economic factor model. The covariance and
likelihood are checked independently against Beta integration and a dense
Gaussian covariance matrix.

The positive-rate integral is transformed to logarithmic rates and evaluated by
Gauss-Jacobi quadrature, matching its integrable endpoint singularity. Rates
above `-log(tolerance/2)` are aggregated into independent white variance. Their
omitted positive daily-lag covariance is bounded by `tolerance/2`; total zero-lag
variance is retained analytically. The sole approximation tolerance remains
`0.001` in normalized absolute autocorrelation units.

For each `(H,kappa)` proposal, orders `1,2,3,...` are eligible, with no preset
factor list or fixed maximum. The first passing order is selected. The factor
count includes the retained white-variance component. The error check covers
all integer lags from one through `training_days + requested_horizon_days - 1`.
Monotonic endpoint enclosures and the positive-mixture curvature bound
`C''(t) <= 4*exp(-2)/t**2` certify intervals; unresolved intervals are bisected
until certified or their daily endpoints are exhausted. Reference covariance
evaluations are reused across proposed quadrature orders. Geometry and already
selected parameter settings are cached without quantizing parameters.

The selection rule is deterministic and frozen before the run. Because latent
states are integrated out, varying the state dimension does not change the
three-dimensional parameter space or require reversible-jump MCMC. MH evaluates
the corresponding deterministic approximate likelihood at each proposal. The
requested horizon is known at the origin and enters the fit cache contract;
future returns and OOS scores never determine resolution. Covariance tolerance
does not assert a bound on posterior or CRPS error. The candidate has its own
identity and scores; the earlier fixed-resolution candidate's partial results
are preserved separately.

The approximation family is supported by [Abi Jaber and El Euch's multifactor
construction](https://arxiv.org/abs/1801.10359) and [Bayer and Breneis's
parameter- and horizon-dependent quadrature error analysis](https://www.wias-berlin.de/people/bayerc/files/breneis_21.pdf).
The Beta representation and daily-lag interval certificate above are the
implementation-specific construction tested here; the cited papers do not
establish its forecasting score or runtime.

## Matched 100-year model-computation diagnostic

The runtime comparison retained six assets, 11,657 daily history observations,
240 paths, a 25,200-day forecast and the same investor-policy rejoin. It used
the current production asset-source implementation at engine revision
`5041ae65ad5cfd8f25e69ce0d1da6809fa7e7f23`, with warmed numerical compilers on
the same Apple M5 Pro. Production used its automatic asset/chain workers; the
rough implementation retained its current sequential asset fitting.

| Model | Fresh fits | Cached fits/prepared calculations, two repetitions |
|---|---:|---:|
| Production Frontier | 1.3808 s | 1.0197–1.0319 s |
| Dynamic rough | 96.5301 s | 3.6508–3.6864 s |

The rough-parameter fitting contribution was 87.3014 seconds. Each model's
complete path array was byte-identical between its fresh and cached executions.
These are local model-computation timings, excluding data retrieval, queueing,
flows, taxes, metrics, serialization, transfer and browser rendering; they do
not establish deployed click-to-visible times. The exact case, source pins,
timing components and output hashes are retained with the existing
[speed-validation receipt](results/rough-bayesian/dynamic-speed-parity.json).

## Output-preserving execution optimization after panel completion

The winning dynamic candidate retains the same priors, two-chain budgets,
convergence extensions, parameter draws, tolerance, eligible factor orders,
production anchor, dependence, random streams and scorer. Its factor-resolution
calculation now reuses Jacobi recurrence coefficients between orders. The
pinned SciPy root refinement and weight normalization are unchanged. Independent
asset fits can use spawn processes inside the existing `state_workers` budget;
a cached fitted bundle avoids process startup on repeat requests. Panel workers
already parallelized across origins and retain their existing CPU allocation.
The independent-OU forecast square root uses its exact sparse structure rather
than a dense matrix-vector call on every simulated day.

For the six-asset 100-year case above, with six asset-fit workers, a fresh run
took **35.3260 seconds**, compared with **96.5301 seconds** in the retained
serial reference. Two cached repetitions took **2.7512 and 2.7586 seconds**,
compared with **3.6508 and 3.6864 seconds**. Every float64 forecast path was
byte-identical to the pre-optimization reference. The reduced-budget 63-horizon
smoke also preserved every loss vector exactly, CRPS **0.03516121238116601**,
in serial and process execution. Full-budget short- and long-history chain
traces/RNG states and 6,400 complete quadrature factor arrays were byte-identical.

The source hashes, case and timing components are appended to the existing
[speed-validation receipt](results/rough-bayesian/dynamic-speed-parity.json).
Short reduced-budget fits can be slower when process startup dominates; their
checks use a single lane for timing comparisons. These measurements concern
local model computation. The validated full-panel score remains the retained
execution result; the full panel was not rerun after this optimization.

## New standalone and corrected-likelihood experiments

These three adapters are implemented for research testing. They have no complete
80-portfolio scores and are not added to the validated catalogue during smoke
validation. The completed dynamic rough result remains the incumbent. All fits
are per asset, with the same canonical data, rolling origins, return copula,
policy rebalancing, costs, daily clipping, simulation count and CRPS scorer.

| Research adapter | Volatility model and inference | Conventional 16-draw compression |
| --- | --- | --- |
| `asset_rough_volterra_sv_dynamic_standalone_bayesian` | The dynamic tempered rough Volterra process supplies the full volatility scale. Two collapsed parameter chains jointly infer H, log kappa, log eta and level under the existing Gaussian log-square quasi-likelihood. Each path uses its parameter draw, conditional terminal rough state and rough-filtered empirical innovation nodes. | Removed; no conventional SV fit is called. |
| `asset_rough_volterra_sv_dynamic_exact_mixture_overlay` | Retains the incumbent's production mean/variance curves and empirical innovations. Joint parameter/history MCMC calibrates its rough process to the Gaussian return likelihood, with the rough level fixed as in the incumbent. | Retained in the conventional anchor; the separate rough sampler has adaptive precision stopping. |
| `asset_rough_volterra_sv_dynamic_exact_mixture_standalone` | Rough volatility supplies the entire scale; H, log kappa, log eta, level and the full latent log-variance history are inferred jointly. Gaussian predictive shocks match its Gaussian calibration likelihood. | Removed; no conventional SV fit is called. |

Each standalone arm has one rough posterior sampled by two chains. The overlay
retains a conventional two-chain fit and a separate rough two-chain fit. These
are separate calibrations, not nested MCMC or a joint conventional/rough posterior.

The standalone arms use the historical sample mean directly in predictive log
returns. The production overlay's numerical mean curve depends on its volatility
calculation. Removing that anchor therefore changes the numerical mean forecast
as well as the variance forecast, while preserving the historical mean estimate.
The Gaussian standalone arm also changes the empirical innovation distribution;
its score cannot be attributed to likelihood accuracy alone. No leverage, jumps,
HMM, vine copula or separate portfolio-level volatility model is introduced.

For the corrected arms, let e be 100 times the asset return minus its training
mean. The likelihood is `e_t | h_t ~ Normal(0, exp(h_t))`. Positive squared
residuals are logged without winsorization or an offset floor. Exactly zero
residuals remain in the raw Gaussian return likelihood. The published Omori
10-normal log-chi-square mixture supplies auxiliary indicators and a Gaussian
proposal only. Its fixed ten components are published numerical coefficients,
not rough factors or a selected number of posterior draws. A final joint
Metropolis-Hastings correction targets the actual Gaussian return likelihood of
the controlled numerical rough lift, rather than leaving the mixture error in
the target. The overlay subsequently retains empirical predictive shocks: this
is Gaussian calibration followed by a semiparametric overlay forecast, not a
claim that its empirical predictive law has a Gaussian likelihood.

Given the indicators, rough states are integrated out during a reversible
parameter Metropolis proposal. In the standalone corrected arm, volatility
level is also integrated analytically with its truncated Gaussian prior; its
conditional level draw is direct rather than a random walk. The whole scalar
volatility history is then sampled conditionally by a simulation smoother. The joint proposal is accepted
with the exact-to-mixture observation-density ratio. Working with the scalar
history keeps its dimension equal to the observed history length while numerical
OU factor counts change with H and kappa; reversible-jump parameter sampling is
not needed. This is a research implementation assembled from published mixture,
smoothing and correction methods, not a reproduction of a published RFSV model.

Rough covariance uses the incumbent's positive spectral quadrature. Every
positive integer order is eligible, and the first passing order is selected
for each parameter value, with maximum normalized autocorrelation error at most
0.001 over every daily lag through history plus forecast horizon minus one.
This remains a numerical approximation of a tempered rough Volterra process,
not exact continuous-time RFSV or rough Heston. Eta is stationary log-volatility
standard deviation. Priors retain H uniform on (0.03, 0.49), log kappa normal
with center log(1/63), SD 2 and bounds log(1/2520) to log(0.5), and log eta normal
with center log(0.7), SD 1.5 and bounds log(0.05) to log(3). In standalone arms,
level has a training-centered normal prior with SD 4 and training log-square
1%/99% quantile bounds expanded by 4. This is data-centered regularization;
adaptive numerical resolution does not eliminate prior or accuracy choices.

The corrected samplers warm up two chains for 2,048 iterations each, freeze
proposal adaptation, and check retained samples first at 256 per chain. They
double retained samples until all monitored quantities have rank split Rhat
below 1.01, bulk ESS at least 400, and batch-means estimated 95% Monte Carlo
halfwidth at most 10% of posterior SD. Monitored quantities cover parameters,
log likelihood plus parameter prior, latent mean/terminal log variance and
stationary/terminal variance. A 65,536-per-chain resource ceiling leaves a
failed precision flag if unmet. These thresholds are declared numerical accuracy
policies, not universally mandated academic constants or CRPS error guarantees.
The quasi-likelihood standalone retains the incumbent's two-chain 2,048 warmup,
8,192 initial retained samples, Rhat below 1.05, ESS at least 100 and the same
ceiling. Neither standalone arm compresses its posterior into 16 representatives:
actual parameter draws and conditional latent uncertainty drive predictive paths.

The corrected arms keep a uniform reservoir of joint parameter/history draws
per chain, select joint predictive draws across chains, then retain only their
conditional terminal state distributions. This avoids storing every historical
path from every retained iteration. Compiled strict-floating-point Kalman and
mixture kernels reuse the same measurement geometry for marginal likelihood and
simulation smoothing. Work is O(T*k^2), with O(T*k) smoother storage; no dense
T-by-T covariance factorization is used. Source-addressed asset fit caches,
predictive preparation reuse and the existing fixed CPU budget are preserved.
Direct conditional level sampling was also tested in the Gaussian quasi-likelihood
standalone arm. It was slower and delivered fewer effective samples per second
on the long-history screen, so that arm retains its faster four-parameter native
Kalman implementation.

The conventional 16 representatives were explored separately against all 4,096
retained conventional draws on two training assets. At 16 representatives,
maximum relative forecast-SD discrepancies were 1.831% and 1.082%, with
standardized empirical-node discrepancies of 0.05782 and 0.04509. Increasing
the evenly spaced representative count did not improve every error monotonically.
A defensible adaptive compression would need to control both forecast moments
and innovation quantiles; relative Monte Carlo stopping for posterior estimation
does not itself certify that compression. The production control remains
unchanged, and the standalone experiments remove this stage entirely.

Independent checks cover dense Gaussian marginal likelihoods and smoothing,
conditional latent moments, an exact scalar posterior, a joint parameter/history
posterior calculated by numerical quadrature, zero residuals, resumable chains
and reservoirs, and serial/parallel/cached score-vector parity. The retained
smoke and local benchmark evidence is in
[`standalone-mixture-validation.json`](results/rough-bayesian/standalone-mixture-validation.json).
Smoke scores are not canonical full-panel scores.

Academic bases: [Kim, Shephard and Chib](https://shephard.scholars.harvard.edu/publications/stochastic-volatility-likelihood-inference-and-comparison-arch-models),
[Omori mixture coefficients and MH correction](https://arxiv.org/html/2404.13986v2),
[Markovian multifactor approximation](https://arxiv.org/abs/1801.10359), and
[relative fixed-width Monte Carlo stopping](https://arxiv.org/abs/1303.0238).

## Matched conventional and heavy-tailed Bayesian SV counterparts

These research adapters complete a two-by-two comparison around the existing
standalone Gaussian rough candidate. They use the **existing canonical runner**,
not another panel: identical snapshot, common 1979 start, 80 portfolios, 48
rolling origins plus three temporal holdouts each, 240 simulations, all eligible
daily horizons, 4,080 origin tasks and 701,280 equally weighted portfolio/horizon
cells. Production is rerun and every control loss vector is checked byte-for-byte
against its production-promotion reference. All new inference is per asset.

| Adapter | Single volatility process | Raw return likelihood | Estimated parameters |
| --- | --- | --- | --- |
| `asset_bayesian_ar1_sv_raw_gaussian` | Stationary Gaussian AR(1) log variance | Gaussian | level, persistence, stationary log-volatility SD |
| `asset_bayesian_ar1_sv_raw_student` | The same AR(1) process | Variance-standardized Student-t | the same three plus degrees of freedom |
| `asset_bayesian_dynamic_rough_sv_raw_student` | The incumbent's dynamically resolved tempered rough covariance | Variance-standardized Student-t | level, H, kappa, eta, degrees of freedom |

The Gaussian rough cell is the already implemented
`asset_rough_volterra_sv_dynamic_exact_mixture_standalone`. The three new models
retain its historical sample-mean forecast, Gaussian dependence uniforms,
calendar, costs, rebalancing, clipping and scorer. The Student inverse CDF acts
on those same dependence uniforms, with unit conditional variance. No production
SV anchor, separate fixed-window multiscale component, 16 representatives, jumps,
leverage, HMM or vine is added. Relative to production, the historical mean
estimator is preserved but the numerical mean curve and conditional shock law
are different. The completed rough hybrid remains an overall benchmark, not a
claim of a pure volatility ablation against production.

In AR(1), h[t+1] = level + phi*(h[t]-level) + sigma*Z, with |phi|<1 and
sigma = eta*sqrt(1-phi**2). The initial distribution is stationary. We sample
atanh(phi) with (phi+1)/2 ~ Beta(20,1.5), including its transformation Jacobian.
Log eta has the same normal center log(.7), SD 1.5 and .05-to-3 bounds as the
rough comparison. Level retains the existing training-centered, truncated normal
prior. These are declared prior choices, not universal academic constants.
Rough priors and the .001 complete-daily-lag covariance accuracy policy remain
unchanged. One AR(1) state is its structural specification, not a quadrature cap.

Student degrees of freedom are inferred via log(nu-2), with nu-2 ~ Exponential
(rate .1) and its Jacobian. There is no fixed tail thickness or finite upper cap.
For each observation, tau|nu ~ Gamma(nu/2,nu/2) and
e|h,tau,nu ~ Normal(0,exp(h)*(nu-2)/(nu*tau)). The sampler first proposes nu
against the exact Student likelihood with tau marginalized, then refreshes all
tau from their exact Gamma conditionals **before** the volatility block. That
ordering is necessary for the partially collapsed update to preserve the joint
posterior. Adjusted log-square observations supply the published normal-mixture
proposal; the final exact observation-density correction remains in place.
Both calibration and future shocks follow the same standardized Student law.
Zero returns are retained without a log offset or winsorization.

All parameters and the whole latent volatility history are inferred jointly.
Conditional level is integrated in hyperparameter proposals and drawn directly.
Two chains use the existing retained-draw precision checks and resource ceiling;
each future path uses an actual joint parameter/history draw. This is an
academically grounded extension of the current research implementation, not a
reproduction of a published canonical RFSV model. New full-panel scores remain
blank until evaluated; validated catalogue additions await evidence.

The conventional counterparts retain the 65,536-per-chain ceiling. The rough
Student candidate allows 131,072 per chain because its initial six-asset pilot
had one fit with minimum ESS 352.6 at 65,536, despite passing Rhat and relative
precision checks. The ESS requirement remains 400; no gate is relaxed. This
ceiling limits resources, not model parameters or the adaptive stopping length.

The optimized implementation compiles strict-floating-point state updates,
reuses measurement geometry and avoids per-observation temporary state arrays.
It evaluates the proposed path's correction without drawing unused mixture
indicators. Terminal preparation streams the filter rather than storing unused
whitened histories and gains. Random-call order, full latent histories, priors,
posterior draws and accuracy gates are preserved. Complete sampler traces and
full pilot loss vectors are byte-identical to the unoptimized implementation;
the production control is byte-identical to its saved promotion vectors.
All 18 pilot asset fits and three 10,213-observation asset checks passed the
declared inference gates. The retained
[validation receipt](results/rough-bayesian/coherent-sv-validation.json) records
panel identity, independent likelihood/posterior checks, parity and profiling.
Interleaved local long-history chain screens measured roughly one-third less
sampling time; this is not a full-panel or deployed runtime claim.

References: [Hosszejni and Kastner, conventional and Student Bayesian SV with
compiled implementations](https://www.jstatsoft.org/article/view/v100i12),
[Abanto-Valle et al., Bayesian heavy-tailed SV and scale augmentation](https://pmc.ncbi.nlm.nih.gov/articles/PMC2923593/),
and the mixture/correction and precision references above.

## MAP predecessor with a dynamic rough overlay

`asset_map_predecessor_dynamic_rough_map` retains the previous production
Frontier's asset-level MAP fit, historical mean, four multiscale half-lives,
filtered empirical innovations, Gaussian dependence and policy rejoin. It
adds the same dynamically resolved tempered rough covariance as the Bayesian
research winner, estimated at a joint MAP point for H, log kappa and log eta.
Historical rough states are marginalized by the Gaussian filter; conditional
terminal states and future volatility innovations are simulated. Both parameter
fits are MAP estimates, with no parameter MCMC or sixteen-draw compression.
This is a Gaussian log-square quasi-likelihood implementation. Its numerical
approximation is accuracy controlled; it is not a raw-return posterior fit.

The predecessor's fixed half-lives remain 5, 21, 63 and 252 trading days,
shortened for insufficient history. Rough factor count is the first passing
positive integer quadrature order at covariance error tolerance 0.001 over
the full training-plus-forecast daily lag domain. Priors and parameter bounds
are unchanged from the dynamic Bayesian overlay. These settings are recorded
in the canonical statistical specification. Future rough multipliers preserve
conditional second moments before the existing return clipping.

| Model | Full canonical empirical CRPS |
| --- | ---: |
| Dynamic Bayesian rough research winner | 0.25074679212444156 |
| MAP predecessor + dynamic rough | 0.2518812750312293 |
| Current production parameter-MCMC Frontier | 0.25246784959071183 |
| MAP predecessor without rough | 0.25439860867855635 |
| Production default naive | 0.292821259593872 |

Only the new candidate ran. Saved controls matched all task identities and
portfolio/horizon keys. All 80 portfolios, 4,080 origins, 240 paths per origin
and 701,280 cells completed. A separate reconstruction from every origin loss
vector reproduced all published cell vectors exactly. The new model wins on
41 of 80 portfolios against production and 50 against its MAP predecessor.
Its aggregate score improves on production by 0.2323%, while remaining
0.4524% worse than the dynamic Bayesian winner. These are descriptive panel
comparisons, not evidence of a universal forecasting improvement.

The twelve-worker run completed in 136.825 seconds, with 1,569.243 total worker
seconds (0.385 seconds per origin on average). Saved production and dynamic
Bayesian runs recorded 3,386.493 and 42,590.790 worker seconds respectively.
Their ratios are historical research timing comparisons, not controlled
hardware/cache benchmarks or deployed website timings.

Optimization reused exact compiled marginal mapping and policy rejoin, removed
a redundant sort and duplicated volatility-moment calculation, and cached
asset/lag fits across portfolios. All five reference smoke loss vectors remained
byte-identical. The longest smoke changed from 2.009 to 1.221 seconds with fresh
fits and from 1.393 to 0.734 seconds with cached fits, excluding compiler startup.
All 2,549 distinct asset/lag fits converged; 52 needed an extended L-BFGS-B
line search with the same likelihood, priors and accuracy tolerances. No Powell
fallback was needed. Selected rough factor counts ranged from 4 to 10.

The [full result, 80 portfolio scores, source hashes and audit](results/rough-bayesian/map-predecessor-dynamic-panel.json)
and [cell-loss evidence](results/rough-bayesian/map-predecessor-dynamic-cells.npz)
are retained. The candidate is canonical index 193 and a validated master
catalogue extension. It is research code; production behavior is unchanged.

```bash
python tools/dynamic_rough_inference/run_panel.py \
  --output /path/to/run-output \
  --reference /path/to/parameter-mcmc-tuned-full \
  --workers 12
```

Use the existing panel runner's frozen data setup and the pinned numerical
dependencies. `screen.py` exercises bounded canonical origins;
`progress.py` reports matched cumulative scores using saved controls.


## Volatility-only research against the MAP multiscale dynamic rough baseline

The frozen baseline is canonical M193, `asset_map_predecessor_dynamic_rough_map`,
with full-panel CRPS 0.2518812750312293. The following six completed experiments
retain its mean, empirical innovations, Gaussian dependence, random streams,
calendar, rebalancing, transaction costs, clipping, simulations and scorer.
Every volatility fit is asset-level. The evidence and complete paired cells are
in `docs/results/rough-bayesian/volatility-frontier-ablations.json` and the adjacent
`volatility-frontier-ablations-cells.npz`. The audit independently reconstructs
all 4,080 origin vectors into 701,280 cells for each candidate, byte-for-byte.
These are research candidates; the frozen baseline has not been replaced.

| Volatility change | Full CRPS | Improvement over M193 | Summed worker seconds |
| --- | ---: | ---: | ---: |
| Baseline M193 | 0.2518812750312293 | — | 1569.24 |
| Rough fit on causal conventional-SV residuals | 0.25281487701385613 | -0.371% | 1407.43 |
| Empirical innovation log-square measurement variance | 0.2509457012962614 | 0.371% | 1489.90 |
| Causal residual fit with conditional rough forecasts | 0.25238690162994976 | -0.201% | 1408.45 |
| Conditional residual fit plus empirical measurement variance | 0.2497332684165023 | 0.853% | 1332.11 |
| Same combination with untrimmed rough observations | 0.2498921919286796 | 0.790% | 1343.75 |
| Jointly estimated rough measurement variance | 0.2532711708690856 | -0.552% | 2106.03 |

The causal residual arms subtract the conventional model's one-step prediction
from the rough observation series. Parameters of that conventional offset are
estimated once using only the origin's training history. This is two-stage
plug-in estimation, not joint Bayesian inference or an exactly orthogonal
multiscale decomposition. The conventional forecast and innovation pool remain
unchanged.

The empirical-noise arms use the variance of the same log-square transform
applied to the existing fitted innovation pool. Gaussian filtering is evaluated
in scaled units to implement this variance; terminal states and covariances
are transformed back before simulation. This corrects the fixed Gaussian
log-chi-square noise moment assumption without changing the return law. It
remains a Gaussian quasi likelihood, not an exact empirical-return likelihood.

The conditional arms use multiplier `exp(.5*latent-.25*stationary_variance)`.
Its stationary expected square is one, but its conditional expected square is
`exp(conditional_mean+.5*(conditional_variance-stationary_variance))`. They retain
the terminal rough state's forecast information. The baseline instead subtracts
the conditional mean and normalizes conditional variance, preserving its daily
variance anchor. This is an explicitly tested volatility-model change.

The untrimmed arm removes quantile flooring and tail winsorization only from the
rough log-square observation and measurement-noise estimate. Machine tiny protects
log(0). Its conventional fit and causal offset still use the original preprocessing.
Both conditional experiments preserve the dynamic kernel's 0.001 daily-lag
covariance certificate and the original rough priors. No posterior MCMC or fixed
posterior representative count is introduced.

These variants combine established Gaussian SV quasi-likelihood, state filtering,
and accuracy-controlled rough covariance approximation in custom research
adapters. They are not presented as canonical implementations of a named
published full Bayesian rough-return model. Relevant primary foundations are
[SV review](https://www.nuffield.ox.ac.uk/economics/papers/2005/w17/palgrave.pdf)
and [Abi Jaber and El Euch's multifactor approximation](https://arxiv.org/abs/1801.10359).
The observed speed differences are local research timings, not website latency
proof. None yet satisfies the requested minimum 1% score breakthrough; the best
recorded worker-time reduction is about 15%, awaiting a matched timing comparison.


Canonical M194–M199 retain all six completed volatility ablations in both catalogues.
The joint-noise arm estimates log measurement variance as an unpenalized Gaussian
quasi-likelihood nuisance parameter alongside the three rough MAP parameters.
It includes the observation-scaling likelihood Jacobian. Bounds of initial
empirical log variance plus/minus 20 are numerical guards. Its forecast uses the
fitted noise variance in the terminal filter. It was slower and less accurate,
so the additional fitted parameter is rejected for this research direction.


The subsequent spectral-inference tests retain M197’s causal conventional-SV residual observations, empirical measurement variance, and stationary normalization. Only rough-parameter estimation changes. M200 uses the debiased Whittle pseudolikelihood on demeaned observations, omitting frequency zero. M201 uses first differences and includes every frequency, with the induced MA(1) measurement-noise covariance treated explicitly. Real zero/Nyquist ordinates have half weight; conjugate pairs have unit weight, restoring the Gaussian likelihood scale before adding the unchanged priors. Both use exact finite-sample expected periodograms of the dynamically resolved covariance, without frequency subsampling. Terminal state filtering and future path generation remain the undifferenced conditional Gaussian model. This is approximate frequency-domain inference, not exact likelihood or full Bayesian parameter integration. See [Sykulski et al. (2019), sections 4.1–4.2](https://doi.org/10.1093/biomet/asy071).

| Candidate | Full canonical CRPS | Improvement versus M193 | Recorded 12-worker wall time |
|---|---:|---:|---:|
| M200: debiased Whittle | 0.249657155081818 | 0.883% | 119.12 s |
| M201: differenced debiased Whittle | 0.24972187973628046 | 0.857% | 103.78 s |

All 8,160 origin vectors independently reconstruct the two 701,280-cell score vectors byte-for-byte; source revisions, cell arrays, fit diagnostics, and portfolio results are bound in `docs/results/rough-bayesian/spectral-rough-panels.json`. Local recorded worker times compare against M193’s 136.82-second panel and do not establish website latency. Neither candidate reaches a 1% score improvement versus M193.


M202 tests a Gaussian Gamma-mixed OU log-volatility overlay in place of the tempered rough covariance, keeping M197's conventional multiscale backbone, empirical measurement variance, residual observations, and stationary normalization. Its covariance is `scale**2*(1+kappa*lag)**(-shape)`, corresponding to Gamma-distributed stationary variance rates and the Gaussian specialization of [Barndorff-Nielsen and Stelzer (2011), example 3.4](https://arxiv.org/abs/1101.0068). Shape, timescale and amplitude are learned per asset; log-shape has a weak truncated Gaussian prior with mean zero, standard deviation 1.5 and shape bounds [.03,10]. The other priors remain unchanged. This Gaussian log-volatility construction differs from the positive Levy-driven matrix-volatility model.

The preliminary exact Gaussian quasi-likelihood implementation was screened out for runtime (fresh smokes 0.28-5.66 seconds). The full M202 candidate instead fits the exact analytic covariance using debiased Whittle inference, resolving numerical OU factors only for terminal filtering and prediction. Five fresh/cached smoke vectors matched exactly, and all 4,080 full-panel origin vectors reconstruct the retained score cells byte-for-byte. Full CRPS is **0.25087597152824515**, 0.399% better than M193 but worse than M197/M200/M201. Recorded wall time is **159.10 seconds**: inference is cheap, but larger predictive factor sets increase path-generation cost. This direction is rejected for the requested speed/score breakthrough. Evidence is retained in `docs/results/rough-bayesian/gamma-supou-panel.json`.


M203 keeps M197's rough covariance and return forecasting backbone but analytically integrates the constant rough-observation level under a flat prior. The covariance parameters maximize the restricted Gaussian quasi likelihood plus unchanged priors; their full posterior is not integrated. A single compiled filter computes the GLS level, its posterior variance, and the terminal latent-state posterior, including the uncertainty from that nuisance level. Four independent dense Gaussian reference tests verify likelihood, level, covariance and shift invariance. All five fresh/cached smoke vectors agree byte-for-byte. Full canonical CRPS is **0.2504647898614585**, a **0.562%** improvement versus M193; wall time is **148.75 seconds**, above M193's recorded 136.82 seconds. The full 4,080-vector reconstruction and parameter-source audit are bound in `docs/results/rough-bayesian/reml-rough-panel.json`. It does not beat M197/M200/M201 and does not satisfy the breakthrough target. Restricted likelihood follows [Patterson and Thompson (1971)](https://doi.org/10.1093/biomet/58.3.545), applied here to the existing Gaussian log-square quasi observation model. The return-mean estimator remains unchanged.


M204 keeps M200 covariance inference exactly and estimates the rough residual observation level by Gaussian GLS at terminal conditioning. Its covariance is conditional on the level point, with no nuisance-level variance addition. Five dense-reference/inheritance tests and five fresh/cached smoke comparisons passed. All 4,080 origin vectors reconstruct its saved 701,280 score cells byte-for-byte. Full CRPS is **0.24986970449578597**, a **0.799%** improvement versus M193; recorded wall time is **122.04 seconds**. It trails M200. GLS follows the constant regression formula in [Zimmermann, equation (2.10)](https://doi.org/10.1155/2010/494070); the covariance fit remains the existing frequency-domain quasi likelihood. Evidence: `docs/results/rough-bayesian/whittle-gls-rough-panel.json`.

M205 keeps M200 predictions and numerical rules, replacing its covariance MAP target with bounded unpenalized debiased Whittle quasi maximum likelihood. Hurst already has a uniform interior prior; this removes only Gaussian log-timescale/amplitude penalties. Original support bounds remain. Three target/isolation tests and five exact fresh/cached smoke comparisons passed. The full raw-vector reconstruction passed. CRPS is **0.24998100070682613**, a **0.754%** improvement versus M193, with recorded wall time **119.60 seconds**. It also trails M200, so removing regularization does not establish a better model. Evidence: `docs/results/rough-bayesian/whittle-mle-rough-panel.json`. Neither is a full Bayesian raw-return fit; the return-mean model, empirical innovations, dependence, seeds and scoring remain unchanged.


M206 retains the M200 covariance MAP fit and integrates the rough observation intercept under a flat prior, adding its uncertainty to the terminal-state covariance. Its audited full-panel CRPS is **0.24984428792635055** (0.809% better than M193), with isolated fresh-fit wall time **111.18 seconds**. M207 fits the analytic tempered fractional covariance directly in the expected-periodogram objective, resolving the same dynamic OU lift only for prediction. Its audited CRPS is **0.2495833333032549** (0.912% better than M193), with fresh-fit wall time **136.45 seconds**. The matched fresh-fit baseline took **125.73 seconds**. Neither meets the revised research goal. Both retain the return mean, empirical innovations, dependence, RNG, calendar and scoring. Evidence: `docs/results/rough-bayesian/whittle-integrated-level-rough-panel.json` and `docs/results/rough-bayesian/exact-covariance-whittle-rough-panel.json`.


M208 removes the arbitrary Hurst support restrictions 0.03/0.49, retaining the theoretical strict rough domain 0 < H < 0.5. It evaluates the same covariance through the equivalent Bessel-K form (DLMF 13.6.10), while preserving dynamic prediction-lift accuracy rules and the return mean. Twelve independent mathematical tests and five byte-exact fresh/cached smokes passed. Full-panel CRPS is **0.2505737023816167**, a **0.519%** improvement over M193, with isolated fresh-fit wall time **123.60 seconds**. It trails M207; the more permissive support does not improve the score leader. Evidence: `docs/results/rough-bayesian/full-hurst-domain-rough-panel.json`.


Two subsequent candidates were screened without a full-panel score. The M207 variance-targeting adaptation preserves return mean curves and scales volatility to the asset's training sample variance; two independent Lyapunov/long-horizon tests passed, but smoke CRPS values were 0.0368199994475318, 0.04674503819673142, 0.08813419690821843, 0.01483129965098614 and 0.1512294751560598. Four of five were worse than M193, so no official full panel was launched. Implementation: `tools/variance_targeted_rough/models.py`.

The constant historical log-return mean candidate retains M207 volatility and removes volatility-scaled Sharpe drift. Four paired 50/100-year mean-growth tests passed. Smoke CRPS values were 0.035411045651616804, 0.057545773842543314, 0.11817038720242261, 0.014995257683460363 and 0.3627914454779456. The long-horizon smoke worsened sharply, so no official full-panel score is claimed. Implementation: `tools/historical_log_mean_rough/models.py`. Both candidates reproduced all five cached smoke loss vectors byte-for-byte; neither is included among fully validated full-panel catalogue entries.

## Analytic covariance with differenced debiased Whittle (M209)

M209 combines M207's analytic stationary rough covariance with M201's first-difference observation likelihood. The expected finite-sample periodogram is calculated directly from the differenced covariance; differenced observation noise is treated as MA(1), rather than independent noise. Zero and Nyquist frequencies receive half weight. This follows the differenced debiased Whittle construction in [Sykulski et al. (2019), section 4.2](https://www.jmlilly.net/papers/sykulski19-biometrika.pdf). Covariance parameters use the original MAP priors and bounds. Dynamic factor resolution is performed only after fitting, for the unchanged undifferenced terminal-state filter and forecast. It remains Gaussian log-square quasi likelihood, rather than exact raw-return Bayesian inference.

All 4,080 origin vectors independently reconstruct the 701,280 retained cells byte-for-byte. Every asset fit reported convergence. Full-panel CRPS is **0.2493515254197939**, 1.004% below baseline M193 and 0.093% below previous leader M207; 38 of 80 portfolio-average scores improved against baseline. The return mean, empirical innovations, dependence, calendar, simulations and scoring remain unchanged.

The isolated fresh-cache run took **99.28025445801904 seconds**, compared with the controlled baseline run's **125.72699987504166 seconds**, a 21.04% reduction. These are local research worker timings from one run per model, not deployed website latency. [The audited receipt](results/rough-bayesian/analytic-differenced-rough-panel.json) binds numerical sources, per-portfolio scores and complete paired cells.

## Relaxed Hurst lower support (M210)

M210 changes only the Hurst support lower bound in M209, from 0.03 to 0.01; the 0.49 upper bound, other priors and estimator remain unchanged. All five smoke repeats are byte-identical. Full-panel CRPS is **0.24863202500906678**, and isolated fresh-cache elapsed time is **97.98336741694948 seconds**. All 4,080 task vectors reconstruct the retained 701,280 cells byte-for-byte. The unchanged mean model means this experiment does not modify long-horizon CAGR methodology. [Audited receipt](results/rough-bayesian/relaxed-hurst-differenced-rough-panel.json).

## Full theoretical Hurst support in differenced inference (M211)

M211 changes M210's H support to the strict mathematical rough domain (0, 0.5). All other priors, mean forecasts and numerical accuracy controls remain unchanged. It scored **0.2495417609939298** on the full 80-portfolio panel in **99.46118345798459 seconds** with a fresh isolated cache, trailing M210. All raw task vectors independently reconstruct the retained scored cells. [Audited receipt](results/rough-bayesian/full-hurst-differenced-rough-panel.json).

## Unregularized multiscale projection smoke screen

A further M210 ablation removes the fixed ridge and median-signal coefficient shrinkage from the multiscale projection, using SVD least squares. Every other fitted component and the original return mean curve remain unchanged; the latter is independently verified through 100 years. This is an ordinary least-squares projection of the filtered latent log-volatility proxy, not an exact raw-return likelihood or an intraday realized-volatility HAR implementation. Four of five smoke scores worsened against M210, so no full panel was launched and no official catalogue score was assigned. All cached repeats matched exactly. [Bounded smoke receipt](results/rough-bayesian/ols-multiscale-smokes.json).

## Empirical observation noise in the conventional offset (M212)

M212 retains M210 and uses the empirical log-square innovation noise variance already used by rough filtering in the one-step conventional-volatility offset. Conventional parameter fitting retains its original Gaussian-noise likelihood: this is a two-stage plug-in quasi-likelihood experiment, not a jointly refitted raw-return model. Five fresh/cached smoke repeats matched exactly; all 4,080 full-panel vectors reconstruct the 701,280 scored cells byte-for-byte, and every asset optimizer converged. CRPS is **0.2484924596795188**, 1.345% below fixed baseline M193 and 0.056% below M210. The isolated fresh-cache run took **97.27906462497776 seconds**. Mean, empirical forecast innovations, dependence and numerical resolution tolerance remain unchanged. [Audited receipt](results/rough-bayesian/consistent-noise-rough-panel.json).

## Parametric filtered-innovation screen

Two matched M212 candidates replace only the empirical forecast quantile nodes with standardized Student-t or Hansen skew-t quantiles, estimating tail shape and asymmetry from each asset's unchanged filtered innovation pool by conditional maximum likelihood. Inverse degrees of freedom include the Gaussian boundary and have no imposed maximum degrees of freedom. The original finite-grid centering and unit-variance normalization remain. Volatility fits, dynamic factor tolerance, return mean, dependence and scorer stay unchanged. This is conditional innovation fitting rather than a jointly estimated raw-return model. Six independent numerical tests cover agreement with the reference Hansen distribution, integral moments and unchanged 100-year mean curves. Ten fresh/cached smoke pairs match exactly. [Hansen (1994)](https://users.ssc.wisc.edu/~behansen/papers/ier_94.html), [reference implementation](https://arch.readthedocs.io/en/latest/univariate/generated/arch.univariate.SkewStudent.html), [smoke evidence](results/rough-bayesian/parametric-innovation-smokes.json).

The Student-t arm is canonical **M213**, scoring **0.25305232973579417** in **98.96461529203225 seconds**. The Hansen skew-t arm is **M214**, scoring **0.24975351800017143** in **101.83853595802793 seconds**. Both completed all 80 portfolios with every retained score cell independently reconstructed from the 4,080 raw vectors. All rough and innovation fits reported convergence. Neither beats M212. [Full audited evidence](results/rough-bayesian/parametric-innovation-panels.json).

## Filtered-proxy log-HAR smoke screen

A matched M212 ablation replaces the conventional multiscale forecast moments with a stationary nonnegative log-HAR(1,5,22) fitted by conditional Gaussian maximum likelihood to the predecessor's RTS-smoothed log-volatility proxy. The return mean is held byte-identical through 100 years. Conditional AR means and variances agree with an independent dense state-space recursion; interior estimates agree with ordinary least squares. This is an adaptation of [Corsi (2009)](https://doi.org/10.1093/jjfinec/nbp001) to a filtered proxy, not the canonical intraday realized-volatility input. All five smoke repeats reproduce exactly. Three shorter cases improve, but both long-horizon cases worsen, one by about 22% against M212. Many assets collapse to a daily AR coefficient with zero weekly/monthly loadings, indicating the smoothed proxy carries the original smoother's imposed persistence. No full panel or official catalogue score is assigned. [Smoke evidence](results/rough-bayesian/har-proxy-smokes.json).

A standard-Gaussian innovation arm provides the zero-tail/zero-skew parameter control on the same M212 shell. All five fresh/cached smoke repeats match exactly and its century-long mean remains unchanged. Only forecast quantile nodes change; empirical innovations remain the inputs to volatility/noise fitting. This isolates the forecast innovation family without a joint refit.

The Gaussian control is canonical **M215**. Its audited full-panel CRPS is **0.25198127331647685**, with isolated fresh-cache run time **99.21976537501905 seconds**. All 4,080 raw vectors reconstruct the 701,280 scored cells exactly; neither the fixed baseline nor M212 is beaten. [Audited receipt](results/rough-bayesian/gaussian-innovation-panel.json).

## Conventional observation-noise likelihood screen

The fixed empirical-noise refit uses the original innovation-pool R in conventional parameter fitting, smoothing and rough inference. It retains original SV priors and the original return mean, but four of five smoke scores worsen, including a severe longest-horizon loss; no full panel is assigned.

A joint-noise version estimates log R alongside conventional level, persistence and amplitude in one compiled Gaussian log-square quasi likelihood, retaining the existing parameter priors and using a flat log-R prior. Strictly positive R is constrained only by floating-point numerical limits; the learned value also supplies the causal offset and rough filter. Conditional innovation nodes are recalculated from the updated smoothed volatility. The original return mean curve stays byte-identical through 100 years. This remains a Gaussian quasi likelihood, not an exact log-chi-square mixture/raw-return posterior. Dense Gaussian conditioning validates the shared likelihood and smoother; the joint fitted target improves its fixed-noise reference, and both volatility fits use the identical R. All five fresh/cached score vectors match exactly, with two improved smokes. [Smoke evidence](results/rough-bayesian/conventional-noise-smokes.json).

The joint-noise arm is canonical **M216**. Full-panel CRPS is **0.24648440999477378**, with isolated fresh-cache run time **101.5046583749936 seconds**. All 4,080 vectors reconstruct the 701,280 scored cells exactly, and all conventional and rough optimizers converged. It improves M193 by 2.143% and M212 by 0.808%; it wins 56 of 80 portfolio-average comparisons against M193. Its mean forecast is unchanged. [Audited receipt](results/rough-bayesian/joint-noise-sv-panel.json).

## Standalone spectral rough variance-targeting screen

A M212 ablation fits rough covariance to the raw log-square proxy, without a conventional-volatility offset, and replaces its conventional multiscale forecast variance curve with the historical centered-return variance. The stationary rough multiplier's unit second moment sets the variance level. The unchanged predecessor fit still supplies the return mean and empirical innovation/noise inputs; this is standalone rough volatility in the variance generator, not an entirely replaced fitting shell. Two independent tests validate the pre-overlay node variance and unchanged century mean. Five cached repeats match exactly. Four smoke scores worsen, including both long horizons, so no full panel or catalogue score is assigned. [Smoke evidence](results/rough-bayesian/standalone-spectral-smokes.json).

## Interpolation-aware innovation normalization

A matched M216 candidate centers and scales empirical quantile nodes using the exact integrals of their piecewise-linear quantile function over uniform probabilities. An interval with endpoints a,b contributes mean (a+b)/2 and second moment (a²+ab+b²)/3; intervals have equal probability width. This replaces discrete equal-node moment normalization, without changing the quantile grid, return clipping, volatility fits, RNG or dependence. It guarantees reference-uniform innovation moments before clipping, not the conditional marginal moments of the unchanged dynamic copula. Four independent tests compare piecewise numerical integration and retain the original century-long mean curve. All five fresh/cached smoke vectors match exactly; two improve and three worsen against M216. [Smoke evidence](results/rough-bayesian/continuous-quantile-smokes.json).

The interpolation-normalized arm is canonical **M217**, with full-panel CRPS **0.2475449989529358** and isolated fresh-cache time **110.8023822910036 seconds**. All 4,080 vectors reconstruct the 701,280 scored cells exactly, and all fitted optimizers converged. It trails M216. [Audited receipt](results/rough-bayesian/continuous-quantile-panel.json).

### Raw-return sparse Laplace conventional-volatility screen

The next asset-level candidate replaces the conventional Gaussian log-square
quasi likelihood with Gaussian return observations and an AR(1) latent log
variance. A tridiagonal Newton solve finds the conditional latent mode, and its
Hessian determinant supplies a Laplace marginal likelihood. Analytic gradients
include the mode's implicit dependence on the three conventional parameters.
Those parameters use the predecessor's priors/support and MAP estimation; this
is approximate latent-state integration, not full INLA or parameter-posterior
integration. The construction follows the latent-Gaussian stochastic-volatility
example in [Rue, Martino and Chopin (2009), section 5.3](https://doi.org/10.1111/j.1467-9868.2008.00700.x).

The original mean curve, multiscale construction, empirical-return mapping,
rough spectral target, dynamically resolved prediction kernel, dependence,
seeds and scorer remain in place. The innovation pool is recalculated from the
new historical state fit. Its empirical log-square variance calibrates the
rough likelihood and causal conventional offset; the extra freely fitted
observation-variance parameter from M216 is absent from this raw-return fit.
Nine independent numerical/mean checks and five fresh/cached smokes passed.
The hypothetical 100-year, 240-path asset guard preserved the mean curve and
showed no systematic median-CAGR collapse on its six assets. Smoke scores are
mixed and do not establish a full-panel ranking.

The shared causal predictor now resides in an importable module, preserving
its arithmetic while allowing compiled caches to survive different candidate
module aliases. A two-process cache regression and all five saved M216 smoke
loss vectors verify the cache repair without changing scores.

The raw-return Laplace arm is canonical **M218**, with full-panel CRPS **0.2525845890080237** and isolated fresh-cache time **117.87531887501245 seconds**. All 4,080 vectors reconstruct the 701,280 scored cells exactly; all conventional and rough optimizers converged. It wins 39 of 80 portfolio averages against M193, but its overall score trails M193 and M216. [Audited receipt](results/rough-bayesian/raw-return-laplace-panel.json).

The Student-return Laplace arm is canonical **M219**, with full-panel CRPS **0.24548488040352726** and isolated fresh-cache time **128.6077183749876 seconds**. All 4,080 vectors reconstruct the 701,280 scored cells exactly; conditional latent modes and all conventional and rough optimizers converged, and conventional MAP projected mean gradients pass the 1e-6 gate. It wins 46 of 80 portfolio averages against M193 and beats M216 overall, but does not satisfy the goal score/runtime tradeoff. The initial failed run was rejected, and the validated run used no recycled scores. [Audited receipt](results/rough-bayesian/student-return-laplace-panel.json).

M219 uses unit-variance Student return observations, jointly fitting inverse degrees of freedom with the three conventional SV parameters. Inverse degrees of freedom has a uniform prior on [0, 1/2), with the Gaussian limit at zero and no imposed maximum degrees of freedom. The original three SV priors remain. Sparse Laplace integration handles historical states; hyperparameters use MAP, not full INLA or posterior integration. The mean-scaled objective uses analytic gradients and a bounded L-BFGS-B fit with an SLSQP retry; accepted fits must satisfy a projected mean-gradient tolerance of 1e-6. The rough layer still uses the differenced Gaussian log-square spectral quasi likelihood and empirical innovation-pool noise. This changes historical fitting, while retaining empirical forecast innovations. The six-asset 100-year simulation guard preserved the original mean curves and passed the positive-CAGR condition.

The Student-implied-noise arm is canonical **M220**, with full-panel CRPS **0.24423173688367905** and isolated fresh-cache time **126.57468083297135 seconds**. All 4,080 origin vectors reconstruct 701,280 scored cells exactly, and conventional MAP fits pass the projected mean-gradient gate. It replaces only M219's rough observation centering/noise with analytic moments from its learned Student tail parameter; the observation likelihood for rough fitting remains Gaussian quasi likelihood, and the original log-square floor remains. The moments apply to the uncensored Student density. Independent return-density quadrature verifies the moments, all five fresh/cached smoke vectors match, and the 100-year six-asset CAGR guard passes. It improves M193 by **3.036961817269601%**, M219 by **0.5104768643137136%**, and M216 by **0.9139211324328711%**. Its runtime is **1.0067422348323924 times** M193, meeting the >=3% score / <=2x runtime breakthrough condition. The research goal remains active while further defensible directions remain. [Audited receipt](results/rough-bayesian/student-implied-noise-panel.json).

The next evidence-backed direction is leverage rather than relaxing conventional parameter support: none of M220's 2,549 distinct cached asset/origin fits hit the persistence upper bound or amplitude lower/upper bounds (persistence range 0.4451–0.9977; amplitude 0.0418–0.8635). Their descriptive lagged return/state-innovation correlations are negative in 85.76% of fits, with median -0.07775. These are plug-in correlations from conditional state modes, not joint leverage-likelihood estimates or proof of causality. The future rough-state innovations remain independent of return innovations; the current moment curves do not use the fitted correlation. A controlled experiment can test a return shock at time t coupled to the volatility innovation that affects time t+1, preserving the pre-clipping conditional mean. [Omori, Chib, Shephard and Nakajima](https://shephard.scholars.harvard.edu/publications/stochastic-volatility-leverage-fast-and-efficient-likelihood-inference) distinguish this timing from contemporaneous return/volatility coupling, which can change the martingale-difference property. The fitted Student-tail likelihood and the independent-OU rough representation require a specific joint model and calibration audit before this can be called a canonical leverage implementation; it is not wired or scored yet. This remaining concrete direction is why the goal has not been closed after M220's threshold breakthrough.

A separate source-audit finding supports a controlled pathwise-volatility experiment before conflating it with leverage: `moment_return_curves` averages the conventional multiscale Gaussian state law into return mean/variance curves, whereas the rough OU factors are simulated path by path. The conventional fitted return/state correlation is not used in those moment curves. Restoring the existing multiscale state's random paths with a lognormal multiplier normalized to second moment one could isolate the discarded higher moments and temporal volatility clustering while preserving the original mean curve and unconditional variance, with no new fitted parameters. This would be a forecast-distribution change, not a parity optimization; separate independent moment/covariance tests and the long-CAGR guard are required before a panel. It is not yet implemented or scored. Leverage would be a further isolated change after that comparison.

M221 is the pathwise multiscale ablation of M220: identical Student/Laplace conventional fit, implied log-square noise and dynamic rough fit, but existing multiscale future states are simulated along each asset path. Its normalized multiplier has second moment one under the same fitted Gaussian state law, preserving pre-clipping daily return means/variances while changing higher moments and serial behavior. No new fitted parameters are added. Native disabled-state control reproduces the original rough kernel byte-for-byte; independent recurrence and Gaussian integration validate state covariance and normalization. All five fresh/cached smoke vectors match and the six-asset 100-year CAGR test passes. Full-panel CRPS: **0.24400259165672855**. All 4,080 origin vectors independently reconstruct all 701,280 cells. [Audited receipt](results/rough-bayesian/pathwise-multiscale-panel.json).

M221 runtime is **147.78993545804406 seconds**, with 55/80 portfolio wins against M193 and 48/80 against M220. Its score improvement is **3.1279353233081397%** versus M193 and **0.09382287080063056%** versus M220; runtime is **1.175482876430126 times** M193. It satisfies the full-panel >=3% score / <=2x runtime threshold, while adding a small increment over M220.

M222 fits leverage jointly with conventional level, persistence, amplitude and inverse Student degrees of freedom. The likelihood uses the Gaussian rank of a raw Student return at t in the conditional Gaussian state transition at t+1. This is sparse Laplace marginal MAP, not a parameter posterior or an Omori mixture sampler. A uniform rho prior has support (-1,1), with machine precision boundary guards. The existing multiscale shock projection couples each next volatility innovation to the previous empirical-return Gaussian rank; the first forecast innovation conditions on the last observed Student return rank. Conditional state moments and the variance curve reflect that first shock; the original return mean is preserved. The rough fit and independent rough driver remain as in M221, so this is a two-stage forecasting-shell experiment, not a jointly estimated rough/Student return model.

Lagged return/volatility dependence follows the [leverage state-space construction](https://shephard.scholars.harvard.edu/sites/g/files/omnuum7741/files/fastleverage.pdf); the Student Gaussian-rank copula extension is explicitly implemented and checked here, rather than claimed to be that paper's sampler. Independent joint-density, finite-difference, sparse/dense curvature, zero-leverage path/RNG parity, conditional covariance and extreme-tail incomplete-beta checks pass. Extreme probabilities are evaluated in log space with accuracy-controlled quadrature when the ordinary Student CDF underflows. This preserves all five original smoke vectors exactly. The initial failed panel is rejected, and the valid panel uses a fresh fit cache and no recycled scores. All five cached repeats match and the six-asset 100-year CAGR guard passes. Full-panel CRPS: **0.2537399472056091**. All 4,080 vectors independently reconstruct 701,280 cells, and all conventional and rough fit gates pass. [Audited evidence](results/rough-bayesian/joint-leverage-panel.json).

M222 completed in **202.7764 seconds**, with **41/80** portfolio wins against M193. Its aggregate score is **0.7379% worse** than M193 and **3.9907% worse** than M221, so it does not meet the research goal. Local installed-wheel suites passed 492 tests on each supported Python version, with the two separately gated checks run in their corresponding environments; installed-wheel metadata/snapshot checks passed on both versions, and the master catalogue passed 29 tests. The research goal remains active; no production promotion is made.

M223 uses M221 with uniform rough Hurst support over (0, 1/2), instead of (0.01, 0.49). The same optimizer interior guards, other priors, Student conventional fit, log-square observation moments, pathwise multiscale states, dynamic OU approximation accuracy target, mean and forecast shell remain. This is MAP under the same Gaussian differenced debiased Whittle rough quasi likelihood, not raw-return joint Bayesian rough inference. The domain follows [Volatility is rough](https://arxiv.org/abs/1410.3394). No new fitted parameter or prescribed factor count is added. All five smoke vectors reproduce cached outputs exactly, and the six-asset 100-year CAGR guard passes. Full-panel CRPS: **0.24454535810237815**. All 4,080 vectors independently reconstruct 701,280 cells, and all conventional and rough fit gates pass. [Audited evidence](results/rough-bayesian/student-full-hurst-panel.json).

M224 retains M221, but estimates the effective rough-proxy noise variance alongside H, log kappa and log amplitude. The original priors on those three rough parameters remain; the additional variance component maximizes the likelihood under a flat variance-scale target. Its numerical coordinate is log1p(R / Student variance), with no Jacobian prior term. The finite upper search bound max(I / N) follows from the sign of the likelihood derivative, rather than an imposed calibration cap. The same estimated R supplies terminal rough filtering. The unchanged Student moments continue to supply the conventional causal predictor. This remains Gaussian differenced debiased Whittle quasi likelihood, not an exact raw-return rough model or a claim to recover true Student observation noise. The transformed proxy includes a floor, clipping and conventional prediction subtraction, which motivate testing an effective fitted variance rather than fixing it to uncensored Student moments. Independent dense differenced-covariance and dense Gaussian-conditioning references pass. No MCMC or representative posterior draws are introduced. All five cached smoke vectors match exactly; the six-asset 100-year CAGR guard passes. Full-panel CRPS: **0.24559602973028682**. All 4,080 vectors independently reconstruct 701,280 cells and all fit gates pass. [Audited evidence](results/rough-bayesian/student-joint-rough-noise-panel.json).

M225 retains M221's conventional Student/Laplace fit, rough inference, multiscale paths, original return mean and dependence. Only the forecast innovation quantile nodes change: they use the same inverse degrees of freedom learned in the raw-return fit, with the exact Gaussian limit at zero. The existing midpoint grid size follows the requested simulation count; finite-grid centering, unit-second-moment normalization, interpolation and clipping remain unchanged. This is a discretized Student-shape forecast within the existing shell, not an exact continuous Student generative model or full Bayesian posterior prediction. No extra fitted parameter is introduced. Student return innovations in stochastic-volatility models are supported by [Jacquier, Polson and Rossi (2004)](https://people.bu.edu/jacquier/papers/jpr.je2004.pdf); this candidate does not implement their MCMC sampler. An independent closed-form Student(3) CDF inversion verifies quantile nodes, and grid moments/order remain valid across the fitted tail support. All five cached smoke vectors match exactly; the six-asset 100-year CAGR guard passes. Full-panel CRPS: **0.25121073766012286**. All 4,080 vectors independently reconstruct 701,280 cells; both volatility fits and implied Student noise moments match M221 across every origin. [Audited evidence](results/rough-bayesian/matched-student-innovations-panel.json).

M226 replaces M221's fixed 0.05 ridge and median-signal shrinkage with conditional Gaussian empirical Bayes projection. The existing four EWMA basis columns are standardized and assigned a shared Gaussian loading variance, estimated with stationary AR1 residual persistence and innovation variance by restricted marginal likelihood. The intercept and loadings are integrated in that marginal likelihood; noise variance is profiled. Forecasting uses the conditional loading posterior mean and fitted residual dynamics, without integrating hyperparameter uncertainty. This is conditional on the estimated Student latent log-volatility path, not joint exact Bayesian inference from raw returns. The variance-zero loading boundary is evaluated explicitly. Analytic gradients and cached small sufficient statistics avoid processing full history at each optimizer step. Numerical support bounds protect floating-point arithmetic, without fixed scientific shrinkage or persistence caps. The original return mean, Student raw-return fit, empirical innovation nodes, rough fit, dynamic kernel tolerance, dependence and scorer stay unchanged. [MacKay (1992)](https://authors.library.caltech.edu/records/r7qgh-q6g10) motivates evidence-based regularization; [Patterson and Thompson (1971)](https://doi.org/10.1093/biomet/58.3.545) gives the restricted-likelihood variance-estimation framework. Sixteen independent covariance/gradient/whitening tests pass, all five fresh/cached smoke vectors agree exactly, and the six-asset century CAGR guard passes. Full-panel CRPS: **0.7479695843647364**. All 4080 vectors reconstruct 701280 cells byte-for-byte, while all Student and rough fit diagnostics match M221. [Audited evidence](results/rough-bayesian/eb-multiscale-panel.json).

M226 is a negative result: CRPS 0.7479695843647364 versus M221 0.24400259165672855, with 149.0312974169501 seconds elapsed and 24/80 portfolio wins against M193. All 2549 unique asset fits converge (maximum mean objective gradient 1.1452600271733147e-7), but 1653 select residual persistence above 0.9999. This is evidence against the joint conditional projection as a long-horizon forecasting change; a better latent-history likelihood does not establish better OOS distributions. No promotion.

The loading-only empirical-Bayes follow-up retains the original multiscale level and residual forecasting estimator, learning only a shared shrinkage variance conditional on the original residual AR1 whitening. Six independent covariance/gradient tests pass and all five cached smoke vectors match exactly, but every smoke scores worse than M221. It is rejected before a full panel and receives no canonical rank or full-panel score. [Bounded evidence](results/rough-bayesian/eb-loading-shrink-smokes.json).

M227 removes the hard scientific support bounds for rough kappa and amplitude, retaining the original proper Gaussian priors on log parameters with unchanged centers and widths. The Hurst support stays (0.01,0.49). MAP optimization uses standardized prior coordinates with unbounded log-parameter axes; numerical exponential representability checks remain. The interior objective is byte-identical to M221, and direct finite-sample covariance sums verify the same differenced debiased Whittle spectrum beyond both old bounds. This remains the [debiased Whittle](https://arxiv.org/abs/1605.06718) Gaussian quasi-likelihood framework, not an exact raw-return likelihood. Original mean, Student raw-return fitting, analytic observation moments, empirical forecast nodes, multiscale future states, dynamic OU accuracy tolerance, dependence and scorer stay unchanged. Five correctness tests pass; all five cached smoke vectors match exactly and the six-asset century CAGR guard passes. Full-panel CRPS: **0.24398463100872367**. All 4080 vectors reconstruct 701280 cells byte-for-byte, and all conventional Student fit/noise diagnostics match M221. [Audited evidence](results/rough-bayesian/untruncated-priors-panel.json).

M227 scores 3.1350659240% better than fixed goal baseline M193 and 0.0073608431% better than M221, in 142.4829198340 seconds elapsed (local research panel). It wins 55/80 portfolios against M193. Of 2549 unique fits, 126 select kappa above the former cap and 38 select amplitude above the former cap. A bounded additional-mode audit on 30 smoke asset histories used starts near both retained H endpoints with fitted kappa/amplitude as the other starting coordinates; the greatest objective gain was 3.2589014154e-8, providing no evidence for adding those restarts to the runtime. [Mode-audit evidence](results/rough-bayesian/untruncated-priors-mode-audit.json).

The multiscale-offset candidate retains M227 and substitutes the one-step Gaussian predictor of its existing four multiscale states plus residual AR1 for the conventional single-state offset used in rough fitting. Its parameters, full innovation covariance and Student log-square observation variance are the existing fitted values; no additional parameter is learned. The state recursion is verified against dense Gaussian conditioning with one, three and five states, including a test that future observations cannot change earlier predictions. All five cached smoke vectors match exactly. Two medium-horizon smokes improve, while three worsen, including roughly 4.1% worse on the longest smoke. The subsequent full-panel result is retained as M228 below. [Bounded evidence](results/rough-bayesian/multiscale-offset-smokes.json).

M228 uses a causal Gaussian prediction from the existing four multiscale states plus residual AR1 instead of the conventional single-state observation offset. The predictor uses the already fitted loadings, full innovation covariance and analytic Student observation variance, with stationary initial state covariance. No additional parameter is estimated. The observation preprocessing, raw Student fit, empirical forecast nodes, original return mean, untruncated proper rough priors, future multiscale state law, dynamic kernel tolerance, dependence and scorer remain unchanged. Dense Gaussian conditioning validates the one-step predictor for one, three and five states, and a causal test confirms future observations cannot change earlier predictions. Five fresh/cached smoke vectors agree exactly, and the six-asset century CAGR guard passes. Full-panel CRPS: **0.2452184293792364**, versus M227 **0.24398463100872367**. Elapsed time is **149.10882762499386 seconds**. This is a negative result, not a promotion. All 4080 vectors reconstruct 701280 cells byte-for-byte; conventional Student fit and noise diagnostics match M221. [Audited evidence](results/rough-bayesian/multiscale-offset-panel.json).

The unclipped-log-proxy candidate removes only the 0.5/99.5 percentile clipping of the rough log-square observations. The existing near-zero finite-value floor remains. It retains M227's Student raw-return fit, analytic Student moments, conventional single-state causal offset and all future forecast components. This eliminates one fixed clipping rule and reduces the mismatch between percentile-clipped observations and uncensored Student observation moments; it remains a Gaussian proxy quasi likelihood. Four of five smokes improve versus M227, including approximately 0.70% on the longest smoke. All cached vectors match, and removing redundant clipped-proxy computation preserves every saved smoke vector byte-for-byte. The six-asset century CAGR guard passes. The subsequent full-panel result is retained as M229 below. [Bounded evidence](results/rough-bayesian/unclipped-proxy-smokes.json).

M229 removes only the 0.5/99.5 percentile clipping of historical rough observations. The original finite near-zero floor, analytic Student centering and observation variance, conventional single-state causal offset, raw Student likelihood, original return mean, empirical forecast nodes, pathwise multiscale law, untruncated proper rough priors, dynamic kernel tolerance, dependence and scorer remain unchanged. Four of five smokes improve over M227; optimization removes an unused clipped-proxy computation and preserves all five complete score vectors byte-for-byte. Fresh/cached vectors also match exactly, and six-asset century CAGR guards pass. Full-panel CRPS: **0.24425777495214046**, versus M227 **0.24398463100872367**; elapsed time **145.1817946669762 seconds**. All 4080 vectors reconstruct 701280 cells byte-for-byte. [Audited evidence](results/rough-bayesian/unclipped-proxy-panel.json).

The Student continuous-quantile pilot retains M227 and changes only normalization of the empirical innovation quantile map. Its existing interpolation uses equally spaced endpoint probabilities, so arithmetic moments across nodes do not equal moments of its piecewise-linear distribution. A 30-asset-fit audit measured reference-uniform innovation standard deviations of 0.9741–0.9903 and mean magnitude up to 0.0024. Analytic integration of each segment gives exact zero mean and unit variance before the existing daily return clipping. This does not imply conditionally uniform innovations under dynamic dependence. No fitted parameter or return mean curve changes. Independent quadrature tests confirm both moments; century mean curves remain byte-exact, cached smoke vectors agree exactly, and the six-asset century guard passes. Two of five smokes improve while the longest worsens; the subsequent full-panel result is retained as M230 below. [Bounded evidence](results/rough-bayesian/student-continuous-quantile-smokes.json).

M230 retains M227 except empirical innovation quantiles use analytic first and second moments of their piecewise-linear interpolation under reference-uniform probabilities before daily return clipping. No fitted parameter, conventional or rough likelihood, prior, original return mean, multiscale law, dependence, calendar or scorer changes. Conditional dependence uniforms are not assumed independently uniform. Independent segment quadrature verifies both moments, fresh/cached smoke vectors agree exactly, the original century mean curve is byte-exact, and six-asset century CAGR guards pass. Full-panel CRPS: **0.24838433757645187**, versus M227 **0.24398463100872367**; elapsed time **151.6883312499849 seconds**. All 4080 vectors reconstruct 701280 cells byte-for-byte. [Audited evidence](results/rough-bayesian/student-continuous-quantile-panel.json).

The current research decision is to retain M227: full canonical CRPS 0.24398463100872367 versus fixed M193 baseline 0.2518812750312293, a 3.1350659240% improvement, with cold panel elapsed time 142.4829198340 versus controlled baseline 125.7269998750 seconds (1.1333 times). Original return mean curves remain unchanged and century asset guards pass. The subsequent multiscale historical offset, removal of observation clipping and exact continuous innovation normalization all fail to beat M227; their complete results are retained as M228–M230. Earlier observation-noise, likelihood, leverage, multiscale empirical-Bayes, rough support and optimization-mode experiments remain documented above. The requested breakthrough threshold is met. No further change supported by a measured inconsistency or a successful bounded pilot remains in this reviewed neighborhood; another architecture or parameter search would require a new substantive hypothesis. This is an evidence-limited research decision, not a claim of global optimality or statistical significance of the small M227–M221 difference, and it does not promote a production engine.

## Learned multiscale decay locations and component order

The renewed research baseline was M227 (CRPS 0.24398463100872367). M231 now leads at 0.24289420743117274. The user removed the runtime ceiling; subsequent candidates must improve the best validated score while remaining optimized for speed. The first matched candidate learns four conventional EWMA decay rates per asset, retaining the existing 0.05 ridge and signal-median loading shrinkage, residual AR1 estimator, covariance coupling, Student raw-return fit, implied noise, rough fit and priors, dynamic kernel tolerance, original mean, empirical innovations, dependence and scorer. Its objective is the profiled conditional Gaussian one-step prediction likelihood on the already fitted Student latent volatility mode. It is not a jointly estimated raw-return multiscale state model. Decay support is (0,1), with machine representability guards only; initial calendar rates are starting values rather than fixed fitted rates. Compiled EWMA and likelihood calculations avoid Python loops over history. Supplying the original rates reproduces every original construction field byte-for-byte. Independent recursion verifies the predictive likelihood.

The second candidate selects its component count by forward conditional BIC: start with one fitted rate, add one initialized in the largest log-timescale interval, and stop at the first increase in 2*NLL + (2*k+3)*log(n-1). The parameter penalty counts rates, loadings, level, residual persistence and innovation variance conservatively despite loading shrinkage. This is approximate conditional model selection, not exact Bayesian evidence or a guarantee of globally optimal order. It follows the model-dimension penalty principle of [Schwarz (1978)](https://doi.org/10.1214/aos/1176344136), adapted here to a fitted latent proxy. There is no prescribed component count or user-portfolio-based choice: selection uses each asset's available history. A failed gradient line search invokes Powell on the same objective; unresolved convergence errors fail closed.

Seven mathematical/construction tests pass. Both candidates improve four of five smokes, but each worsens one long-history case. The fixed-four candidate's longest smoke improves by about 7.4%; the adaptive-count variant improves by about 13.1%. Fresh/cached complete vectors agree exactly and both six-asset century CAGR guards pass; original mean curves remain byte-identical. No full-panel rank or score is assigned yet. [Bounded evidence](results/rough-bayesian/learned-multiscale-rates-smokes.json).

M231 learns conventional multiscale decay locations and selects component count by forward conditional BIC. Full-panel CRPS: **0.24289420743117274** versus M227 **0.24398463100872367**; elapsed time **145.81819229200482 seconds**. All 4080 loss vectors reconstruct 701280 cells byte-for-byte; conventional Student fit/noise and rough parameters match M227 exactly. Original return mean remains unchanged and century asset CAGR guards pass. This is conditional proxy forecasting likelihood, not joint raw-return Bayesian multiscale inference. Existing loading ridge/shrinkage and residual/covariance estimators remain. [Audited evidence](results/rough-bayesian/adaptive-rates-panel.json).

M232 learns four conventional multiscale decay locations while retaining a fixed four-component count. Full-panel CRPS: **0.41701921590549546** versus M227 **0.24398463100872367**; elapsed time **181.41140141600044 seconds**. All 4080 loss vectors reconstruct 701280 cells byte-for-byte; conventional Student fit/noise and rough parameters match M227 exactly. Original return mean remains unchanged and century asset CAGR guards pass. This is conditional proxy forecasting likelihood, not joint raw-return Bayesian multiscale inference. Existing loading ridge/shrinkage and residual/covariance estimators remain. [Audited evidence](results/rough-bayesian/learned-rates-panel.json).

The matched single-component control retains M231’s learned-rate fitting, Student likelihood, residual AR1 and dynamic rough overlay, but stops after the first learned EWMA component instead of performing forward conditional-BIC selection. This isolates count selection; it is not a single-timescale model overall because the residual state and rough factors remain. It is unranked until a full-panel audit.

M233 fits exactly one learned EWMA component instead of adaptive component count. Full-panel CRPS: **0.2427871001627023** versus M227 **0.24398463100872367**; elapsed time **154.94550479203463 seconds**. All 4080 loss vectors reconstruct 701280 cells byte-for-byte; conventional Student fit/noise and rough parameters match M227 exactly. Original return mean remains unchanged and century asset CAGR guards pass. This is conditional proxy forecasting likelihood, not joint raw-return Bayesian multiscale inference. Existing loading ridge/shrinkage and residual/covariance estimators remain. [Audited evidence](results/rough-bayesian/single-rate-panel.json).

M233 is the new numerical leader at 0.2427871001627023, 0.044096% lower than M231. It uses one data-learned EWMA component plus the retained residual AR1 and dynamic rough factors. The next loading ablation removes only signal-median shrinkage in both rate fitting and final construction, with the original ridge retained. The same mathematical objective and compiled recursions are used; no chain, representative draw or fixed rough-factor count is introduced. These unshrunk arms are unranked smoke candidates until full-panel validation.

M234 removes median loading shrinkage consistently from fitting and construction. Full-panel CRPS: **0.2437557969452693** versus M227 **0.24398463100872367**; elapsed time **143.57121374999406 seconds**. All 4080 loss vectors reconstruct 701280 cells byte-for-byte; conventional Student fit/noise and rough parameters match M227 exactly. Original return mean remains unchanged and century asset CAGR guards pass. This is conditional proxy forecasting likelihood, not joint raw-return Bayesian multiscale inference. Original residual/covariance estimators and ridge remain; loading shrinkage details are specified in the resolved model definition. [Audited evidence](results/rough-bayesian/single-unshrunk-panel.json).

A predictive-loading candidate learns a common loading contraction in (0,1) per asset jointly with EWMA decay rates, instead of the fixed signal-median formula or deleting contraction. It uses the same conditional Gaussian prediction objective on the fitted latent proxy and the same forward BIC order search, adding the fitted contraction to the parameter count. Ridge 0.05 and the residual/covariance estimators remain. This is a regularized conditional forecasting approximation, not joint raw-return Bayesian inference. Five fresh/cache smoke vectors match exactly and century asset guards pass; it remains unranked pending the canonical panel.

M235 removes median loading shrinkage consistently from fitting and construction. Full-panel CRPS: **0.2434399646252218** versus M227 **0.24398463100872367**; elapsed time **150.88841825001873 seconds**. All 4080 loss vectors reconstruct 701280 cells byte-for-byte; conventional Student fit/noise and rough parameters match M227 exactly. Original return mean remains unchanged and century asset CAGR guards pass. This is conditional proxy forecasting likelihood, not joint raw-return Bayesian multiscale inference. Original residual/covariance estimators and ridge remain; loading shrinkage details are specified in the resolved model definition. [Audited evidence](results/rough-bayesian/adaptive-unshrunk-panel.json).

M236 learns loading contraction jointly with rates and selects component count by conditional BIC. Full-panel CRPS: **0.24169600516723397** versus M227 **0.24398463100872367**; elapsed time **158.23470458301017 seconds**. All 4080 loss vectors reconstruct 701280 cells byte-for-byte; conventional Student fit/noise and rough parameters match M227 exactly. Original return mean remains unchanged and century asset CAGR guards pass. This is conditional proxy forecasting likelihood, not joint raw-return Bayesian multiscale inference. Original residual/covariance estimators and ridge remain; loading shrinkage details are specified in the resolved model definition. [Audited evidence](results/rough-bayesian/predictive-loading-panel.json).

M236 is the numerical research leader at 0.24169600516723397, 0.449404% below M233 and 0.938020% below M227. It selected one EWMA component in 1830 histories, two in 707 and three in 12; fitted loading contraction median was 0.136961. The next matched probe changes only the uniform Hurst support from (0.01,0.49) to the theoretical rough domain (0,1/2), with the same relative interior optimizer guard and untruncated log-kappa/amplitude priors. This domain follows [Volatility is rough](https://arxiv.org/abs/1410.3394); our estimator remains a Gaussian differenced debiased Whittle quasi likelihood. The earlier M223 domain experiment was negative on its older backbone. The matched M236 probe has mixed smoke evidence and passed the century asset guard; it remains unranked.

M237 extends only rough Hurst support to (0,1/2). Full-panel CRPS: **0.2416808319759003** versus M227 **0.24398463100872367**; elapsed time **150.4711924159783 seconds**. All 4080 loss vectors reconstruct 701280 cells byte-for-byte; conventional Student fit and noise match M227; the exact unchanged-component checks are specified in each audit receipt. Original return mean remains unchanged and century asset CAGR guards pass. This is conditional proxy forecasting likelihood, not joint raw-return Bayesian multiscale inference. Original residual/covariance estimators and ridge remain; loading shrinkage details are specified in the resolved model definition. [Audited evidence](results/rough-bayesian/learned-loading-full-hurst-panel.json).

M237 is the numerical leader at 0.2416808319759003, 0.006278% below M236. Its full theoretical Hurst domain retains the fitted Student conventional parameters, analytic observation moments, learned conventional rates/loadings and original mean exactly. The no-ridge probe first exposed singular normal equations; replacing them with standard minimum-norm SVD least squares resolved those equations but worsened four smoke scores and produced nonfinite long-horizon forecasts. It is rejected before a canonical panel and has no catalogue rank. [Failed bounded evidence](results/rough-bayesian/zero-ridge-svd-smokes.json). The next matched count ablation fits exactly one EWMA decay and loading contraction, retaining M237’s Hurst support, priors, residual dynamics, rough kernel accuracy and all other forecasting components. Three smoke scores improve, one ties and the longest worsens; cached outputs and century guards pass. It remains unranked.

M238 fits exactly one EWMA decay and predictive loading contraction. Full-panel CRPS: **0.24188836805455496** versus M227 **0.24398463100872367**; elapsed time **136.68520212499425 seconds**. All 4080 loss vectors reconstruct 701280 cells byte-for-byte; conventional Student fit and noise match M227; the exact unchanged-component checks are specified in each audit receipt. Original return mean remains unchanged and century asset CAGR guards pass. This is conditional proxy forecasting likelihood, not joint raw-return Bayesian multiscale inference. Original residual/covariance estimators and ridge remain; loading shrinkage details are specified in the resolved model definition. [Audited evidence](results/rough-bayesian/single-loading-full-hurst-panel.json).

M239 removes percentile clipping of rough log-square observations, retaining all conventional fitting and forecasting components. Full-panel CRPS: **0.24202526341064862** versus M227 **0.24398463100872367**; elapsed time **145.4038729169988 seconds**. All 4080 loss vectors reconstruct 701280 cells byte-for-byte; conventional Student fit and noise match M227; the exact unchanged-component checks are specified in each audit receipt. Original return mean remains unchanged and century asset CAGR guards pass. This is conditional proxy forecasting likelihood, not joint raw-return Bayesian multiscale inference. Original residual/covariance estimators and ridge remain; loading shrinkage details are specified in the resolved model definition. [Audited evidence](results/rough-bayesian/learned-loading-unclipped-panel.json).

A matched M237 residual-persistence pilot minimizes total one-step volatility forecast error at fixed component rates and loadings, rather than fitting the decomposed residual AR1 directly. The analytical constrained OLS estimate is used consistently during rate/loading selection and state construction; the original stationary guard, residual innovation covariance rule, rough fit and other model components remain. Independent tests verify the minimizer and construction agreement, and all five cached smoke vectors are exact. Two smokes improve slightly, but the longest worsens from 0.102487583 to 0.529998868; it is rejected before a full panel and receives no canonical rank. The century mean/median CAGR guard passes, which does not offset the distribution-score failure. [Bounded evidence](results/rough-bayesian/predictive-residual-smokes.json).

A matched M237 observation-offset candidate uses the causal Gaussian predictor of its learned multiscale states and residual AR1, with the same Student log-square observation variance, before fitting rough covariance. No new parameter or future state law is added. This revisits the negative M228 fixed-four-scale offset on a substantially changed learned-rate/loading backbone, rather than repeating that configuration. Dense Gaussian conditioning and causal perturbation tests validate the predictor; all 30 smoke conventional fit receipts match saved M237 exactly. Three smoke scores improve, including both longer horizons, and cached vectors and century CAGR guards pass. It remains unranked until the full canonical panel and audit. [Bounded evidence](results/rough-bayesian/learned-multiscale-offset-smokes.json).

M240 uses causal learned-multiscale prediction as the rough observation offset. Full-panel CRPS: **0.24269856328739714** versus M227 **0.24398463100872367**; elapsed time **149.45633587497286 seconds**. All 4080 loss vectors reconstruct 701280 cells byte-for-byte; conventional Student fit and noise match M227; the exact unchanged-component checks are specified in each audit receipt. Original return mean remains unchanged and century asset CAGR guards pass. This is conditional proxy forecasting likelihood, not joint raw-return Bayesian multiscale inference. Original residual/covariance estimators and ridge remain; loading shrinkage details are specified in the resolved model definition. [Audited evidence](results/rough-bayesian/learned-multiscale-offset-panel.json).

A joint Gaussian state experiment retains M237's Student conventional fit and learned rates/loadings, but calibrates rough covariance against the sum of conventional, rough and observation-noise covariances on the raw Student-debiased log-square proxy. Conventional and rough historical states are conditioned together, including cross-block posterior covariance. This is conditional MAP with a Gaussian differenced debiased Whittle proxy likelihood, not joint raw-return Bayesian inference. Mean, empirical innovation nodes, dependence, policy and scorer remain. Three future-volatility controls use the same joint state law: direct sigma, conditional second-moment normalization, and retained conventional/rough stationary normalization around the M237 variance curve. Eight independent numerical tests pass; native paths and random streams match a scalar reference exactly, cached smoke vectors match, and century asset CAGR guards pass. Direct sigma worsens four of five cases and conditional normalization worsens all five; neither enters a full panel. Stationary normalization improves two of five cases and is selected for a full canonical comparison, without claiming a smoke winner. These candidates have no canonical rank until full-panel validation. [Bounded evidence](results/rough-bayesian/joint-gaussian-volatility-smokes.json).

M241 jointly conditions conventional and rough Gaussian states and uses retained stationary normalization around the M237 variance curve. Full-panel CRPS: **0.24296927753915046** versus M227 **0.24398463100872367**; elapsed time **158.02270416601095 seconds**. All 4080 loss vectors reconstruct 701280 cells byte-for-byte; conventional Student fit and noise match M227; the exact unchanged-component checks are specified in each audit receipt. Original return mean remains unchanged and century asset CAGR guards pass. This is conditional proxy forecasting likelihood, not joint raw-return Bayesian multiscale inference. Original residual/covariance estimators and ridge remain; loading shrinkage details are specified in the resolved model definition. [Audited evidence](results/rough-bayesian/joint-gaussian-stationary-panel.json).
