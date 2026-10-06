# Joint Bayesian rough volatility candidates

Nine new asset-level candidates are implemented in `tools/rough_bayesian` and
included in both public catalogues. The canonical catalogue now has 203 entries;
the broader active research catalogue has 395. The eight-factor Bayesian upgrade
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
