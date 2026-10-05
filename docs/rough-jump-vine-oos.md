# Asset-level rough volatility, jumps, and HMM-vine experiment

This experiment compares the current production Frontier with four asset-level
ablations. Every configuration uses the same frozen data, training origins,
historical means, production SV variance anchors, policy rejoining, transaction
costs, and public scoring functions. The new mechanisms change asset marginals
or joint asset dependence before portfolio policies are applied.

## Configurations

| Configuration | Asset volatility and innovations | Joint asset dependence |
| --- | --- | --- |
| Production baseline | Two-chain parameter MCMC; sixteen-node moment-matched SV; filtered empirical innovations | Production dynamic Gaussian factor |
| Rough volatility | Production variance anchors multiplied by conditional pathwise rough volatility | Production dynamic Gaussian factor |
| Jumps | Empirical diffusion body plus compensated time-varying compound-Poisson jumps | Production dynamic Gaussian factor; separate correlated jump-count stream |
| HMM R-vine | Production asset marginals | Three-state HMM with a full fifteen-pair R-vine in each state |
| Combined | Rough volatility and compensated jumps | Three-state HMM R-vines; common regime path for diffusion and jump-count streams |

Rough volatility is an eight-factor daily Gaussian Volterra approximation,
fitted separately to each asset's training log-square returns. Its Hurst
parameter is fitted within `[0.03, 0.49]`; mean-reversion and scale are also
fitted. The conditional Gaussian state posterior is propagated pathwise. The
lognormal multiplier is normalized to preserve the production conditional
daily variance in the Gaussian law. This is a specific finite-factor rough-SV
candidate, rather than an exact fractional-Brownian-motion solver.

Jump events are identified at absolute standardized filtered innovation greater
than three. They are removed from the empirical diffusion pool before adding
jumps. A Beta(1,99) event-rate prior uses the posterior-averaged time sequence,
rather than treating sixteen parameter draws as sixteen observations. The
conditional rate decays from its 21-day EWMA terminal estimate toward the
training-average rate. Gaussian jump marks are estimated from the tail pool;
Poisson counts use an exact inverse-CDF recurrence. Drift compensation and
variance normalization preserve the production mean and variance anchors.

The HMM uses training-only rank-transformed filtered residuals. Twenty
Gaussian-copula initialization iterations precede weighted BIC selection of
full R-vines, allowing independence, Gaussian, Student, Clayton, Gumbel, and
Frank pairs with rotations. Twelve maximum HMM/vine updates retain the
selected structures and families and refit their parameters. The relative
likelihood stopping tolerance is `1e-5`. Future states begin one transition
after the final filtered state. Vine dependence describes joint statistical
returns; it does not implement liquidation feedback or an endogenous crash
mechanism.

## Evaluation and numerical verification

The unmodified canonical harness uses 80 six-asset portfolios, a common start
of 1979-12-31, 48 rolling origins and three temporal holdouts per portfolio,
240 simulations per origin, and every eligible daily horizon. Each model has
4,080 origin tasks and 701,280 portfolio-horizon cells. Exact empirical CRPS
uses the `n²` pairwise denominator on cumulative terminal log returns. Origins
are averaged within each cell before equally averaging cells. Missing tasks,
incorrect origin counts, and nonfinite forecasts fail completion.

Every newly evaluated production-baseline loss vector must match the validated
production reference byte for byte. Paired smoke runs compare the original and optimized implementations of all
five models on short-history, long-history, and maximum-horizon tasks. All
44,610 scored loss values must match byte for byte. Unit tests check complete
rough paths and filtering against reference recurrences, fused jump paths,
Poisson quantiles against SciPy, HMM smoothing against exhaustive state-path
enumeration, and CRPS against its explicit pairwise definition.

## Runtime implementation

The native C++ patch memoizes exact Student marginal transforms when only
correlation changes, hoists invariant Student density normalizers, and
computes repeated exact rank-grid quantiles once. It preserves the original
Boost arithmetic and falls back to the original quantile function for other
inputs. No approximate quantiles, fast-math flags, shortened fits, or reduced
vine family sets are used. A long-history fitted HMM, including all vine
parameters, transitions, terminal probabilities, and likelihood iterations,
matched the original byte for byte. With three state workers and three edge
threads, the stock implementation took 32.9817 seconds and 159.1698 CPU
seconds; the retained C++ patch took 22.8616 seconds and 103.9966 CPU seconds.
Additional cache and compiler variants were measured and rejected because
they provided no material gain.

Other retained changes include compiled numerical recurrences, exact
steady-state covariance reuse, contiguous rough-path traversal, fused
in-place jump transformations, parallel independent vine inversions, and
reuse of identical fitted models, quantile nodes, and random streams across
ablations. All histories, origins, paths, horizons, RNG sequences, fit budgets,
and scoring denominators are retained. Process and native thread counts are
selected with a matched twelve-fit worker calibration under a twelve-thread
compute budget. The selected twelve-process, single-thread configuration
completed the same twelve cold HMM fits in 120.2867 seconds, versus 129.2026
seconds for four processes with three state workers and 134.7076 seconds for
two processes with three state workers and two edge threads. All twelve
complete fitted-model hashes agreed across configurations.

A separate maximum-horizon case compared one and three C++ threads on the
same 2,921 training rows, six assets, 240 paths, 8,766 days, and two copula
streams. Full fitted models and both complete simulated uniform tensors were
byte-identical. One-thread fitting and simulation took 17.6401 and 14.8421
seconds; three threads took 9.4390 and 5.4493 seconds. These are local research
timings. Full-run worker totals share fitted caches across configurations and
include reference verification; they are not independent model latency or
production click-to-visible measurements.

## Stopped panel and rough-only continuation

The five-model run was stopped at the user's request after **2,769 of 4,080
origins per model**, including **53 complete portfolios**. The jumps, HMM-vine,
and combined configurations are retained as partially scored research models.
The rough configuration continues alone, reusing every valid completed rough
checkpoint; it does not rerun HMM fitting or the other configurations.

| Model | CRPS across the 53 complete portfolios | Full 80-portfolio CRPS |
| --- | ---: | ---: |
| Production | 0.25695328925791999 | — |
| Rough volatility | 0.25486122574946996 | — |
| Jumps | 0.26118982245282568 | — |
| HMM R-vine | 0.26148387863266082 | — |
| Combined | 0.26504135845165278 | — |

[Partial panel results](results/rough-jump-vine/partial-panel.json) contain all
80 portfolio rows for every configuration. Unfinished portfolio scores and
full-panel scores are null. [Retained horizon losses](results/rough-jump-vine/partial-cell-losses.npz)
contain the observed cell means and actual/expected origin counts, including
completed origins from unfinished portfolios. The frozen manifest, checkpoint
hashes, executed source archive, numerical smoke evidence, and native build
receipt are saved alongside these results.

## Completed rough-volatility result

The rough-only continuation completed all **4,080 origin tasks**, **80
portfolios**, **701,280 cells**, and **240 simulations per origin**. Its exact
empirical CRPS is **0.2511998559613309**, versus the production reference's
**0.25246784959071183**: a **0.5022396442% improvement**. Rough volatility
scored better on **45 of 80 portfolios**. Every cell's origin count and all
checkpoint hashes passed an independent reconstruction audit. All 2,769 reused
rough loss vectors were byte-identical to the stopped run; the remaining 1,311
origins evaluated only rough volatility.

[Full results](results/rough-jump-vine/rough-full-results.json),
[independent audit and all 80 portfolio scores](results/rough-jump-vine/rough-full-audit.json),
and [full horizon losses](results/rough-jump-vine/rough-full-cell-losses.npz)
are retained. The 46.4286-second evaluation continuation used cached training
fits and prior checkpoints; it is not a fresh full-panel or production latency
comparison. Jumps, HMM-vine, and combined remain partially scored with blank
unfinished portfolios and full-panel scores.

## Reproduction

Install the project with its `all-models,rough-jump-vine` extras, build the
parity-tested native patch, retain the validated
production checkpoint directory, and run:

```bash
python tools/rough_jump_vine/build_native.py --workspace /path/to/native-build

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
python tools/rough_jump_vine/run_panel.py \
  --output /path/to/experiment/full \
  --reference /path/to/validated-production-checkpoints \
  --workers 12
```

The run manifest freezes numerical source hashes, data provenance, every task
identity, versions, and worker count. Atomic per-origin checkpoints permit
resumption with that same manifest. Full results are emitted only after all
selected configurations complete every task and every denominator gate passes.

To continue only rough volatility from the stopped five-model checkpoints:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
python tools/rough_jump_vine/run_panel.py \
  --output /path/to/experiment/full-rough \
  --reference /path/to/validated-production-checkpoints \
  --model-id asset_rough_volterra_sv_eight_factor \
  --reuse /path/to/experiment/full-final \
  --workers 12
```

Reuse checks bind the candidate flags, numerical model and MCMC source hashes,
native binary, snapshot, task identities, simulation count, and scoring contract.
Only compatible, finite, complete origin checkpoints are imported.
