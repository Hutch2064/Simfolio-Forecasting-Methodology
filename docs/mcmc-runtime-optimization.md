# Asset-level parameter MCMC: runtime optimization

The scored two-chain, sixteen-node stochastic-volatility parameter-MCMC
candidate has been optimized without reducing its inference or forecast
workload. Its full canonical score is **0.25246784959071183**, compared with
**0.25439860867855635** for the previous Frontier: a **0.7589503330515897%**
reduction in CRPS. Lower is better. This is the current production Frontier specification, now included in the
canonical catalogue alongside both earlier production Frontiers.

## Statistical contract

Each asset retains its historical mean and the existing Gaussian log-square
SV quasi-posterior, prior terms, Jacobians, parameter support, and MAP
initialization. Two independently seeded Metropolis chains each use 1,024
burn-in and 2,048 retained iterations. Covariance adaptation occurs only
during burn-in at the original boundaries. Sixteen equally weighted retained
parameter draws provide conditional smoothed volatility paths. Prediction
matches moments across the sixteen component laws and uses the same empirical
innovation quantiles, four volatility scales, clipping, dynamic Gaussian
dependence, calendar rebalancing, and 15-basis-point turnover cost.

This is **parameter MCMC with moment-matched prediction**. It is not PGAS,
a full joint parameter/state posterior sampler, or an exact posterior
predictive mixture. Runtime changes do not alter these qualifications or
establish universal chain convergence.

The canonical evaluation uses all 80 portfolios, 51 origins per portfolio,
240 simulations, all 8,766 daily horizons, and 701,280 portfolio/horizon
cells. Origin losses are averaged within each cell, then cells receive equal
weight, exactly as in the public evaluator. The candidate improved the
portfolio-average score for 52 of 80 portfolios and the horizon-average
score for 7,907 of 8,766 horizons. All seven reported horizon buckets improved.
These are results on the research panel used for model selection, not a new
independent confirmation sample.

## Retained execution changes

| Component | Change | Fidelity condition |
| --- | --- | --- |
| SV likelihood | Compile the scalar recurrence; reuse gain/variance/log terms after an exact floating-point variance fixed point | Continue every observation and preserve accumulation order |
| Metropolis transitions | Compile transition blocks between the original covariance adaptations | Identical random draws, support checks, adaptation boundaries, proposals, retained traces, and acceptance rates |
| MAP initialization | Use the same compiled likelihood in the unchanged optimizer | Identical parameter tuple, bounds, starts, and iteration budget |
| Multiscale fitting | Use ordinary reductions, skip unused missing-data cleanup, share identical EWMA starting medians, batch component quantiles, and reuse unchanged standard deviations/means for finite inputs | Identical complete fit dictionaries; original fallback for nonfinite inputs |
| Predictive moments | Cache repeated powers and standardized empirical quantiles | Training/configuration keys only; identical arrays |
| Dependence filtering | Reuse inverses only when covariance bytes match | Identical terminal posterior; no approximate convergence rule |
| Forecast mapping | Stream draw temporaries and interpolate ascending quantile nodes in place | Identical complete copula uniforms, generator state, marginal values, and asset paths |
| Independent forecast paths | Parallelize simulations; overlap the next draw block using one Generator-owning producer | Identical random consumption and arithmetic within each simulation |
| Portfolio rejoining | Compile reductions with NumPy's original summation order; parallelize independent paths for long forecasts | Identical daily portfolio paths |
| Independent fitting | Bounded asset threads, concurrent independent chains for long histories, and GIL release in pure smoothing kernels | Preserve seed and result order; no shared sampler state |

Draw temporaries use at most two 1,024-day blocks; rejoining uses eight-simulation blocks.
All asset paths remain present. Memory therefore still grows with assets,
simulations, and forecast days, while unnecessary full-size temporary cubes
are eliminated. Automatic asset-worker counts are two below 2,048 observations and six
otherwise, capped by the CPU count and number of assets. At 8,192 observations,
two chains run concurrently and the CPU-based asset-worker cap is divided by
two. On the reference host, at most twelve fitting threads run per model
process. Path simulation uses at most four numerical threads plus one draw
producer. These thresholds are execution choices, not statistical parameters.

## Matched local runtime

Warm compilers, cold fitted-model caches, single-threaded BLAS, and interleaved
arms were used on one macOS ARM64 host with 15 logical CPUs and 24 GiB RAM.
Python 3.12.13, NumPy 2.5.3, SciPy 1.18.1, and Numba 0.67.0 were used.
Each entry is the median of four measurements; complete original/optimized
daily paths were compared byte for byte.

| Assets | Training observations | Forecast days | Original MCMC seconds | Optimized seconds | Elapsed speedup | Original/optimized CPU seconds | Frontier seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 507 | 126 | 0.4179 | 0.1068 | 3.91x | 0.4176 / 0.1706 | 0.0308 |
| 6 | 10,024 | 1,663 | 3.1034 | 0.3328 | 9.32x | 3.1014 / 1.7243 | 0.3203 |
| 6 | 2,921 | 8,766 | 2.3796 | 0.3901 | 6.10x | 2.3781 / 1.0396 | 1.2619 |

Elapsed improvements include concurrency; CPU reductions are reported
separately. MCMC remains slower than Frontier for the two shorter-horizon
cases and is faster in the long-horizon case. These are local harness
diagnostics, not native production or authenticated browser click-to-visible
measurements. Cross-platform bitwise equality is not asserted.

The separate asset-count benchmark uses 10,024 observations, 2,520 forecast
days, 240 paths, equal asset weights, and two interleaved measurements per
arm. These are scaling diagnostics rather than additional OOS score results.

| Assets | Original MCMC seconds | Optimized, six asset workers | Speedup | Original/optimized CPU seconds | Frontier seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 3.2276 | 0.3575 | 9.03x | 3.2257 / 1.7841 | 0.4376 |
| 25 | 13.0650 | 1.3787 | 9.48x | 13.0569 / 7.2156 | 1.3206 |
| 50 | 26.1081 | 2.6011 | 10.04x | 26.0898 / 14.2471 | 2.6537 |

Asset fitting remains linear in asset count for a fixed chain budget and
history length. Bounded concurrency reduces elapsed time without changing
that asymptotic cost. The shared factor/dependence fitting cost is inherited
from Frontier. This is a constant-factor optimization, not a claim of a new
asymptotic scaling law. Serial-asset, two-worker, and four-worker measurements are
also retained in the [raw evidence](results/mcmc-runtime-optimization/scaling-benchmark-final.json).

## Full replay and stopping evidence

The final implementation completed all **4,080 tasks across all 80 portfolios**
with literal byte equality across **8,649,331,200 asset-path values** and
**1,441,555,200 daily portfolio values**, **5,098 unique retained chain traces**,
and every scored loss. All denominator gates passed and CRPS remained
**0.25246784959071183**. The final [receipt](results/mcmc-runtime-optimization/final-full-panel-receipt.json),
[source archive](results/mcmc-runtime-optimization/final-validated-source.zip),
[manifest and complete task receipts](results/mcmc-runtime-optimization/README.md)
bind these checks to the published implementation. The publication audit verifies
identical executable ASTs after one trailing-blank-line cleanup. The serial, parallel-path,
and calibrated six-worker implementations also completed independent full
replays; their receipts and source snapshots are retained. Verification
elapsed time includes reference computations, comparisons, and explicit
pauses for isolated benchmarks. It is not used to claim runtime savings.

Profiles force serial asset/chain fitting to expose its remaining cost, while
path simulation retains its bounded parallelism and producer. The long-history
case spends approximately 81% of profiled wall time in chain transitions.
For the longest-horizon case, transitions account for approximately 56%,
dependence draws including the producer wait 11%, predictive moments 7%,
multiscale fitting 8%, and rejoining 8%. Nested timings overlap; they should
not be summed. Threaded elapsed and whole-process CPU times are measured
separately in the tables above. Raw profiles are archived alongside the report.

The following further opportunities were tested:

- Releasing the GIL in the unchanged pure RTS/EWMA kernels repeatedly improved
  the long-history case by 3.5–3.8%, with every daily path unchanged. Retained.
- Reusing the component power arrays improved the long-horizon case by
  2.3–3.1% in repeated isolated tests. Retained. Broader caching of drift and
  geometric terms added less than 1% beyond this simpler change; not retained.
- Sharing the identical EWMA starting median across four scales and batching
  component quantiles repeatedly improved the short-history case by about 2%,
  with complete fit dictionaries unchanged. Retained. Recycling normal-draw
  buffers yielded less than 1% benefit in two cases and regressed the longest
  horizon; rejected.
- Two asset workers beat one by about 23% for short histories; six beat the
  prior two-/four-worker settings by about 18–19% for medium/long histories.
  Eight beat six by less than 1% while using about 5% more CPU. Four path
  threads were retained; eight offered no material improvement.
- Compiling the dependence terminal filter changed last-bit results by up
  to 1.44e-15. Rejected because it failed exact output parity.
- Reusing proposal buffers retained traces but yielded approximately 1% or
  less improvement/noise at canonical history lengths. Not retained.
- Direct compilation without exact fixed-point reuse regressed the
  long-history case. The measured better implementation was retained.

Earlier accepted probes removed unused finite-input cleanup (about 2.7%
end-to-end gain in two cases) and combined parallel paths with next-block draw
overlap (about 26% in the long-horizon case). Complete fit dictionaries,
Generator states, marginal values, and paths remained unchanged.

The remaining dominant work is required chain transitions and complete
forecast computation. No identified material, profile-backed,
parity-preserving optimization remains untested. Reducing iterations,
posterior nodes, observations, horizons, assets, or numerical precision would
change the scored contract. The stopping criterion is exhaustion of concrete
measured opportunities, not a proof that no possible program can run faster.

The complete installed-wheel repository test suite and source/tests/tools
lint passed. See [test evidence](results/mcmc-runtime-optimization/test-receipt.json)
and [reproduction instructions](../tools/mcmc_runtime/README.md).
