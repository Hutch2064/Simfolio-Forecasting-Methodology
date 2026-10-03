# Alternative SV parameter inference: exploratory research

The wider, all-eighty-portfolio screen **rejected the initially promising
mixture-importance candidate**: its CRPS was 0.25736549661342245, worse than
Frontier's 0.2558861226485342 and optimized MCMC's 0.2538154340980102 on that
same screen. No tested replacement is promoted.

These experiments construct and test three inference families while preserving
the existing asset-level forecasting architecture: gradient-based Hamiltonian
Monte Carlo, full-rank Gaussian variational inference, and Laplace importance
sampling. The strongest preliminary score came from a multi-start Student-t
mixture importance proposal. This is a research experiment, not a production
change or a new entry in the canonical ranking.

## Preserved statistical and scoring contract

The target is the same bounded three-parameter Gaussian log-square SV
quasi-posterior as the optimized parameter-MCMC candidate, including its priors
and Jacobians. Each asset retains its historical mean, empirical standardized
innovation pool, four volatility scales, sixteen parameter components, and
moment-matched prediction. Conditional state means come from the same RTS
smoother. The asset-level Gaussian dependence, calendar rebalancing, and
15-basis-point turnover charge are unchanged. This is not an exact joint
parameter/state posterior predictive simulation.

All forecasts use the frozen public snapshot, canonical portfolio definitions,
240 paths, and the unchanged public evaluator. Origin losses are averaged
within each portfolio/horizon cell, then cells receive equal weight. All
specified tasks and their full daily horizons remain in the denominator,
including tasks whose inference diagnostics fail. No bad fit is silently
removed or replaced with another model.

The initial and tuned screens use portfolios 1, 2, 3, 4, and 7 at rolling
origins 09 and 29 and all three chronological holdouts: train the first quarter,
first half, and first three quarters and test the remaining observations.
That is 25 tasks and 43,830 portfolio/horizon cells per arm. The wider screen
uses all eighty portfolios at those five origins: 400 tasks and 701,280 cells.
Having all cells does not reproduce the full canonical score because fewer
origins contribute to their averages. Scores from these screens must never
be compared numerically with full canonical scores in the ranking.

## Initial 25-task screen

Lower CRPS is better. Seconds are summed local cold-fit elapsed time over the
screen, with warm compilers and interleaved arms; they are not repeated-median
benchmarks or production CTV.

| Arm | CRPS | Elapsed seconds | Process CPU seconds |
| --- | ---: | ---: | ---: |
| Frontier | 0.17509934495472698 | 11.722 | 13.789 |
| Optimized parameter MCMC | 0.17033165914627443 | 8.723 | 29.376 |
| Single Laplace Student-t importance | 0.17005201398065756 | 7.662 | 19.050 |
| Full-rank Gaussian VI | 0.17087381509483993 | 10.143 | 33.329 |
| VI with importance correction | 0.16933071089759438 | 10.129 | 33.309 |
| Four-chain HMC, shorter budget | 0.16997917408850438 | 62.877 | 348.345 |

These promising score values did not establish reliable inference. Single
Laplace importance had minimum raw-weight ESS near one and maximum Pareto k
2.668. VI importance diagnostics reached k 4.991; 113 of 150 asset-fit calls
exceeded 0.7. The shorter HMC budget reached R-hat 1.249 and bulk ESS 12.3.
Optimizer success flags alone were insufficient: a VI fit reported success
while its largest gradient component was approximately 44.

All 150 model tasks completed. A quadratic postprocessing mistake then made
aggregation unnecessarily slow; the process was terminated during that step.
The aggregation was corrected and independently rerun from every persisted,
hashed task loss. Forecasts were not selectively rerun or discarded. All fifty
Frontier/MCMC comparator loss vectors matched the archived full reference
byte for byte. The original executed sources are preserved with the evidence.

## Training-only coverage repair and tuned screen

The three histories with the worst initial importance coverage were used to
investigate numerical geometry. This selection used training-fit diagnostics,
not their realized future returns. Multiple optimizer starts found distinct
stationary regions. The revised proposal samples an equal-weight Student-t
mixture covering distinct modes at both Hessian scale and twice that scale,
with 4,096 draws. Importance weights use the density of the entire mixture.
On these three histories, raw-weight ESS rose to 1,155–1,291 and Pareto k
ranged from -0.271 to 0.056. Increasing HMC to four chains with 1,024 warmup
and 2,048 retained draws also improved their mixing diagnostics.

| Arm | CRPS | Elapsed seconds | Process CPU seconds |
| --- | ---: | ---: | ---: |
| Frontier | 0.17509934495472698 | 11.853 | 13.934 |
| Optimized parameter MCMC | 0.17033165914627443 | 8.773 | 29.426 |
| Multi-start Laplace mixture importance | 0.16541711637219786 | 9.220 | 27.735 |
| Longer four-chain HMC | 0.16898421921218618 | 199.378 | 1,165.459 |

The mixture importance score is 5.5296% lower than Frontier and 2.8853% lower
than optimized MCMC on this screen. It takes approximately 5.1% more elapsed
time than MCMC in this single-pass comparison. Longer HMC is approximately
22.7 times slower than MCMC; its score is 0.7911% lower than MCMC.

The tuned importance proposal still has weaknesses: three of 150 asset-fit
calls exceed Pareto k 0.7, four have raw-weight ESS below 400, and the largest
normalized weight is 0.0707. Longer HMC passes the mixing thresholds across
the screen (maximum R-hat 1.00521, minimum bulk ESS 1,495.7 and tail ESS
1,200.1), but records eleven retained proposals with nonfinite or absolute
energy error above 1,000. These are diagnostics of numerical risk, not a proof
that rejected Metropolis proposals bias the chain. They preclude presenting
this configuration as cleanly validated posterior inference.

The split scores also caution against relying on a single aggregate: mixture
importance improves the first-quarter holdout but worsens the first-half
holdout relative to both comparators. All fifty tuned-screen comparator loss
vectors again match the archived full reference exactly.

## Wider screen: all eighty portfolios

All 400 specified origin tasks per arm completed, with all 701,280 daily
portfolio/horizon cells and all three holdouts. Every one of the 800 comparator
loss vectors matched the prior full canonical reference byte for byte. All
denominator and checkpoint-integrity checks passed.

| Arm | CRPS | Elapsed seconds | Process CPU seconds |
| --- | ---: | ---: | ---: |
| Frontier | 0.2558861226485342 | 186.022 | 218.106 |
| Optimized parameter MCMC | 0.2538154340980102 | 137.150 | 465.735 |
| Multi-start Laplace mixture importance | 0.25736549661342245 | 145.560 | 441.414 |

Mixture importance is approximately 0.5781% worse than Frontier and 1.3987%
worse than MCMC, with 6.13% more elapsed time than MCMC in this screen. Its
initial five-portfolio advantage did not generalize across the same eighty
portfolios. Runtime remains local diagnostic evidence.

| Split | Frontier CRPS | Optimized MCMC CRPS | Mixture importance CRPS |
| --- | ---: | ---: | ---: |
| Two rolling origins | 0.09438355911102274 | 0.0961901215870435 | 0.09568936549573662 |
| Train first quarter, test remainder | 0.24724610266385583 | 0.24681796188954458 | 0.25098420595598325 |
| Train first half, test remainder | 0.21438319460158262 | 0.20955905874356506 | 0.21439374739041153 |
| Train first three quarters, test final quarter | 0.10567009941498362 | 0.102738784196895 | 0.10248409138354594 |

Importance diagnostics across 2,400 asset-fit calls reach minimum raw-weight
ESS 97.1, maximum Pareto k 0.792, and maximum normalized weight 0.095. Thirty-six
fit calls exceed k 0.7 and fifty-one have ESS below 400. Calls can repeat a
shared asset/history across portfolios; the evidence also reports unique
training-history counts: 249 unique histories, with eighteen failing at least
one conservative diagnostic qualification. These are further reasons to avoid promotion.

The full-rank VI family was rejected for poor target coverage, shorter HMC for
poor mixing, and longer HMC for its large runtime cost and retained energy
errors. Only mixture importance justified the wider screen; it was then rejected
for worse aggregate accuracy. No new candidate advanced to the full fifty-one
origin schedule. This is a bounded test of six inference configurations, not a
claim that SMC, normalizing flows, alternative targets, or all possible inference
settings have been exhausted.

The aggregator records a conservative, post-screen diagnostic qualification:
stationary positive-curvature proposals with maximum gradient at most 0.01;
importance k at most 0.7, raw ESS at least 400, and maximum weight at most 0.02;
HMC R-hat at most 1.01, bulk and tail ESS at least 400, and no retained large
energy errors; VI success and maximum gradient at most 0.01. These thresholds
are qualification checks rather than prespecified statistical hypothesis tests.
They do not alter scoring or discard any forecast. Failed configurations remain
visible with their full scores and diagnostic failure reasons.

## Runtime and inferential qualifications

The local host uses the prior reference Python/NumPy/SciPy/Numba environment,
single-threaded BLAS, and the existing bounded asset and path concurrency.
Elapsed and process CPU time include all work performed by each arm.
HMC includes ArviZ diagnostics and trace retention, whereas the comparator
does not calculate those diagnostics. Pure VI also calculates 2,048 importance
draws solely for coverage diagnosis before discarding their weighted nodes.
These timings therefore do not isolate inference-kernel throughput and do not
establish an authenticated deployed-browser speed change.

The initial HMC diagnostic arrays were in unconstrained coordinates. Tuned
HMC diagnostics use physical target coordinates. Rank diagnostics are largely
invariant to monotone transformations, but folded components need not be;
the initial numbers are retained as measured rather than relabeled.

Training-only seeds and fits do not imply an independent validation sample:
these architectures and settings were selected after exploratory work on this
research panel. Another model-seed schedule, more posterior components, or a
new untouched validation panel could change the measured ordering.

## Reproduction and sources

See [the tools README](../tools/inference_candidates/README.md) for complete
settings and commands. Eight focused numerical tests validate the incumbent
target, transform Jacobian, analytic likelihood and ELBO gradients, and the
full-mixture importance density. Randomized gradient validation additionally
checks thirty interior parameter settings. The greatest observed target
difference is 2.28e-13; finite-difference gradient discrepancies are below
6.10e-7. This validates the target calculations, not posterior convergence.
The complete local test suite passes 192 tests with one distribution-metadata
test reserved for clean-wheel CI. Repository-wide Ruff checks pass. A dedicated
CI job installs ArviZ explicitly and runs the new numerical checks. After import
sorting, one published-source task was replayed with all three loss vectors
byte-exact against the wider screen; that is a bounded source check, not a
second full-panel execution.

[The evidence receipt](results/sv-inference-candidates/receipt.json) records
archive hashes, task counts, baseline matches, and diagnostic qualifications.
The [initial screen](results/sv-inference-candidates/screen25-results.json),
[tuned screen](results/sv-inference-candidates/tuned-screen25-results.json), and
[all-eighty screen](results/sv-inference-candidates/mixture-screen400-results.json)
retain exact scores and split diagnostics. The adjacent ZIP parts contain every
persisted task loss and its original manifest/diagnostic records, so aggregation
does not require trusting a printed score. Executed source archives and the
numerical environment are retained too. The initial inference overlay applies
on top of the complete tuned source archive for the initial screen's source
version; later aggregation fixes and diagnostic qualification do not change
any forecast.

To independently aggregate the saved wider-screen losses:

```sh
mkdir recovered-wide
for part in docs/results/sv-inference-candidates/mixture-screen400-part-*.zip; do
  unzip "$part" -d recovered-wide
done
python tools/inference_candidates/aggregate.py recovered-wide
```

Pass the original MCMC reference directory with `--reference` to repeat the
baseline byte comparisons. Without it, aggregation verifies all saved hashes
and scores, but reports zero newly checked baseline reference matches.

The implementation uses standard [HMC methodology](https://www.jmlr.org/papers/v15/hoffman14a.html),
but implements randomized fixed-trajectory HMC rather than NUTS. Its
[variational formulation](https://jmlr.org/papers/v18/16-107.html) uses a
full-rank Gaussian approximation. [Pareto importance diagnostics](https://www.jmlr.org/papers/v25/19-556.html)
are diagnostic only; predictive weights are not Pareto-smoothed.
