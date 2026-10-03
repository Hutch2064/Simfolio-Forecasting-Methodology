# Alternative SV inference experiments

Research-only alternatives for the existing bounded, three-parameter
Gaussian log-square SV quasi-posterior. These tools change parameter
inference, while retaining the optimized MCMC adapter's sixteen-component
moment-matched forecast, empirical innovations, four volatility scales,
historical mean, asset-level dependence, calendars, and transaction costs.
They do not register a production model or change the canonical evaluator.

See [the experiment report](../../docs/sv-inference-candidates.md) for results
and limitations. SMC means sequential Monte Carlo; HMC means Hamiltonian Monte
Carlo. The experiments here implement HMC, full-rank Gaussian variational
inference, and importance sampling. They do not implement SMC, NUTS, PGAS, or
normalizing flows.

## Run

Use the repository's pinned Python 3.12 reference environment. Install the
additional diagnostic dependency and prepare the exact public data snapshot:

```sh
python -m pip install -r tools/inference_candidates/requirements.txt
simfolio-oos data prepare --destination .simfolio-oos-data --json
export SIMFOLIO_OOS_DATA="$PWD/.simfolio-oos-data"
export PYTHONPATH="$PWD/src"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=4
python tools/inference_candidates/validate.py
python -m pytest tests/test_inference_candidates.py -q
```

The first screen uses five portfolios and five origins per portfolio,
including all three chronological holdouts. The wider screen uses all eighty
portfolios at those five origins. Neither is the full 4,080-task canonical
experiment, even though the wider screen reaches all 701,280 horizon cells.

```sh
python tools/inference_candidates/run_study.py --stage screen25 \
  --methods laplace_is,full_rank_vi,vi_importance,hmc --output initial-screen
python tools/inference_candidates/aggregate.py initial-screen
python tools/inference_candidates/run_study.py --stage screen25 \
  --methods laplace_mixture_is,hmc_long --output tuned-screen
python tools/inference_candidates/aggregate.py tuned-screen
python tools/inference_candidates/run_study.py --stage screen400 \
  --methods laplace_mixture_is --output wide-screen
python tools/inference_candidates/aggregate.py wide-screen
```

Use `--stage full` for all eighty portfolios and all fifty-one origins. Each
stage uses 240 paths and the unchanged public evaluator. Aggregation validates
persisted checkpoint hashes and loss vectors before scoring; no failing fit
or task is dropped. To verify both comparators byte for byte against the
original full reference, pass `--reference /path/to/mcmc-reference` to the
aggregator. Generate that reference with the existing
`tools/generate_mcmc_reference.py` tool if needed.

## Inference settings

The support, target priors, and physical-parameter Jacobians are inherited
from the previous MCMC candidate. An additional logistic transform maps the
bounded parameters to unconstrained coordinates, retaining its Jacobian.
The Kalman likelihood and its analytic sensitivities are compiled in Numba.
All seeds derive from training bytes; future returns and OOS scores never
enter an asset fit. Candidate settings were selected during exploratory
research on this panel; the scores are not independent confirmation.

* `laplace_is`: Hessian-based proposal, Student-t with five degrees of freedom,
  2,048 importance draws, sixteen systematically resampled parameter nodes.
* `full_rank_vi`: three-dimensional Gaussian approximation with unrestricted
  Cholesky covariance; frozen sixty-four scrambled Sobol nodes for the ELBO,
  at most one hundred optimizer iterations, sixteen direct Gaussian draws.
* `vi_importance`: same variational approximation, with 2,048 importance draws
  and sixteen systematically resampled parameter nodes.
* `hmc`: four chains, 384 warmup and 512 retained draws per chain; fixed
  Laplace whitening, four to twelve randomized leapfrog steps, dual averaging
  to 0.8 target acceptance during warmup only. Sixteen evenly spaced pooled
  retained draws provide forecast components.
* `hmc_long`: same HMC with 1,024 warmup and 2,048 retained draws per chain.
* `laplace_mixture_is`: five training-only optimizer starts (the existing MAP
  center plus persistence starts 0.1, 0.5, 0.95, 0.995 at innovation scale
  0.35); distinct stationary points supply equal-weight Student-t proposals,
  each at Hessian scale and twice that scale. A total of 4,096 importance
  draws use the density of the entire proposal mixture, followed by sixteen
  systematic-resampling nodes.

These are explicit numerical/inference choices, not parameters estimated
from future observations. The fitted level, persistence, and innovation scale
remain asset-specific random quantities. Laplace covariance eigenvalues are
floored at 1e-5; optimizer coordinates are bounded to [-18,18]. These numerical
safeguards can produce poor proposals, which must be assessed with diagnostics.

## Diagnostics and runtime interpretation

Importance diagnostics use raw-weight ESS, largest normalized weight, and
Pareto k. Pareto smoothing is **diagnostic only**: predictive resampling uses
the original importance weights. HMC reports rank-normalized R-hat, bulk and
tail ESS, acceptance, and retained proposals with nonfinite or absolute energy
error greater than 1,000. These diagnostics assess finite computations; they
do not prove exact posterior coverage.

Runtime is local elapsed and process CPU time, with warm compilers and cold
model fits for every arm, alternating arm order, on one host. Fits use the
existing bounded asset threading. This is not deployed browser CTV. Each
task is timed once, so differences are screening evidence rather than a
repeated median speed benchmark. HMC timing includes ArviZ diagnostics and
trace retention. Pure VI timing also includes a 2,048-draw importance diagnostic
whose weighted nodes are unused in its forecast. The baselines do not perform
those additional diagnostics. Do not interpret these figures as isolated
sampler throughput or use them to claim a production speed improvement.

The forward distribution is still moment-matched from conditional smoothed
state means. Replacing the inference algorithm does not turn it into an exact
joint latent-state posterior predictive simulator.
