# MCMC runtime research

These tools preserve the scored two-chain, sixteen-node asset-level SV
parameter-MCMC candidate. They do not replace either production forecasting
model or change the canonical scoring protocol. See
[the numerical/runtime evidence report](../../docs/mcmc-runtime-optimization.md).

The target is the existing Gaussian log-square SV quasi-posterior, with the
explicit parameter support inherited from the candidate's optimizer-domain
bounds. The latent mean stays fixed at the historical asset mean. Parameters
are sampled with two independent Metropolis chains: 1,024 burn-in and 2,048
retained draws per chain, with covariance adaptation only during burn-in.
Sixteen equally weighted, evenly spaced retained parameter draws supply
conditional smoothed volatility paths. Forecast moments are matched across
these sixteen component laws. This is parameter MCMC with moment-matched
prediction; it is neither PGAS nor a full joint parameter/state posterior
predictive mixture.

The execution changes preserve the original target, seeds, transition order,
adaptation schedule, draws, quantile nodes, copula, calendar, and turnover
cost. Fixed-point reuse requires exact floating-point equality. Temporary
arrays are bounded using 1,024-day draw blocks and eight-simulation rejoin
blocks, without reducing observations, simulations, posterior nodes, or
horizons. Independent simulation paths use at most four numerical threads.
For forecasts spanning more than one draw block, a single producer owns the
Generator and prepares the next block while the current block is computed.
Long forecasts also rejoin independent paths in parallel; single-block
forecasts avoid that overhead. Asset fitting uses two workers for fewer than 2,048 observations and six
otherwise, capped by the available CPU count and number of assets. At least
8,192 observations enables concurrent independent chains, with the asset
worker cap divided by two. With automatic settings, at most twelve numerical fitting threads run
per model process on the reference host. Pure compiled smoothing kernels
release the GIL. These are execution choices, not
additional forecasting parameters. `OptimizedMCMC(fit_workers=1)` gives the
serial-asset execution path.

Initialization installs numerical callbacks in the dedicated research process
and caps the calling thread's Numba thread mask at four. That mask remains in
effect for subsequent Numba work in the same process.

## Environment and data

Use the repository's Python 3.12 requirements and optional numerical packages.
The reference measurements use macOS ARM64, Python 3.12.13, NumPy 2.5.3, SciPy
1.18.1, and Numba 0.67.0. Exact cross-platform equality is not asserted.

```sh
python -m pip install '.[all-models]'
simfolio-oos data prepare --destination .simfolio-oos-data --json
export SIMFOLIO_OOS_DATA="$PWD/.simfolio-oos-data"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=4
export PYTHONPATH="$PWD/src"
```

Generate the original paired Frontier/MCMC reference on all eighty portfolios,
all fifty-one origins, 240 simulations, and the full daily-horizon schedule:

```sh
python tools/generate_mcmc_reference.py --workers 3 --output mcmc-reference
```

Run full optimized parity and scoring. It requires the original reference
checkpoints and enables complete-output verification automatically:

```sh
export SIMFOLIO_MCMC_REFERENCE="$PWD/mcmc-reference"
export SIMFOLIO_MCMC_OUTPUT="$PWD/mcmc-optimized-parity"
python tools/mcmc_runtime/replay_full_panel.py
```

This replay validates every retained chain draw/acceptance rate, MAP fit,
multiscale fit dictionary, dependence posterior, copula uniform, sorted
marginal value, mapped asset path, daily portfolio path, and CRPS loss vector.
It then applies the unmodified public evaluator and denominator gates.
Verification time includes original calculations and comparisons; it is not
a runtime benchmark. The replay uses four bounded processes; numerical BLAS
threads remain fixed at one. Its concurrency is recorded in the source manifest.

Benchmark separately while the machine is otherwise idle:

```sh
python tools/mcmc_runtime/benchmark_model_cases.py
python tools/mcmc_runtime/benchmark_scaling.py
python tools/mcmc_runtime/profile_runtime.py
python -m pytest tests/test_mcmc_runtime_optimization.py -q
```

Benchmarks interleave original MCMC, optimized MCMC, and Frontier; they warm
compilers and clear fitted-model caches before every measured arm. Optimized
and original MCMC daily paths are compared in full. All these measurements are
local model diagnostics, not authenticated production-browser latency.
