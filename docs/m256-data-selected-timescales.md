# M256: timescale-only controlled experiment

The best of five completed alternatives replaces the remaining fixed calendar
timescales with the persistence already estimated for each asset's conventional
volatility. It scores **0.22672308848311776**, compared with the saved optimized
M256 score **0.22687131940042085**: **0.065337% lower CRPS**.

This is a small empirical improvement on the existing research panel, not a
claim of statistical significance or new untouched validation. Production is
unchanged.

## Exact scope

M256 already simulates one stationary conventional AR(1) log-volatility state,
plus its dynamic-resolution rough overlay. The four remaining calendar scales
are used by the original volatility scaling of its **mean curve**. This
experiment replaces only their selection. The historical mean estimator and
the mean formula remain unchanged; their numerical output can change through
the newly selected timescales.

The same original component constructor receives the learned rates. Its EWMA
recursion, initialization, ridge, loading shrinkage, residual covariance and
caps remain intact. An EWMA is an exponentially weighted history: its fitted
persistence determines how quickly older observations lose weight. Its
half-life is `-log(2)/log(phi)`, with machine-level open-interval guards.

Everything else is retained: Student raw-return fitting, empirical return
innovations, stationary conventional volatility dynamics and normalization,
rough MAP inference and kernel accuracy tolerance, terminal latent-state
uncertainty, Student corrected-DCC return dependence, RNG schedules, clipping,
portfolio policies, frozen inputs and scoring. No MCMC or posterior compression
is added or removed.

## Complete results

Each candidate completed 80 portfolios, 48 rolling origins plus three temporal
holdouts per portfolio, 4,080 tasks and 701,280 equally weighted
portfolio-horizon cells, with 240 paths per origin. Scoring is the unchanged
terminal-log-return empirical CRPS, including its n-squared pair denominator.
M256 was not rerun; saved full-panel cells provide the comparator.

| Catalogue | Timescale selector | Full-panel CRPS | Change vs M256 | Wall seconds |
|---|---|---:|---:|---:|
| M256 | Saved four-calendar-scale reference | 0.22687131940042085 | Reference | 249.650 |
| M257 | Centered conditional lag-one regression | 0.22910715447179145 | 0.985508% worse | 223.981 |
| M258 | One rate fitted by conditional predictive proxy likelihood | 0.22983785139197935 | 1.307584% worse | 241.684 |
| M259 | Forward BIC-selected rates and count | 0.22981601892906445 | 1.297960% worse | 273.394 |
| M260 | Stationary AR(1) likelihood on the existing latent history | 0.22804312178001787 | 0.516505% worse | 249.561 |
| **M261** | **Reuse existing fitted SV persistence; zero extra fits** | **0.22672308848311776** | **0.065337% better** | **242.596** |

M261's initial complete run took 245.249 seconds. Its optimized fresh-cache
confirmation took 242.596 seconds, **2.8253% below the saved M256 time**. These
are separately scheduled measurements on the same hardware, numerical
environment and twelve-worker configuration; they do not establish a
controlled total-runtime ratio. A paired benchmark of the changed component
alone measured **58.514% less time**, with unchanged inputs and interleaved
observations. Most end-to-end cost remains in unchanged components.

## Validation and reproduction

All five raw task-vector sets were independently reconstructed into the
published cell arrays. Checkpoint checksums, finite values, full denominators,
portfolio/horizon ordering and aggregate scores passed. For M261, the runtime
optimization preserves **every one of the 4,080 complete loss vectors
byte-for-byte**, including the final 701,280 cells. Constructor tests compare
every numerical field against the retained reference, and scope tests verify
the original asset fit changes only its component dictionary.

The winner is a one-rate control with a **data-fitted timescale**; it does not
claim to select its component count dynamically. M259 separately tested
data-selected count and rates and scored worse. Proxy-likelihood controls are
conditional estimators, not a new jointly fitted Bayesian return model.

Run from the pinned source with the recorded numerical environment:

```sh
SIMFOLIO_TIMESCALE_VARIANT=existing_sv_rate \
PYTHONPATH=src:tools/mcmc_runtime \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
python tools/data_selected_timescales/run_panel.py \
  --output results/local/reproduction/existing_sv_rate-panel \
  --reference results/local/timescales/reference \
  --workers 12 --state-workers 1 \
  --model-id asset_m256_data_timescales_existing_sv_rate
```

The reference directory contains the saved M256 `results.json` metadata.
The output-parent data directory contains the unchanged verified canonical
input snapshot. The runner creates a candidate-specific fit cache. Other
selectors are `autocorrelation`, `predictive_one`, `predictive_adaptive` and
`stationary_one`; each runs in its own process/cache namespace.

Full source closures, task identities, environment, independent audits,
per-portfolio scores, timings and confirmation receipts are retained in
[`timescale-only-existing_sv_rate-panel.json`](results/rough-bayesian/timescale-only-existing_sv_rate-panel.json).
Sibling `timescale-only-*-panel.json` and `timescale-only-*-cells.npz` files retain
the other completed candidates. Both canonical and master catalogues retain all
five, including the negative results.
