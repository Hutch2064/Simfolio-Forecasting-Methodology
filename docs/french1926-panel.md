# French long-history single-series panel

This is a **new, unscored panel**, separate from canonical52 and historical80.
No model is registered, selected, or run by data preparation. Model selection
and execution require an explicit subsequent research decision.

## Frozen return universe

The common observed calendar is July 1, 1926 through August 31, 2026:
26,317 daily observations per series, including historical Saturday trading.
There are **552 independently tested input series: 551 equity portfolios and
one risk-free cash control**. Each is supplied as one return column at unit
weight; no combinations of these series are constructed.

| Family | Unique daily equity series |
| --- | ---: |
| Size | 36 |
| Book-to-market | 36 |
| Size by book-to-market | 186 |
| Short-term reversal | 78 |
| Industries | 214 |
| Broad U.S. market total return | 1 |
| **Equity total** | **551** |

Both published value- and equal-weighted portfolio returns are included.
Numerically identical histories across files are deduplicated by their entire
common-window float64 return vectors. These are distinct return histories,
not an assertion of statistically independent exposures. Portfolio construction
and constituent rebalancing are already embedded in the provider's returns.

The exact selection, rejected columns, source ZIP byte hashes, aliases and
calendar/missingness audit are retained in
`resources/french1926/source_audit.json`; the 550 selected portfolio columns
are listed in `equity_series.csv`. Broad-market total return is computed as
**Mkt-RF + RF in simple percent units**, then converted with `log1p(r / 100)`.
RF is converted in the same units and labelled `cash_control`. Factor spreads,
firm counts, monthly/annual tables, ex-dividend series and later-starting
families are not inputs. No missing returns are forward-filled, back-filled
or interpolated. Dates align exactly to the market source calendar.

Source: [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).
The frozen vintage is essential: changed provider ZIP hashes fail preparation
rather than silently refreshing an existing benchmark.

## Scoring contract

The implementation imports the existing whitepaper `origin_schedule`,
`coherent_daily_seed`, and `empirical_crps_by_horizon` functions. The paper
source is the 38-page repository PDF at web revision
`11ed29af89100ef1c5419e8f2234a947dfa90443`, SHA256
`a19eae33d5f5f5591e95449d2f43a15bf5c33433833f94208deeb9c3f5423fb0`.

- 48 approximately equally spaced eligible calendar-quarter-end origins,
  plus the three 25%, 50% and 75% historical training partitions.
- Rolling origin position >= 504 means at least 505 training observations.
  Rolling forecast horizons are every observation from 1 through
  `min(remaining_history, floor(training_observations / 4))`.
- Each temporal partition forecasts its complete remaining history.
- Exactly 240 coherent paths per origin, evaluated at every eligible daily
  horizon, with the existing origin-date/horizon/simulation seed derivation.
- Target: cumulative **log return**, not wealth, simple return or CAGR.
- Exact empirical CRPS, including all self-pairs in the **240 squared** pairwise
  denominator. No historical adjusted/fair-score substitution.
- Average origin losses within each series × horizon cell, then equally
  average all cells. The cash control participates in this all-552 aggregate;
  role/family slices remain separately available from saved origin vectors.
- Strict finite outputs and complete task and per-cell-origin counts are
  required for a final score. A partial score covers only named complete series.

The resulting frozen plan has **28,152 origins** and **10,895,376 series ×
horizon cells**, with a maximum of **19,738 future observations**. Old 52/80
panel counts, dates and maximum horizon are not substituted into this plan.
Training and realizations use observed French dates. As in the existing
harness, the model's future policy calendar uses synthetic weekdays after
the origin; with one holding and no outer rebalancing it adds no basket trades.

## Preparation and readiness

From the methodology repository, with its Python dependencies installed:

```sh
python -m simfolio_forecasting_methodology.french_panel prepare /path/to/french1926
python -m simfolio_forecasting_methodology.french_panel validate /path/to/french1926
```

Preparation downloads only the 17 checksum-pinned public source ZIPs and
materializes a read-only memory-mapped float64 input matrix, calendar, manifest
and value-free origin schedule. It imports no model adapter. Validation checks
the packaged frozen definition, input hashes, dimensions, dates and schedule.

For future model adapters, the execution entry point is
`simfolio_forecasting_methodology.french_panel.runner.run_panel`. Its adapter
interface is the existing `ForecastModel`/`TrainingData`/`ForecastContext`.
Only immutable historical input copies reach the adapter; held-out returns
remain inside the scorer. Atomic checksummed per-origin checkpoints bind
the panel, source closure, implementation and runtime identities. Completed
valid origins resume unchanged; model errors are recorded separately so the
remaining work can proceed. Changed identities and corrupt checkpoints fail
closed. Mean/median path CAGR versus historical training CAGR is saved as a
diagnostic, not a model exclusion rule. Per-origin forecast/scoring compute
time is saved separately from score and checkpoint I/O.

## Live sweep controls

Copy `resources/french1926/sweep_control.json` to an owned run directory. Its
initial model map is empty. The controller accepts explicit future adapter
factories and implementation hashes only when models are subsequently chosen.
It supports at most 12 worker processes and checkpoints each completed origin.
Enabled additions enter the waiting list without restarting other models.
Disabled or removed models stop before their next origin and retain their
saved checkpoints; re-enabling the identical implementation resumes them.

```sh
python -m simfolio_forecasting_methodology.french_panel.sweep \
  --panel-root /path/to/french1926 --output-root /path/to/sweep \
  --control /path/to/sweep/control.json
python -m simfolio_forecasting_methodology.french_panel.sweep \
  --output-root /path/to/sweep --status
```

Live status includes separate mean- and median-based remaining-time estimates
using the completed models' saved forecast/scoring compute totals and current
worker allocation. The sample count and assumptions are visible; neither ETA
is populated without completed-model timing evidence. These are compute-based
planning estimates, excluding checkpoint I/O and assuming comparable future
models, rather than measured website latency. Completed-model rankings never
mix in incomplete scores.

## Preservation

The source code, selected-column inventory, frozen panel definition, origins,
validation receipt and archive receipt belong in the public methodology
repository, with a matching panel index in the master research repository.
Raw provider ZIPs and normalized inputs use the existing private research
release process. The archive receipt records a downloaded checksum-verified
copy, every restored member hash and a successful restored-panel validation.
Only redundant temporary inputs and staging archives may be removed after
that check. Source code can reconstruct the panel from official ZIPs while
their frozen hashes remain available; the archive preserves this exact vintage.

Neither historical panel nor any model ranking is replaced by creating this
panel. Its model registry and result list remain empty.

## Proposed model list, not an executable registry

`resources/french1926/shortlist_proposal.json` preserves one proposed list:
31 candidate methods and two required comparison controls, for 33 methods.
It applies the user's 52-asset bound (no more than 10% worse than M261),
method-family reduction, and the subsequent exclusion of candidates with
saved historical80 scores worse than website Default. Unknown historical80
comparisons remain unknown and retained. No runtime-percentile exclusion was
requested or applied. Every omitted identity remains in historical catalogues.

The two mandatory controls are the exact website Default
`asset_level_naive_iid_mean_shrunk_rebalanced` and the paper's Testing comparator
`model_testing_joint_stationary_bootstrap_v1`. Their source identities are
recorded separately; adapter parity must be verified when models are wired.
The standalone website Default has no completed canonical52 score attached
here; the earlier 50/50 mixture cannot substitute for that missing result.

The slow-tail table and saved historical80 comparison are separate screening
records. Historical scores have not been promoted into results on this new
panel. The controller's executable model map is still empty.
