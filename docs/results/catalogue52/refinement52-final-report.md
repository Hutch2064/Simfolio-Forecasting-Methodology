# Completed 52-asset refinement

The selected candidate improves the complete 52-asset CRPS by **2.9482% versus
production M261** and **1.2497% versus production with rough volatility removed**.
Its score is 0.5957716608071159 over 2,652 original origins, 240 paths per origin
and 455,832 equally weighted asset/horizon cells. No dates, seeds, data, path
counts or scoring weights changed.

The model retains the existing fitted mean and evolving variance curve, empirical
Student shock nodes, and accounting. Starting from rough-off, it removes the
future conventional stochastic-volatility multiplier, then corrects the variance
of the continuously interpolated shock-node distribution while preserving its
existing mean. The continuous moments are analytic, not fitted on held-out
returns. This adds no Sharpe cap, blend weight, ticker rule or forecast-horizon
switch. The exact calendar/accounting adapter avoids redundant single-holding
work while retaining the generic multi-asset fallback.

## Score and breadth

| Candidate | Complete 52-asset CRPS | Better than production | Better than rough-off |
| --- | ---: | ---: | ---: |
| Production M261 reference | 0.6138695810898457 | reference | - |
| Optimized rough-off reference | 0.6033109364170391 | 1.7200% | reference |
| Deterministic conventional RMS | 0.5968189282792725 | 2.7776% | 1.0761% |
| Selected RMS plus variance-only node correction | 0.5957716608071159 | 2.9482% | 1.2497% |

The selected candidate improves on rough-off for 24 of 52 assets, with positive
aggregate improvement after removing any one asset. It wins on 45 of 52 assets
against the public naive benchmark overall. The extra node correction improves
all nine reported horizon bands versus RMS and wins on 26 of 52 assets. Most of
the RMS aggregate improvement is at long horizons; precious metals contribute
a large share. This is a real aggregate improvement, not universal asset gains.

## Measured speed and exactness

The completed matched study uses six prespecified canonical contexts, four
models and three fresh processes per context/model: 72 trials and 288 timed
samples. JIT warmup uses another asset; input fitting is fresh for the first
sample, then three cached-fit samples clear the path cache. Common priority and
one numerical thread apply while the independent main sweep runs. Median of
six context ratios:

| State | Faster than production | Faster than rough-off |
| --- | ---: | ---: |
| Fresh fit | 2.8646x | 1.7000x |
| Cached fit | 4.1387x | 2.9260x |

These measure local forecast compute, not website click-to-visible time or an
all-52 average. The variance-only node change has approximately the same compute
cost as RMS; both methodological simplification and exact implementation work
contribute to speed versus the references.

The optimized selected implementation was checked across **all 2,652 origins**:
937,010,880 complete daily path values and 3,904,212 CRPS entries match the
unoptimized selected model byte for byte. Calendars, context identities,
finiteness and restoration of temporary hooks pass with zero unresolved failures.

## Growth and validation limits

At the longest scored 8,766-day horizon, 46 assets have positive historical CAGR.
The median forecast-median/historical ratio is 0.74959 and forecast-mean/historical
ratio is 0.75375, similar to production and RMS. None falls below half of positive
historical CAGR or becomes nonpositive. The inherited roughly 25% conservatism
remains visible. This is a diagnostic, not an imposed 75% acceptance rule. All
156 long-context growth summaries are retained. A historical-growth floor was
not added: moving forecasts closer to history worsened predictive scores.

**The selected candidate has not run the 80-portfolio panel and is not deployed.**
The historical rough-off reference completed that separate panel 1.6730% worse
than M261; this is not the new candidate's score. Research reused the same panel
for methodological selection; post-search independent validation is not claimed.

## Why the search stops here

Every evidence-supported branch and its justified interactions completed. We
tested conditional-mean versus RMS integration, original versus coupled variance,
continuous node center/spread separately and jointly, transient fitted growth
with and without uncertainty, stationary residual resampling with both original
and corrected nodes, its signed-only block selector, and direct retained-Student
variance/mean laws. Rejected candidates and failed-start diagnostics remain in
the evidence. Detailed scores and the untested alternatives are in
`STOPPING_LEDGER.md`. The latest completed-sweep scan supplied no better mechanism.

More fitted blend weights, asset switches, cutoff horizons or unsupported flag
combinations would now be speculative. This is a bounded stopping judgment on
the available evidence, not a claim that all forecasting ideas are exhausted.
The separate full catalogue sweep and its archiver remain autonomous and running.

## Preservation

The complete evidence (98,980 files; 3,193,674,770 logical bytes) is preserved in
a checksum-pinned [private archive receipt](https://github.com/Hutch2064/simfolio-engine/releases/download/research-catalogue52-20261018/refinement52-3811e86e6c93aa0e6c40.receipt.json).
Every uploaded part and every member passed independent fresh-download hash,
file-mode and readability checks. The actual downloaded archive was fully
restored, and the selected model reproduced four complete 240-path forecasts
byte for byte against the archived benchmark oracles. The separate
[scientific recovery receipt](https://github.com/Hutch2064/simfolio-engine/releases/download/research-catalogue52-20261018/refinement52-3811e86e6c93aa0e6c40-scientific-smoke-05def275e3561204.json) was uploaded and freshly downloaded with identical bytes.
Both catalogue indexes retain the complete result set and exact user objective.
Archive recovery and forecast checks do not constitute a clean-host dependency
installation, a new independent evaluation panel or production deployment.
