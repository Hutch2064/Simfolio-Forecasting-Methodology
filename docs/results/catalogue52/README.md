# Exhaustive 52-asset catalogue campaign

This campaign ranks models using **only the 52-asset CRPS**. The separate,
historical 80-portfolio score is a production-validation gate, never a weighted
part of this ranking. No new 80-portfolio panel is run in this campaign.

The frozen population is in `inventory.json`: all 263 canonical records,
with only the four direct M261 timescale-selector siblings M257–M260 excluded.
The initial population contains 258 fresh evaluations and the independently
revalidated saved M261 control (`saved-control-verification.json`). The later
user-authorized scope amendment is recorded separately in `screening.json`;
the frozen inventory and historical catalogue definitions remain intact.
Retain broad methodological coverage and at least 100-plus models: remove only
clear nonunique variants or disproportionate runtime outliers without leading
evidence. Old portfolio rankings alone do not establish single-asset inferiority.
M003, M006, M262 and M263 are user interests, not a narrowed shortlist.
Missing historical evidence remains explicitly
unqualified; a 52-asset result alone cannot establish production eligibility.

Each fresh model evaluates all 2,652 origins at 240 paths, preserving frozen
price-derived asset logs, origin dates and seeds, temporal holdouts, and every
eligible daily horizon. Exact terminal-log-return empirical CRPS is averaged
over origins within each asset/horizon, then equally over 455,832 cells. Partial
panels and smoke tests cannot publish a full score.

The naive reference is the retained whitepaper joint raw-return stationary
bootstrap, including its original return/squared-return block selector and
holdings policy. The original catalogue's IID model remains a distinct tested
model; it is not renamed or substituted for this reference.

`tools/catalogue52/campaign.py` controls independent processes, immutable source
snapshots, model-specific environments and atomic origin checkpoints. Live
additions start the entire same grid while other models continue. Cooperative
removal saves the current origin; explicit force removal records interruption
and affects only the verified model process group. Changed implementations use
new execution identities. No automatic wall-clock cutoff rejects long fits.

Expensive controls can evaluate independent origins in separate processes,
within a total twelve-slot CPU budget and reduced scheduling priority. The
original model, sampling effort and per-origin seeds stay fixed. Six-way
execution registrations were prepared for M184–M189, but the subsequent
runtime screen removed both execution forms before their full panels began.
Their implementations and measured evidence remain saved. The bounded serial/parallel
proof, including identical selected loss vectors and all 2,652 task identities,
is in `origin-parallel-proof.json`.

Queued original full-latent-MCMC models can use `latent_exact_filter.py`, which
compiles only their unchanged deterministic Kalman recurrence. The sampler,
FFBS state draws, adaptation, stopping, seeds and original model factories stay
fixed. `latent-filter-proof.json` records byte-exact complete fits, daily and
cumulative paths and CRPS on twelve cases, plus an actual campaign-worker loss
vector comparison. A representative warmed origin was 2.47 times faster;
that is a measured case, not a blanket full-panel runtime claim. Only executions
that had never started were replaced; active and completed runs were untouched.

The retained rough controls M191/M192 use four independent origin lanes each,
within that same twelve-slot budget. Their serial executions stopped after
checkpointing the current origin; new executions run the entire unchanged grid.
`retained-rough-parallel-proof.json` records the selected loss-vector and
inference-diagnostic parity gate. All old checkpoint evidence is retained for
the subsequent complete-prefix comparison. The controller reserves released
slots for the first queued eligible origin pool, so repeated admission of small
jobs cannot defer it to the end. This can briefly leave slots idle while other
running models finish; it never interrupts those models.
M255/M256 are also scheduled earlier using their original frozen source and
full grids, to measure their actual single-asset cost before they become a late
campaign tail. Queue order changes neither their scores nor the retained list.

Periodic top-ten updates combine completed campaign results with previously
validated results on this same 52-asset grid, including M262 and M263. Updates
are approximately fifteen minutes apart, or earlier for a new leader, and
include the entire campaign's remaining-time estimate. Runtime estimates based
on evaluator worker time remain distinct from matched latency benchmarks.

The original materializer creates a shared exact input cache once. Its roundtrip
proof covers all 2,652 identities, 13,260 arrays, bytes, strides, writeability and
buffer aliasing (`preprocessing-parity.json`). Every load checks source,
environment, prepared-input and payload hashes. No statistical approximation
or calendar shortcut is introduced.

The older rough suites rejected one-dimensional dependence. Their bridge uses
the exact identity copula C(u)=u, preserving their candidate objects, mean,
volatility and jumps, while delegating multidimensional calls unchanged. See
`single-asset-bridge-receipt.json`. Historical M178/M179 fitter substitutions
were recovered with exact source hashes and compared byte-for-byte to native
branches (`legacy-source-branch-parity.json`); reference text is retained under
`tools/catalogue52/reference_legacy/`.

Active inputs/checkpoints are temporary local working files. Completed model
outputs and execution envelopes are archived losslessly to the approved private
research archive, freshly downloaded and verified member-by-member before
redundant raw local files are retired. Public score summaries and source are
published here; authenticated recovery locators remain in the storage ledger.
Archived runs require explicit restoration before resume. Scheduling priority
and bounded worker counts preserve local interactivity; the campaign owns a
20-GiB total storage budget with at least 100 GiB free-space reserve.

The campaign remains open until every included model or justified addition has
a complete result or a documented runtime removal/genuine implementation
blocker after reasonable recovery. Final reporting includes the full ranking,
runtimes, additions/removals, historical validation, and long-horizon CAGR
diagnostics for relevant leaders. Valid completed results are published in both
canonical and master catalogues. Production is unchanged by this research.

`scores.json` is the live campaign report; its progress section is not a full
score. The independent, packaged result index is
`src/simfolio_forecasting_methodology/resources/results/single52.json`.
It contains full 52-asset results plus the separately verified saved M261 control.
Each fresh result retains its explicit raw-archive verification state; pending
archival does not hide a valid completed score. Partial executions do not enter
this index. A subset of completed models does not imply that the catalogue-wide
campaign is complete.

After generating the public report, mirror the validated completed results to
both catalogues with:

```bash
python tools/catalogue52/sync_catalogues.py --master-root ../master
```

The helper validates immutable model IDs/ranks, specification fingerprints,
52 assets, 2,652 tasks, 455,832 cells, 240 simulations, finite scores and evidence
hashes before writing identical portable JSON to the packaged methodology
resource and `catalogs/single52_results.json` in the master checkout. It retains
every completed row's evidence fields and the source report's SHA-256. A result
declaring a verified raw archive must include its archive and receipt hashes;
a pending archive remains explicitly pending. The
historical80 score and its verification status remain separate from single52;
no blended score, catalogue membership change or specification edit occurs.
Refreshing the live report requires another explicit synchronization.

The independent loaders are
`simfolio_forecasting_methodology.single_asset_results.load_single52_results()`
and `simfolio_forecasting.single_asset_results.load_single52_results()` in the
master repository. Both return a dictionary keyed by immutable model ID, without
merging result fields into model metadata. The master manifest, canonical score
CSVs, methodology ledger and all existing model fingerprints remain unchanged.
