# Exhaustive 52-asset catalogue campaign

This campaign ranks models using **only the 52-asset CRPS**. The separate,
historical 80-portfolio score is a production-validation gate, never a weighted
part of this ranking. No new 80-portfolio panel is run in this campaign.

The frozen population is in `inventory.json`: all 263 canonical records,
with only the four direct M261 timescale-selector siblings M257–M260 excluded.
There are 258 fresh evaluations and the independently revalidated saved M261
control (`saved-control-verification.json`). Old rankings and smoke scores
never prune the population. Missing historical evidence remains explicitly
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
