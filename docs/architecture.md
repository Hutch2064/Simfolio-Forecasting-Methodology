# Canonical research contracts

The public research scope is exactly the 175 identities in the canonical
ledger. A recovered parameter dictionary is not a complete numerical
specification. A model becomes executable only after its numerical dependency
closure and concrete factory are integrated and validated. Unsupported
identities must fail explicitly; names must never select substitute models.

## Numerical boundary

`TrainingData` contains portfolio daily log returns, optional aligned asset
daily log returns, ordered portfolio holdings and policy, and explicit training
dates. Source adapters requiring a calendar must reject an absent calendar.
`ForecastContext` carries the public model identity, portfolio identity, origin
label, optional explicit origin date, simulation count, horizon, seed, and future
dates. Historical seed identities and argument order belong to each verified
source adapter; presentation aliases cannot alter them.

Concrete model factories expose `simulate_daily_log_returns(training, context)`
or `simulate_terminal_log_returns(training, context)`. Both return finite
float64 matrices with simulations on axis zero and days on axis one. The
method name declares whether entries are daily increments or cumulative
terminal returns; the evaluator cumulatively sums only daily increments. A univariate portfolio fit cannot implement
the asset-level Frontier model. Frontier must fit asset marginals, construct
joint dependence, and apply the ordered portfolio policy and calendar.

For daily-path adapters, the evaluator constructs terminal log returns by
cumulative summation. It computes empirical CRPS with the ensemble-size-squared pairwise denominator.
Origin losses are averaged within portfolio-horizon cells before equal cell
weighting. Missing tasks and nonfinite cells must remain visible and cannot
reduce a successful result's denominator.

## Identity and evidence

The ledger is the sole membership and verification authority. The catalogue,
coverage report, and model reference are generated from it. Source references
use relative paths and content digests; source repositories and private caches
are not runtime dependencies. Numerical functions shared by multiple canonical
models have one implementation and explicit parameterizations.

Retained results are imported evidence, never forecasting output. Their genuine
serialized precision and source artifact identities must remain separate from
new bounded results and full-run checkpoints. Historical linkage that has not
been verified must remain marked unresolved.

The recovered dense adapter contains a direct daily-column selection in some
model branches. This differs from terminal cumulative scoring. Until the exact
scored adapter and this discrepancy are reconciled, retained-score equivalence
cannot be claimed for those branches. The executable terminal-return contract
above does not silently relabel those historical results as a validated rerun.

## Integration ownership

Ledger and registry changes are integrated independently of numerical families.
Frontier extraction owns its shared numerical dependency module. Later family
adapters must import verified shared functions rather than copy them. Data and
protocol work owns canonical calendars, portfolio policies, task construction,
fingerprints, and checkpoint identity. Family branches contribute explicit
per-model evidence patches for integration; they do not modify the authoritative
ledger concurrently or merge to main. Files in `audit/` retain extraction-stage
evidence; their historical staging flags do not override the integrated ledger
or the current `coverage` command.

## Optional source-fixture regeneration

The narrowly scoped development tools under `tools/` can regenerate historical
reference fixtures from an explicitly supplied, read-only original source
checkout. Those tools verify source digests before importing the numerical
reference. They are not shipped in the wheel or called by the research CLI.
Installation, bounded tests, data preparation, and forecasting use only the
public package and its bundled resources; they require no original source
checkout or private service.
