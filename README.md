# Simfolio Forecasting Methodology

This repository is a standalone, public-safe research package for the
out-of-sample comparison of probabilistic portfolio forecasts. It contains
the canonical catalogue and protocol contracts, source-derived numerical
implementations that have passed bounded parity checks, and an offline frozen
data snapshot. It has no dependency on a production website, customer system,
private datastore, or operational API.

## Additional parameter-MCMC research

The [asset-level parameter-MCMC optimization report](docs/mcmc-runtime-optimization.md)
records a separate full-panel candidate, exact complete-output replay, matched
local runtime measurements, and reproducible experimental tools. It is not
registered in the combined catalogue or deployed as a production model.

The [alternative SV inference experiments](docs/sv-inference-candidates.md)
test Hamiltonian Monte Carlo, full-rank variational inference, and defensive
Laplace importance sampling against the same forecasting components. Their
screening scores remain separate from the canonical ranking.

## Current combined score ranking

The current public ranking is the combined 176-row numeric sort available from
`simfolio-oos candidate-scores --json`. It keeps historical evidence status
visible and adds the one independently audited new execution:

| Display rank | Model | Exact empirical CRPS | Evidence |
| ---: | --- | ---: | --- |
| 1 | Filtered-Innovation Moment-Matched SV with Fixed Historical Mean | 0.25439860867855635 | New, full canonical paired execution; independently audited |
| 2 | M001 — Asset FastMAP + Dynamic Gaussian Factor | 0.2557255171048505 | Retained historical evidence; not newly rerun |
| 3 | M002 — Bayesian SBB + Bias-Corrected SV-AR1 | 0.25582280588825473 | Retained historical evidence; not newly rerun |

The candidate's full paired run scored 4,080 tasks and 701,280 cells with 240
simulations per origin. Its score is lower than the separately audited Frontier
reference score of 0.25554347087969403, whose loss vectors were reused in that
paired report. Historical rank values remain in the immutable 175-model ledger;
the added candidate has no canonical rank. The mixed sort is descriptive and
does not mean all 176 rows were freshly run. See
[the candidate evidence report](docs/verified-candidate-results.md) for the
paired-run statuses, replay receipt, and selection qualification.
The [complete 176-row ranked JSON](docs/results/combined-176-score-ranking.json)
is available for inspection without installing the CLI.

## Current release state

The canonical ledger contains **175 exact model identities**. The public
runtime has **175 explicit executable factories**. Every model has passed
instantiation and bounded forecasting/source-parity checks. The numerical
implementations reuse the recovered statistical methods and their source
parameter defaults; no model was dropped or replaced with a generic baseline.

Historical score tokens are retained as evidence. Historical score linkage has
not been verified, and no retained score is presented as a newly reproduced
result. All 175 ledger entries have complete, fingerprinted statistical specifications.

The packaged canonical data snapshot contains 60 manifest-whitelisted files:
52 asset series, `EFFRX`, five additional canonical drift-proxy series,
and the French daily and Q5 factor inputs. Verification is offline and hash based. The frozen common
window is 1979-12-31 through 2026-05-13 with 11,687 dates. The protocol
constructs 80 portfolios, 4,080 origin tasks, and 701,280 scored cells per
model, with 240 simulations per origin.

## Install and inspect

The reference numerical environment is macOS ARM64 with Python 3.11 or 3.12
and `requirements-lock.txt`. Cross-platform byte equality is not established.
The optional `all-models` extra installs the numerical dependencies:

```bash
python -m pip install '.[all-models]'
simfolio-oos data verify --json
simfolio-oos data prepare --destination .simfolio-oos-data --json
simfolio-oos coverage --json
simfolio-oos canonical-175 --plan --json
```

The data commands use the packaged snapshot by default. Preparation writes a
caller-local execution cache after rechecking every source hash and derived
fingerprint; it never downloads or refreshes data. Use
`simfolio-oos scores --json` to inspect retained score evidence without
starting a forecast run. That command remains scoped to the original 175
historical records; use `simfolio-oos candidate-scores --json` for the current
combined ranking.

The bounded implementation surface can be exercised with an exact model ID or
the Frontier rank-one selector. A full canonical run requires the preserved
240 simulations per origin and is not implied by a smoke run.

Inspect the new candidate's task plan with
`simfolio-oos candidate --model experimental_filtered_innovation_moment_sv_fixed_mean --plan --json`.
The same exact selector runs it on the canonical dense plan when `--plan` is
omitted.

## Frontier production validation

The leading asset-level Frontier method matches the verified live numerical
source on three bounded cases. With identical random streams and production
storage precision, joint asset paths and rejoined portfolio paths match
exactly. The historical and production default seed schedules differ; the
[production parity report](docs/frontier-production-parity.md) records both
contracts, intermediate comparisons, and source hashes. This is bounded
numerical validation, not a rerun of the retained white-paper score.

## Reproducibility

Read [README_REPRODUCIBILITY.md](README_REPRODUCIBILITY.md) for the numerical
environment, source/data identities, parity checks, and known validation
limits. [docs/quickstart.md](docs/quickstart.md) shows a clean wheel install;
[docs/validation.md](docs/validation.md) describes the CI gates.
