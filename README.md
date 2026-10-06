# Simfolio Forecasting Methodology

This repository is a standalone, public-safe research package for the
out-of-sample comparison of probabilistic portfolio forecasts. It contains
the canonical catalogue and protocol contracts, source-derived numerical
implementations that have passed bounded parity checks, and an offline frozen
data snapshot. It has no dependency on a production website, customer system,
private datastore, or operational API.

## Current catalogue and score ranking

The canonical catalogue contains **193 models**: the original 175 plus eight
asset-level models with validated full-panel scores, four partially scored
candidates, and six unscored asset-level rough-volatility candidates. The
dynamic-resolution candidate completed the full panel with CRPS
**0.25074679212444156**, the lowest catalogue score. Six canonical rough-volatility
alternatives remain unscored. Their [implementations, results and diagnostics](docs/rough-bayesian-oos.md) are saved. The current production
Frontier and both previous Frontiers are included. The original model IDs,
historical ranks and score tokens are preserved; additions have canonical
indices 176–193 and no historical source rank.

`simfolio-oos candidate-scores --json` returns all 193 models once, sorted by
exact empirical CRPS, with unscored full-panel rows last and unranked.
[Partial candidate results](docs/rough-jump-vine-oos.md) retain completed
portfolio scores and blank unfinished portfolios. The [complete ranked JSON](docs/results/combined-176-score-ranking.json)
and [model reference](docs/canonical-model-reference.md) are also available.
Existing command names and catalogue filenames remain compatible.

| Model | Exact empirical CRPS | Role |
| --- | ---: | --- |
| Asset MAP Multiscale + Dynamic Rough Volatility | 0.2518812750312293 | Validated fast research candidate |
| Asset Rough Volterra SV — Eight Factors | 0.2511998559613309 | Validated research candidate |
| Asset Parameter MCMC + Filtered Moment SV | 0.25246784959071183 | Current production Frontier |
| Asset Filtered Innovation Moment SV + Fixed Mean | 0.25439860867855635 | Previous production Frontier |
| Asset Filtered Innovation Moment SV + DLM | 0.2547972662964723 | Validated candidate |
| Asset FastMAP + Dynamic Gaussian Factor | 0.2557255171048505 | Earlier production Frontier; retained historical score |
| Asset Gaussian Moment SV + Fixed Mean | 0.2588375710788695 | Validated candidate |

Each fully scored addition was evaluated on 4,080 origin tasks, 701,280 cells and 240
simulations per origin using the whitepaper's exact empirical CRPS and equal
portfolio-horizon cell weighting. Complete statistical specifications, source
hashes, run manifests and validation references are recorded per model.

The public package has **176 explicit executable factories**. Other entries
retain their standalone research entrypoints and source details.
The [parameter-MCMC report](docs/mcmc-runtime-optimization.md) includes the
current Frontier's complete-output replay and optimized numerical sources.
The [alternative SV inference experiments](docs/sv-inference-candidates.md)
retain their screening results separately.

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
starting a forecast run. That command includes the expanded canonical records; use `simfolio-oos candidate-scores --json` for the current
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
