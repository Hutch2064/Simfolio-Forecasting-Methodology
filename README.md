# Simfolio Forecasting Methodology

This repository is a standalone, public-safe research package for the
out-of-sample comparison of probabilistic portfolio forecasts. It contains
the canonical catalogue and protocol contracts, source-derived numerical
implementations that have passed bounded parity checks, and an offline frozen
data snapshot. It has no dependency on a production website, customer system,
private datastore, or operational API.

## Current release state

The canonical ledger contains **175 exact model identities**. The public
runtime currently has **157 explicit executable factories**: 84 canonical base models,
40 full MCMC SV models, 32 portfolio reference models, and Frontier. Those 157
factories have passed package instantiation and bounded forecast/source-parity
checks. The other ledger rows remain registered evidence identities and fail
closed when execution is requested.

Historical score tokens are retained as evidence. Historical score linkage has
not been verified, and no retained score is presented as a newly reproduced
result. The ledger's full statistical specification gate remains open.

The packaged canonical data snapshot contains 60 manifest-whitelisted files:
52 asset series, `EFFRX`, five additional canonical drift-proxy series,
and the French daily and Q5 factor inputs. Verification is offline and hash based. The frozen common
window is 1979-12-31 through 2026-05-13 with 11,687 dates. The protocol
constructs 80 portfolios, 4,080 origin tasks, and 701,280 scored cells per
model, with 240 simulations per origin.

## Install and inspect

The optional `all-models` extra installs the numerical dependencies used by the
validated source-derived paths:

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
starting a forecast run.

The bounded implementation surface can be exercised with an exact model ID or
the Frontier rank-one selector. A full canonical run requires the preserved
240 simulations per origin and is not implied by a smoke run.

## Reproducibility

Read [README_REPRODUCIBILITY.md](README_REPRODUCIBILITY.md) for the numerical
environment, source/data identities, parity checks, and known validation
limits. [docs/quickstart.md](docs/quickstart.md) shows a clean wheel install;
[docs/validation.md](docs/validation.md) describes the CI gates.
