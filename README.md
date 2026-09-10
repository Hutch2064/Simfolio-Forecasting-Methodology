# Simfolio Forecasting Methodology

This repository is a standalone, public-safe research package for the
out-of-sample comparison of probabilistic portfolio forecasts. It contains
the canonical catalogue and protocol contracts, source-derived numerical
implementations that have passed bounded parity checks, and an offline frozen
data snapshot. It has no dependency on a production website, customer system,
private datastore, or operational API.

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
