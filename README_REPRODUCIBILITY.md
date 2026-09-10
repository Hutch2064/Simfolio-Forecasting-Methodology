# Reproducible Forecasting Methodology

This package implements the public contracts for the canonical dense daily
out-of-sample study. Catalogue membership, protocol identity, data identity,
model execution, and score evidence are kept as separate records so a passing
smoke test cannot be mistaken for a reproduced historical result.

## Fixed identities

The catalogue has 175 rows and an immutable membership digest in
`resources/canonical_175/ledger.json`. The dense protocol uses 80 deterministic
portfolios, 48 rolling origins plus three temporal holdout origins per
portfolio, 4,080 origin tasks, 240 simulations per origin, and equal-weighted
portfolio-horizon cells. It scores terminal log-return distributions with
exact empirical CRPS.

The packaged data manifest identifies source revision
`773bc1c325559e6bf57a567f1d8bf473a3427fbc`, the 52-series asset panel, `EFFRX`,
five additional drift-proxy series, and the French/Q5 factor inputs. The normalized return-matrix identity is
`52c5bdd96b39762183ef0c204fa8165c2dfd5a4864e7198662615daddb8d6a49`. All
verification and preparation is local; no substituted histories, network refresh, or
implicit annualization is allowed.

## Runtime and dependency matrix

The installed runtime exposes the verified factories reported by
`simfolio-oos coverage`. Factories are selected by exact
canonical IDs. Unknown IDs and ledger rows without a verified factory fail
closed rather than falling back to a generic model.

CI validates Python 3.11 and 3.12 with the optional `all-models` dependency
set. The lock file records the tested interpreter-specific scientific stack:
Python 3.11 uses NumPy 2.4.6 and SciPy 1.17.1; Python 3.12 uses NumPy 2.5.3
and SciPy 1.18.1. The three NIG MCMC paths can therefore have different
environment-specific digest values. Each matrix leg must match its own
source-local fixture; tolerances are not widened to hide a dependency change.

Strict fixture identity also includes the reference platform. The retained
source replay used macOS 27 arm64 with Apple Accelerate BLAS/LAPACK; the
recorded Ubuntu x64 run fails 18 strict parity/data tests in each Python leg.
Cross-platform byte-exact equivalence is unsupported, and a local source
replay or GitHub Actions wheel test does not establish live website, API, or
deployed-server behavior. See [Numerical environment and cross-platform parity](docs/numerical-environment.md)
for the measured differences and the pending explicit ARM64 reference-job
policy. No tolerance widening is part of that policy.

The wheel checks run outside the source checkout. They verify package-resource
loading, the 60-file offline snapshot, the 80/4,080/701,280 protocol counts,
the registered factory map, bounded family and Frontier parity, and public-safety
scans before the complete test suite. Run `simfolio-oos coverage --json` for
the current per-row counts; the coverage report is authoritative for what is
executable and what remains evidence-only.

## Evidence status

Source-derived implementations have bounded fit-state, simulation, and rejoin
parity fixtures where the ledger marks them executable. The Frontier report and
the retained-wrapper replay are bounded method checks; they do not rerun the
retained score artifact. The historical score artifact remains evidence-only,
and its linkage is qualified by the [runtime discrepancies](docs/historical-runtime-discrepancies.md).

The ledger records 175 recovered identities and 175 protocol, dataset, and
panel fingerprints. It separately records the currently executable factory
coverage and complete statistical-specification coverage. A successful
installation, plan, factory smoke run, or bounded source replay does not
establish reproduction of the retained white-paper ranking. No score is
relabelled as reproduced until its historical execution and score linkage are
verified.

The latest canonical-only coverage reports 175 registered factories,
instantiations, forecast smoke checks, and bounded source-parity checks; 175
rows have complete statistical specifications, and zero historical scores are
verified. These counts are status fields from `simfolio-oos coverage --json`,
not evidence that the retained score ranking has been rerun.

The checkout has no `LICENSE` file at present. License selection remains an
administrative handoff item; no license terms are invented for the source or
the data snapshot.

For the exact file hashes, attribution statement, protocol fingerprint, and
prepared-cache checks, see
[docs/data-and-reproducibility.md](docs/data-and-reproducibility.md).
