# Simfolio Forecasting Methodology

A standalone, public-safe research codebase for reproducing the out-of-sample methodology used to compare probabilistic portfolio forecasting models.

This repository is intentionally independent of any production website, deployment environment, customer system, private datastore, or operational API. Its purpose is methodological reproducibility: model definitions, deterministic portfolio construction, forecast-origin selection, simulation policy, distributional scoring, ranking, and auditable experiment manifests.

## Canonical study

The canonical catalogue contains **175 admissible forecasting specifications**. The leading specification is the asset-level FastMAP + dynamic Gaussian-factor/Kalman model represented by the immutable historical source ID `asset_level_fastmap_kalman_dynamic_gaussian_factor_rebalanced`.

The dense evaluation protocol uses 80 deterministic multi-asset portfolios over the common 1979-12-31 through 2026-05-13 evaluation window. Each portfolio contributes 48 rolling origins selected across the full eligible history plus three temporal holdout origins, for 4,080 origin tasks. Each origin uses 240 forecast simulations. Every eligible daily horizon is scored using exact empirical CRPS on terminal log returns. Origin losses are averaged within portfolio–horizon cells before those cells receive equal weight.

The authoritative membership and verification ledger is packaged at `src/simfolio_forecasting_methodology/resources/canonical_175/ledger.json`. See the generated [canonical model reference](docs/canonical-model-reference.md) for row-level status.

## Reproducibility design

This repository implements only the 175 canonical white-paper identities. The ledger records model membership and evidence; the protocol defines their evaluation.

Historical source identifiers remain attached to evidence and execution records. Shorter `M###` names are presentation aliases only; renaming a table row never changes the model's research identity.

## Current implementation status

Implementation and validation are in progress. Run `simfolio-oos coverage` for evidence-backed counts and `simfolio-oos scores --experiment canonical-whitepaper` to inspect imported retained results without forecasting. Unverified model mappings fail explicitly. A successful bounded smoke run does not establish reproduction of a retained white-paper score.

See [reproducibility status](README_REPRODUCIBILITY.md) and [architecture](docs/architecture.md).
