# Simfolio Forecasting Methodology

A standalone, public-safe research codebase for reproducing the out-of-sample methodology used to compare probabilistic portfolio forecasting models.

This repository is intentionally independent of any production website, deployment environment, customer system, private datastore, or operational API. Its purpose is methodological reproducibility: model definitions, deterministic portfolio construction, forecast-origin selection, simulation policy, distributional scoring, ranking, and auditable experiment manifests.

## Canonical study

The canonical catalogue contains **175 admissible forecasting specifications**. The leading specification is the asset-level FastMAP + dynamic Gaussian-factor/Kalman model represented by the immutable historical source ID `asset_level_fastmap_kalman_dynamic_gaussian_factor_rebalanced`.

The dense evaluation protocol uses 80 deterministic multi-asset portfolios over the common 1979-12-31 through 2026-05-13 evaluation window. Each portfolio contributes 48 rolling origins selected across the full eligible history plus three temporal holdout origins, for 4,080 origin tasks. Each origin uses 240 forecast simulations. Every eligible daily horizon is scored using exact empirical CRPS on terminal log returns. Origin losses are averaged within portfolio–horizon cells before those cells receive equal weight.

The protocol is encoded in `methodology/canonical_dense_oos.yaml`. The immutable ranked catalogue is in `catalogs/canonical_175_part1.csv` and `catalogs/canonical_175_part2.csv`.

## Reproducibility design

Catalogue membership and evaluation methodology are separate objects. The canonical 175 catalogue answers **which models are evaluated**; the dense protocol answers **how they are evaluated**. The broader research catalogue can therefore use the same evaluation protocol without creating another incompatible harness.

Historical source identifiers remain attached to evidence and execution records. Shorter `M###` names are presentation aliases only; renaming a table row never changes the model's research identity.

## Current implementation status

The repository currently contains the canonical ranking, deterministic 80-portfolio generator, dense-origin selection primitives, exact empirical CRPS scoring, experiment-protocol definitions, readable model naming, CI, and a fail-closed publication-safety scan. Statistical model implementations are promoted into this public tree only after standalone reconstruction and provenance checks in the private staging repository.

See `README_REPRODUCIBILITY.md` for the reconstruction principles and `methodology/canonical_dense_oos.yaml` for the machine-readable protocol.
