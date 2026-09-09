# Reproducible Forecasting Methodology

This repository is a standalone research implementation for out-of-sample evaluation of probabilistic multi-asset portfolio forecasts. It is intentionally independent of any production web service, deployment environment, customer system, private datastore, or operational API.

## Design principles

1. **Catalogue and protocol are separate.** A catalogue selects which statistical specifications are tested; a protocol defines how they are tested.
2. **No look-ahead.** Every fit and simulated distribution may use only observations available at the corresponding forecast origin.
3. **Distributional scoring.** The canonical dense protocol scores terminal log-return distributions with exact empirical CRPS.
4. **Cell-first aggregation.** Origin losses are averaged within each portfolio–horizon cell before cells receive equal weight.
5. **Immutable experiment identity.** A reproducible result records the model catalogue, evaluation protocol, portfolio panel, seed schedule, software environment, and source revision.
6. **Public-safe implementation.** The code in this repository has no dependency on private production repositories or infrastructure.

## Repository status

The standalone scoring and experiment-contract layer is available on this reconstruction branch. The exact 175-model catalogue, full model registry, canonical portfolio panel, and model implementations are promoted into the public tree only after their private reconstruction has passed provenance and sanitization checks.

This staged publication rule is deliberate: an incomplete research artifact is not labeled canonical merely because it has the expected model count.
