# Catalogue Governance

## Two identities: specification and experiment

A statistical specification has an immutable historical source identifier. An experiment has a catalogue membership set plus an evaluation protocol. These identities are kept separate so changing a table label cannot change a statistical model, and adding research candidates cannot silently change the OOS methodology.

## Canonical publication catalogue

The active publication catalogue contains 175 unique specifications. The retained comparable score artifact from which it is reconstructed contained source ranks 12 through 186 for these models. Canonical rank is therefore a deterministic presentation rank over this retained set: `canonical_rank = source_rank - 11`.

The first canonical row is the asset-level FastMAP + dynamic Gaussian-factor/Kalman specification. Its immutable historical identifier is `asset_level_fastmap_kalman_dynamic_gaussian_factor_rebalanced`.

The canonical files preserve both source rank and canonical rank. Source ranks are historical evidence and are never rewritten.

## Excluded historical specifications

Eleven historical source specifications preceding canonical source rank 12 are not active members of the canonical or future master execution catalogues. Their exclusion is a catalogue-governance decision based on rejecting arbitrary fixed/tuned parameterizations as research specifications to carry forward.

The public catalogue does not erase the fact that a broader historical comparison existed. Conversely, those historical rows are not executable members merely because their old scores remain part of provenance.

This repository does not claim that the exclusion policy was prespecified before all historical scores were observed unless a retained artifact establishes that chronology. That distinction matters for interpretation and is intentionally explicit.

## Master research catalogue

The master research catalogue is intended to contain every surviving unique statistical specification recovered from historical harnesses plus every canonical model. It must satisfy the machine-checked invariant:

`canonical_175 ⊆ master_research_catalog`

The older retained master source records 369 runnable methods while saved consolidated artifacts record 370 historically observed unique candidate identifiers. Those counts describe different provenance states and are not treated as competing definitions of the canonical publication set.

A model absent from the older 369-row registry but present in the canonical 175 is added to the normalized master registry as an explicit later extension. Historical source files are not silently rewritten to make them appear contemporaneous.

## Naming

Historical IDs are provenance keys and remain stable. Human-facing output uses concise aliases such as `M001 — Asset FastMAP + Dynamic Gaussian Factor`. Aliases are presentation metadata only.

## Duplicate handling

Two historical names are not automatically two unique models. Master-catalogue normalization compares statistical component metadata and maintains aliases where multiple historical IDs resolve to the same specification. Deduplication must preserve all source references so old result artifacts remain interpretable.

## Retirement boundary

Retiring a statistical specification does not imply deleting a shared mathematical primitive. Mean estimators, volatility filters, bootstrap routines, distribution samplers, numerical kernels, or dependence functions used by retained models remain available even if one historical configuration that composed them has been retired.
