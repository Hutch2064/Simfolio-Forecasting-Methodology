# Expanded short-horizon takeover, v7

This record documents a versioned extension of the existing 64,049-request asset-level panel. It adds native forecast horizons 1, 5, 10, and 21 while preserving the original requested horizons, model identities, forecast-path accounting, and v7 scoring hierarchy. The expansion is still running; this folder contains no final score, winner, canonical-ranking change, or promotion claim.

## Scoring and schedule

For each request, the score remains the equal mean over native leads of `CRPS_h / (training-only portfolio SD, ddof=1 × sqrt(h))`. Request scores continue through the existing series/quarter, family, class, and matched asset/portfolio hierarchy. Horizons are scored separately before aggregation; scores are not pooled as raw CRPS and are not reweighted into horizon bands.

The original 64,049 requested rows per model remain intact. Their schedule contains 63,992 unique training prefixes; 57 prefixes have two original descriptors. The appended schedule creates one row for each unique prefix and each of the four added horizons. Duplicate descriptors remain trace aliases and do not add score votes. The frozen execution plan uses two inner paths for past-only selection and 1,000 final joint paths per forecast. It catches up only missing original work before continuing through the combined schedule with one shared 12-worker pool.

## Handoff evidence

At the independently recorded handoff snapshot on 2026-10-22 03:19:28 UTC, the takeover was running with 12 workers and had completed 540 of 767,904 new short-horizon forecasts, with zero native failures, scoring-integrity errors, or I/O errors. The prior saved handoff retained 24,574 completed candidate forecasts and reused 128,098 baseline score rows. These are point-in-time coverage facts, not a completed comparison.

A separate source-bound smoke covered two sampled origins, three models, and all four added horizons (24 forecasts total); its independent audit passed. The source review bound 91 inputs to the frozen plan and reported no change to the numerical source or environment. No production method or canonical score was changed.

## Recovery and provenance

The frozen runner, 91-source closure, plan, reviews, drain evidence, and completed smoke are preserved in a private release. The archive excludes the active run's checkpoints, progress files, inputs, and the active baseline SQLite payload; its locator records their recovery references and checksum. See `PROVENANCE.json` for exact source hashes and immutable release asset IDs.
