# Completed expanded request panel: endpoint mixture and two references

The three models completed the same **320,017 requests each**, with identical
final path count within every comparison: **59,906 requests at 1,000 paths and
260,111 at 240 paths** per model. The completed forecast and score snapshot is
preserved here; independent full replay is verified. Full private-archive recovery is verified, including the final-V9 and
bookkeeping addendum; the three executions are admitted to the catalogue. No new forecasts were run for this
catalogue update.

| Method | V9 main Q | Better versus fixed WR |
|---|---:|---:|
| Request-specific endpoint mixture — 167 recipes, 2 inner paths | 0.3732134377688732 | 25.484505671234402% |
| Joint blocks — With Replacement (WR, fixed reference) | 0.5008534683031683 | reference |
| Joint blocks — History Partition (WOR) | 0.5069609928754524 | −1.219423435955358% |

The headline is `100*(1-Q_endpoint/Q_fixed_WR)`. WR stays the fixed reference even
though WOR has the higher loss on this completed V9 panel. The three bucket
losses enter Q with equal one-third coefficients; the headline is not an average
of three percentage improvements.

V9 scores each request by mean raw daily cumulative-log-return CRPS. It uses no
training-SD divisor, square-root-lead divisor, calendar-log lead weights or
quarter grouping. The full planned schedule fixes request weights before loss
reduction: within each calendar-horizon bucket, equal assets/portfolios branches,
classes, families and series, then equal planned requests within each series.
Short is through 3 calendar years, medium more than 3 and less than 20 years,
and long 20 years or more, using actual native dates. There are 278,893 short,
37,695 medium and 3,429 long requests. Each bucket's assets and portfolios
branches have 50% planned weight.

The external score was selected retrospectively after comparisons on this
research cohort. Forecasts, origins and horizons are extensively reused and
overlap. These results are adaptive descriptive evidence, with no claim of
independent validation, statistical significance, production readiness, or
universal superiority of this scoring objective. Older protocols and scores
remain separate historical records; their numerical Q values are not comparable.

The actual algorithm is **endpoint CRPS learning**, not the earlier fading
volatility method or the aligned daily-objective candidate. It learns a mixture
from request-local historical validation origins over the fixed universe of
167 flattened recipes, with two inner paths and the half-doubling schedule.
Eligibility depends on available history. The historical objective remains
endpoint CRPS divided by square root of internal horizon. The V3 adapter reuses
identical fits and path work with the qualified solver; it does not change the
financial recipe universe or train weights on external V9 losses.
The [optimization qualification](optimization_alias_evidence.json) preserves
the small certified solver differences:355contexts across3inner seeds include
some nonidentical weights/paths/losses. The maximum component-weight difference
was5.7355346596080214e-6. This is qualified same-algorithm optimization, not a
universal bitwise-equivalence claim.

The exact logical ID is
`model_testing_asset_information_endpoint_components_v1_1a5c92ef4802021d`.
[registry.json](registry.json) binds it to the original endpoint source identity
and separately records the actual optimized execution closure. Optimization
aliases count once. The two baseline identities retain their previous canonical
implementation digests. The completed expanded execution changes the catalogue from
**23 unique methods / 28 executions to 24 / 31**, with 22 methods having at least
one failure-free complete result. Rescoring these same forecasts adds neither
methods nor executions. Earlier panels, failed methods and every historical
execution remain preserved.

The [109-file scientific source manifest](scientific_source_closure.json),
[immutable final scoreboard](series_buckets_v9_live/final_scoreboard_snapshot_v9.json), exact V9 scorer and
[protocol snapshot](series_buckets_v9/protocol.json) are public-safe evidence.
The protocol's original preparation-time status and the earlier partial review
are preserved verbatim; the [completed independent full review](series_buckets_v9_live/independent_full_replay_review.json) now supersedes that partial review.
[execution_records.json](execution_records.json) binds all131 main archive parts
and the verified metadata-repair/final-V9 addendum. The original pending records
remain as preparation history, not the current admission state. Loss arrays,
provider inputs, raw caches and full paths belong in the private lossless archive.
Worker time was not reconstructed across inherited and new generations; no
zero-time entry or performance comparison is reported.

The [binding addendum](series_buckets_v9_live/final_scoreboard_binding_addendum.json)
checks nine unchanged scientific/coverage fields between the independently
replayed report and the immutable final snapshot. A refreshed reporting timestamp
does not create a new financial result or catalogue execution.

Private recovery: [full archive receipt](FULL_ARCHIVE_RECEIPT.json),
[verified owner bindings](verified_recovery_bindings.json), and
[metadata addendum](BOOKKEEPING_ADDENDUM_RECEIPT.json). The owner retired
189,268 verified eligible payload files (2,445,985,257 bytes); the exact offload
marker is retained. The earlier archive receipt captured originals before that
retirement; its local-retention flag is a historical snapshot. A detached job's
post-offload bookkeeping error was repaired with the original error preserved
in the private addendum; it is not a native forecast or scoring-method failure.
No new forecasts or score changes were made for catalogue publication. PPW and
MC studies remain development-only and add no full-panel catalogue entries.
