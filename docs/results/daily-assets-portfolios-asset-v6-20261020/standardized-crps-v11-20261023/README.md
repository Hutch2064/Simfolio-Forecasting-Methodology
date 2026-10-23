# Official requested-horizon standardized CRPS (V11)

The user approved this distributional evaluation rule on 2026-10-23 before comparative V11 results were computed. Source qualification, full scale-index verification and independent exact full-panel replay passed. Existing V9 evidence remains immutable. This is an approved posthoc research amendment on adaptively reused data, not new independent validation. Win-rate V10 is an experimental diagnostic, not the headline.

## Definition

For each dated forecast request q, let H be its actual requested native horizon. Score the cumulative portfolio-logreturn marginal distribution at every native forecast lead using ordinary CRPS, then take the equal-day mean C. Compute the sample standard deviation sigma from the full training-prefix portfolio daily log returns, using only information known at that origin. The request loss is:

    Z(model, q) = C(model, q) / (sigma(q) * sqrt(H(q)))
    Q(model) = sum_q fixed_weight(q) * Z(model, q)

Lower Q is better. Every candidate and baseline receives exactly the same positive request divisor and fixed weight. The divisor is constant over all leads within a request. This differs from dividing each lead's CRPS by sqrt(lead), which would change the relative day weights within that forecast.

The portfolio training return follows the panel's daily target-rebalanced accounting: log1p(sum_a w_a * expm1(asset_logreturn_a)). Sample SD uses n-1. It is not an average of constituent volatilities, a selected model's predicted spread, or a volatility measured from the evaluation period.

## Fixed panel balance

Keep the existing planned request weights: one third for short (at most 3 calendar years), one third for medium (more than 3 and less than 20), and one third for long (at least 20), with the existing 50/50 asset/portfolio balance and class/family/series/request hierarchy inside each regime. Apply the outer one third once. Do not rebalance according to observed model performance, missing results, or whichever weighting produces a larger improvement.

This is balance across the three declared calendar regimes and their planned request distribution. It does not mean every integer horizon receives identical weight. Actual short-horizon requests use their own selected forecasts; early leads from a long forecast are not relabelled as short requests.

## Baseline interpretation

Retain the same fixed with-replacement (WR) reference forecasts. Report the descriptive matched reduction:

    100 * (1 - Q(candidate) / Q(WR))

Compute one ratio after aggregation, not ratios against each realized baseline loss. The proper primary quantity is Q. Its reported relative reduction has the same observed ranking on the matched dataset; the percentage itself is not a separate proper scoring rule. If the baseline Q is zero, report the absolute scores and the undefined percentage explicitly rather than inserting an epsilon.

## Grounding and limits

Positive normalization measurable before the outcome and fixed positive averaging preserve CRPS's conditional proper-score incentives. The scored marginals assess distributional forecast accuracy; they do not identify the entire temporal joint dependence structure. Square-root horizon scaling is a simple diffusion-scale benchmark convention, not a claim that market uncertainty always follows that law. It reduces ordinary horizon and volatility scale dominance but does not bound extreme losses or guarantee improvement for every portfolio or horizon.

The score-blind audit covers all 320,017 planned requests, 64,049 training prefixes and 1,264 series. All request scales are positive and finite; no exclusions, volatility floors or fallback constants are needed. Zero training SD is outside this formula's domain. Future such cases must be reported explicitly and require a separately approved convention; do not silently drop them or invent a numerical floor.

Report regime, asset-class, family, and concentration diagnostics alongside Q. Win rate remains diagnostic. Preserve adverse cases; improve the forecasting algorithm rather than suppressing valid large errors.

[Gneiting and Raftery (2007)](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf) provides the proper-scoring foundations. [Bolin and Wallin](https://arxiv.org/abs/1912.05642) explains scale-dominance concerns; their forecast-spread-dependent SCRPS is a different construction from this past-history normalization. Neither source uniquely prescribes our horizon buckets or square-root convention.

## Reproducibility and scope

V11 rescoring reuses stored raw request mean losses, exact request IDs, forecast counts/seeds and source bindings. It does not change generated paths, model recipes, internal learning, or the reference forecast. Internal learners are subsequently compared and improved against this external objective. Historical V9/V10 records retain their original definitions. Rescoring does not add unique models or forecast executions to the full-panel catalogue.

Complete-panel results and representative development estimates must remain separate. The original 167-recipe algorithm has completed all 320,017 expanded requests. Later finite-menu and horizon-selection variants have development evidence; a rescore cannot turn that into completed full-panel coverage.

## Complete expanded-panel result

All320,017 requests per model were rescored without new forecasts. The unchanged167-recipe endpoint mixture scored **0.544202657106937**, versus fixed WR **0.6510567494403149**: **16.41240835383946% lower aggregate standardized CRPS**. History Partition scored0.652235953173445. The historical raw-V9 gain25.484505671234402% is preserved under its old definition.

[Official scoreboard](official_scoreboard_v11.json), [exact full rescore](full_primary_rescore.json), [independent replay](requested_horizon_crps_v11_independent_full_replay_review.json), and [bucket diagnostics](full_score_diagnostics_v11.json) retain identities and hashes. The requested-horizon scale index covers all64,049 training prefixes; raw arrays and the scale database remain private.

## Current development comparison under V11

The independently replayed all-series design covers17,395 requests per arm across all1,264 series at240 final paths. Its predeclared representative-mass quadrature estimates the full planned objective; it is not the actual320,017-request score.

| Method | Representative V11 Q estimate | Lower vs matched WR |
|---|---:|---:|
| finite41 | 0.595076456036 | 14.6446% |
| finite77 | 0.595358915993 | 14.6041% |
| mapped_ratio | 0.576557437896 | 17.3009% |
| Fixed WR240 | 0.697175316541 | reference |

Mapped-ratio geometry remains the development incumbent: 3.1120% lower estimated standardized CRPS than finite41 on exactly matched coverage. The actual completed full-panel headline remains16.41% for the unchanged167-recipe algorithm. Neither candidate promotion to production nor final full-suite admission follows from this rescore. [All-series evidence](all_series_confirmation_v11.json) and [independent replay](requested_horizon_crps_v11_independent_all_series_replay_review.json) preserve the separate selected-row diagnostic and exact reference bindings.

## Continuing controlled research

Under V11, the Gaussian mean-CDF learner was1.0599% better than its mapped parent on the predeclared diverse24 pilot, but worsened the separate long-only incident group. It warrants a broader matched screen, not promotion. The same learner with the older geometry worsened diverse24 by6.42%. Ratio-partition geometry worsened diverse24 by24.26% versus the current geometry despite helping the five incident targets; it is retained as a non-improver. [Aggregate pilot evidence](closed_screen_summary_v11.json) and [independent replay](requested_horizon_crps_v11_independent_closed_screens_review.json) keep the cohorts separate.

A saved-output precision check on the historical flat167 family favored2 over8 internal paths in all three seeds;8 increased standardized CRPS by11.41%,1.66% and4.19%. Separate legacy-menu comparisons also favored2. No forecasts or archive restores were needed. This does not establish optimal precision for the later finite41/mapped models. [Precision reuse evidence](historical_precision_reuse_v11.json) preserves the exact scope and has no unmatched WR-relative headline.

[Private recovery references](private_recovery_references.json) link verified full inputs, scale-index reconstruction and closed forecast evidence. Active research inputs remain local until their readers finish; these references do not authorize deleting active files.
