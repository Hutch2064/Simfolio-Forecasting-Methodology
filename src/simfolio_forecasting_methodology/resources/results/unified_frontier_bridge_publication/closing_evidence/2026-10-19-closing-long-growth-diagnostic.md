# Closing long-horizon growth diagnostic

**Date:** 2026-10-19  
**Scope:** saved-record reconciliation only. No forecasts, model imports, fits, or new candidate runs.

This note uses the corrected 52-panel reconciliation and the independently audited original-node IID full run. The true horizons are 2,922, 5,844, and 8,766 trading days (annualized at 252 days). The 52 and 80 panels are different portfolio-context samples, not a matched expansion of the same assets. Ratios below are medians of context-level mean-path or median-path CAGR divided by that context's training CAGR, restricted to positive-training contexts; they are not ratios of pooled medians. The displayed training CAGR is the median over all contexts in the panel/horizon. Paired changes are reassembled from the pinned rows matched on horizon, context index, and portfolio.

## Training-relative portfolio growth

Each ratio cell is `mean-path / median-path` as a percent of training CAGR. Counts in the next table are descriptive; the 75% level is inherited from the saved audits and is not a floor or selection rule.

| Panel | Days | Contexts; positive training | Median training CAGR | M261 ratio | Selected FAST ratio | Original-node IID ratio |
|---|---:|---:|---:|---:|---:|---:|
| 52 | 2,922 | 52; 47 | 10.77% | 76.56 / 75.03 | 76.39 / 74.56 | 76.03 / 74.00 |
| 52 | 5,844 | 52; 46 | 10.52% | 82.35 / 82.22 | 83.40 / 85.79 | 83.53 / 83.95 |
| 52 | 8,766 | 52; 46 | 14.72% | 75.46 / 74.84 | 75.38 / 74.96 | 75.43 / 75.40 |
| 80 | 2,922 | 80; 80 | 9.51% | 71.93 / 71.52 | 72.13 / 72.69 | 71.94 / 72.25 |
| 80 | 5,844 | 80; 80 | 9.53% | 91.73 / 86.47 | 91.97 / 90.72 | 91.36 / 90.99 |
| 80 | 8,766 | 80; 80 | 10.69% | 95.51 / 83.90 | 98.89 / 98.35 | 95.26 / 93.03 |

At 8,766 days, the 52-panel mean-path and median-path shortfall is shared by M261, FAST, and IID: the IID ratios are 75.43% and 75.40%, versus 75.46%/74.84% for M261 and 75.38%/74.96% for FAST. The 80-panel picture differs by path statistic: IID is near M261 on mean-path CAGR (95.26% vs 95.51%) and above M261 on median-path CAGR (93.03% vs 83.90%), while FAST is nearer training on both. These are descriptive differences, not evidence that IID corrects the underlying growth mechanism.

The 52-panel audit reports separate `CAGR of mean terminal wealth` ratios at 8,766 days of 83.36% for IID, 83.17% for M261, and 83.58% for FAST. This is not the same statistic as mean-path CAGR, and no matched 80-panel control field exists for it; it is therefore not used as a cross-panel comparison.

## Paired CAGR shifts

Entries are the average and median of the within-context IID-minus-control change, in percentage points, followed by the count of contexts with a higher/lower IID forecast CAGR. “Mean path” and “median path” are kept separate. A higher forecast CAGR is not necessarily closer to realized growth or a better forecast.

| Panel | Days | Control | Mean-path Δ (average / median; higher/lower) | Median-path Δ (average / median; higher/lower) |
|---|---:|---|---:|---:|
| 52 | 2,922 | M261 | +0.010 / +0.020; 27/25 | −0.035 / −0.028; 23/29 |
| 52 | 2,922 | FAST | +0.010 / +0.021; 32/20 | −0.022 / −0.025; 20/32 |
| 52 | 5,844 | M261 | −0.011 / +0.007; 29/23 | +0.049 / +0.052; 29/23 |
| 52 | 5,844 | FAST | −0.112 / −0.027; 18/34 | −0.393 / −0.205; 3/49 |
| 52 | 8,766 | M261 | +0.025 / +0.015; 30/22 | +0.024 / +0.031; 32/20 |
| 52 | 8,766 | FAST | +0.036 / +0.017; 34/18 | +0.028 / +0.030; 33/19 |
| 80 | 2,922 | M261 | −0.009 / −0.006; 38/42 | +0.041 / +0.037; 49/31 |
| 80 | 2,922 | FAST | −0.029 / −0.020; 28/52 | −0.091 / −0.066; 22/58 |
| 80 | 5,844 | M261 | −0.117 / −0.087; 25/55 | +0.409 / +0.349; 70/10 |
| 80 | 5,844 | FAST | −0.100 / −0.051; 20/60 | −0.093 / −0.070; 31/49 |
| 80 | 8,766 | M261 | +0.061 / +0.011; 48/32 | +0.835 / +0.157; 69/11 |
| 80 | 8,766 | FAST | −0.229 / −0.075; 18/62 | −0.306 / −0.132; 14/66 |

The audited below-75%-of-training counts (mean-path / median-path) at 8,766 days are 19/23 of 46 positive-history contexts for IID, 20/23 for M261, and 20/25 for FAST in panel 52; in panel 80 they are 17/15 of 80 for IID, 14/20 for M261, and 14/13 for FAST. Nonpositive forecast-CAGR counts across all contexts are 6/6 for every 52 model at 8,766 days and 0/0 for every 80 model. These counts remain descriptive.

## Interpretation and boundary

- The 52-panel long-horizon shortfall persists under original-node IID and is nearly indistinguishable from M261 and FAST. On 80, IID improves the median-path comparison to M261 but not mean-path growth; it remains behind FAST on both. Neither pattern identifies a general growth correction.
- The 80 result uses reused development contexts and one fixed seed/context; the independent audit notes overlapping holdings/origins. It is not independent validation. The candidate's full scores are +2.1505% vs M261 on 52 and +0.0840% on 80, while it is 0.8219% worse than FAST on 52; this growth diagnostic does not change that selection result or the separate speed requirement.
- The 396 saved growth records are portfolio-context CAGR summaries. Although the saved run also contains asset terminal-log diagnostics for accounting, these pinned growth summaries do not provide a per-asset forecast/training CAGR panel. Thus they do not establish the user's cross-asset-and-portfolio growth criterion at the individual-asset level.
- No additional general growth law is supported by these growth outcomes. The saved growth stopping assessment already covers raw historical-log-mean substitution, permanent HAC uncertainty, zero-centered mean-reverting AR(1), predecessor-mean uncertainty, the history-fitted conditional mean, and conditional mean plus state uncertainty. Full-52 mean/state variants regressed; their training-only likelihood evidence and leave-one-asset-out results did not produce a stable activation rule. The 8,766-day shortfall is not a basis for imposing a historical-growth floor, choosing a horizon, or adding another mean-strength grid. Reopen only if a new training-only residual or chronological inner-training test isolates a stable, specified missing growth mechanism; none is present in the pinned evidence.

## Evidence pins

- `catalogue_support/drafts_original_nodes_IID/growth-supplement-v3.json` — SHA-256 `6c63bc7a79a01e7f7f6bfa3c043cd1a1eba5bdcd85acbcc702026566f716c676` (52-panel growth summaries; ratios and descriptive counts).
- `baseline52_growth_reconciliation/revision_v2/matched52-growth-v3.json` — SHA-256 `9f23b790d2e23418781dba9876224b8d6cc2441ad7a6acc47e8ec508a125bdc9` (52-panel corrected source, paired contexts, and training values).
- `iid_original_nodes/full-52d4c26b608ac1ac/long-growth.json` — SHA-256 `266fa8bc507ce0a16316c996be807450f8017237a165d7a52652c7628725ac10` (396 saved portfolio-context growth records).
- `diagnostics/m261-relative-long-controls.json` — SHA-256 `9d39a18ffb340fa0b32ffd441bb3bd9839bbccbd31e091ce88891b6323fbd4f0` (matched 80-panel M261/FAST records and historical growth controls).
- `diagnostics/iid_original_nodes_full_actual_audit/review.json` — SHA-256 `601dcd72bfd541533ad428e65d92cdae37de72e6aee30a0602aa920da3120347` (independent saved-record audit; binds the 396 candidate records, matched 80 controls, and selected80 analysis SHA-256 `97bd5da948be6a8a0393ba3f228e34b4c9b54330ad2ce58ddc6177f74ae069d0`).
- `conditional_growth/growth-stopping-assessment.json` — SHA-256 `64c10aef687f2a3304c78915a91a27506f9305558db98a28c4d8d2e0f51beef1` (prior growth-variant audit and reopening condition).

The paired average/median deltas and median training CAGRs above were summarized from these saved rows only, matched by panel/horizon/context index/portfolio; no score, model, or path was regenerated.
