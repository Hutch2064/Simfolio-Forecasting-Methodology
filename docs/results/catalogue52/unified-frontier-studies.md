# Unified Frontier studies

Research evidence only. Scores are separate by panel; compute timings are not browser CTV.

## UF-FACTORIAL-f560a3f6b73353ed

Isolate conventional future-volatility removal and continuous shock-node variance correction on both complete canonical panels.

| Panel | Status | Canonical C-order empirical CRPS by label | Raw-result scalar differences |
| --- | --- | --- | --- |
| Full 52 | complete_scored_and_audited | M261_saved_control: 0.61386958108984568; RMS_corrected_nodes_saved_control: 0.59577166080711608; RMS_original_nodes_saved_control: 0.59681892827927252; conventional_SV_corrected_variance_nodes: 0.60221546105451607; rough_off_original_nodes_saved_control: 0.60331093641703915 | RMS_corrected_nodes_saved_control: raw 0.59577166080711585, 2 ULP |
| Full 80 | complete_scored_and_audited | M261_saved_control: 0.22672308848311776; RMS_corrected_nodes_saved_control: 0.23457102422540146; RMS_original_nodes: 0.23258808209819795; conventional_SV_corrected_variance_nodes: 0.23007252374266968; rough_off_original_nodes_saved_control: 0.23051618147311073 | M261_saved_control: raw 0.22672308848311779, 1 ULP; RMS_corrected_nodes_saved_control: raw 0.23457102422540138, 3 ULP |

Runtime: not_measured. This study shares fits/uniforms and records diagnostic stage/campaign time only. New representative single/multi fresh/cached speed measurement remains pending; prior six-single-context timings remain in the historical refinement sidecar.

Production: not validated.

Canonical panels: exploratory reused evidence. Seed robustness: not_tested. Independent post-selection: not_tested.

- UF-CONTROL-M261 (diagnostic): Unchanged exact saved baseline, kept separately on both panels.
- R52-ROUGH-OFF (diagnostic): Retained factorial reference; improves full52 but deteriorates full80.
- UF-RMS-ORIGINAL-NODES (rejected): Full52 improves 2.777569% versus M261; full80 deteriorates 2.586853%, failing the joint objective.
- UF-CONVENTIONAL-SV-VARIANCE-NODES (rejected): Full52 improves 1.898468%; full80 deteriorates 1.477324%. It fails the joint objective and trails the selected full52 reference.
- R52-RMS-CONTINUOUS-VARIANCE-NODE-CORRECTION (retained): Retain as the fast full52 research reference, not a winner: full52 improves 2.948170%; full80 deteriorates 3.461463%.

Archive receipt: `short-horizon-mechanisms/refinement52/unified_frontier/catalogue_support/factorial_archive_closure.json`.

## UF-CONDITIONAL-1825f585d4e8d7d5

standalone conditional covariance and conditional mean mechanisms on one unified selected base; no combined candidate

| Panel | Status | Canonical C-order empirical CRPS by label | Raw-result scalar differences |
| --- | --- | --- | --- |
| Full 52 | complete_scored_and_audited | M261_saved_control: 0.61386958108984568; RMS_corrected_nodes_saved_control: 0.59577166080711608; RMS_original_nodes_saved_control: 0.59681892827927252; conventional_SV_corrected_variance_nodes: 0.60221546105451607; covariance_only: 0.59577166080711608; mean_only: 0.59699804263973377; rough_off_original_nodes_saved_control: 0.60331093641703915 | RMS_corrected_nodes_saved_control: raw 0.59577166080711585, 2 ULP; covariance_only: raw 0.59577166080711585, 2 ULP; mean_only: raw 0.59699804263973366, 1 ULP |
| Full 80 | complete_scored_and_audited | M261_saved_control: 0.22672308848311776; RMS_corrected_nodes_saved_control: 0.23457102422540146; RMS_original_nodes: 0.23258808209819795; conventional_SV_corrected_variance_nodes: 0.23007252374266968; covariance_only: 0.23387900572210277; mean_only: 0.23385584849519025; rough_off_original_nodes_saved_control: 0.23051618147311073 | M261_saved_control: raw 0.22672308848311779, 1 ULP; RMS_corrected_nodes_saved_control: raw 0.23457102422540138, 3 ULP; covariance_only: raw 0.23387900572210274, 1 ULP |

Runtime: not_measured. Campaign/shared-fit stage times are diagnostic only. The sealed 14-case byte-oracle packet supports future matched single/multi fresh/cached measurement; no timing modes or benchmark results are claimed.

Production: not validated.

Canonical panels: exploratory reused evidence. Seed robustness: not_tested. Independent post-selection: not_tested.

- UF-CONTROL-M261 (diagnostic): Unchanged exact saved baseline, kept separately on both panels.
- R52-ROUGH-OFF (diagnostic): Retained factorial reference; improves full52 but deteriorates full80.
- UF-RMS-ORIGINAL-NODES (rejected): Full52 improves 2.777569% versus M261; full80 deteriorates 2.586853%, failing the joint objective.
- UF-CONVENTIONAL-SV-VARIANCE-NODES (rejected): Full52 improves 1.898468%; full80 deteriorates 1.477324%. It fails the joint objective and trails the selected full52 reference.
- R52-RMS-CONTINUOUS-VARIANCE-NODE-CORRECTION (retained): Retain as the fast full52 research reference, not a winner: full52 improves 2.948170%; full80 deteriorates 3.461463%.
- UF-CONDITIONAL-COVARIANCE (rejected): Full52 CRPS 0.59577166080711608 (-2.948170% vs M261); full80 CRPS 0.23387900572210277 (+3.156237% vs M261). It fails the joint objective; preserve the exact tested law and controls.
- UF-CONDITIONAL-MEAN (rejected): Full52 CRPS 0.59699804263973377 (-2.748391% vs M261); full80 CRPS 0.23385584849519025 (+3.146023% vs M261). It fails the joint objective; preserve the exact tested law and controls.

Archive receipt: `short-horizon-mechanisms/refinement52/unified_frontier/catalogue_support/conditional_archive_closure.json`.

## UF-MOMENTS-f4ef8715429fc6e6

source-proven total conventional+rough covariance and conventional mean+covariance on unified selected base; no routing or fitted weights

| Panel | Status | Canonical C-order empirical CRPS by label | Raw-result scalar differences |
| --- | --- | --- | --- |
| Full 52 | complete_scored_and_audited | M261_saved_control: 0.61386958108984568; RMS_corrected_nodes_saved_control: 0.59577166080711608; RMS_original_nodes_saved_control: 0.59681892827927252; conventional_SV_corrected_variance_nodes: 0.60221546105451607; conventional_mean_plus_covariance: 0.59699804263973377; covariance_only: 0.59577166080711608; mean_only: 0.59699804263973377; rough_off_original_nodes_saved_control: 0.60331093641703915; total_covariance_only: 0.59577166080711608 | RMS_corrected_nodes_saved_control: raw 0.59577166080711585, 2 ULP; conventional_mean_plus_covariance: raw 0.59699804263973366, 1 ULP; covariance_only: raw 0.59577166080711585, 2 ULP; mean_only: raw 0.59699804263973366, 1 ULP; total_covariance_only: raw 0.59577166080711585, 2 ULP |
| Full 80 | complete_scored_and_audited | M261_saved_control: 0.22672308848311776; RMS_corrected_nodes_saved_control: 0.23457102422540146; RMS_original_nodes: 0.23258808209819795; conventional_SV_corrected_variance_nodes: 0.23007252374266968; conventional_mean_plus_covariance: 0.2331864906759; covariance_only: 0.23387900572210277; mean_only: 0.23385584849519025; rough_off_original_nodes_saved_control: 0.23051618147311073; total_covariance_only: 0.23239664339923255 | M261_saved_control: raw 0.22672308848311779, 1 ULP; RMS_corrected_nodes_saved_control: raw 0.23457102422540138, 3 ULP; conventional_mean_plus_covariance: raw 0.23318649067589994, 2 ULP; covariance_only: raw 0.23387900572210274, 1 ULP |

Runtime: not_measured. Campaign/shared-fit stage times are diagnostic only. The sealed 14-case byte-oracle packet supports future matched single/multi fresh/cached measurement; no timing modes or benchmark results are claimed.

Production: not validated.

Canonical panels: exploratory reused evidence. Seed robustness: not_tested. Independent post-selection: not_tested.

- UF-CONTROL-M261 (diagnostic): Unchanged exact saved baseline, kept separately on both panels.
- R52-ROUGH-OFF (diagnostic): Retained factorial reference; improves full52 but deteriorates full80.
- UF-RMS-ORIGINAL-NODES (rejected): Full52 improves 2.777569% versus M261; full80 deteriorates 2.586853%, failing the joint objective.
- UF-CONVENTIONAL-SV-VARIANCE-NODES (rejected): Full52 improves 1.898468%; full80 deteriorates 1.477324%. It fails the joint objective and trails the selected full52 reference.
- R52-RMS-CONTINUOUS-VARIANCE-NODE-CORRECTION (retained): Retain as the fast full52 research reference, not a winner: full52 improves 2.948170%; full80 deteriorates 3.461463%.
- UF-CONDITIONAL-COVARIANCE (rejected): Full52 CRPS 0.59577166080711608 (-2.948170% vs M261); full80 CRPS 0.23387900572210277 (+3.156237% vs M261). It fails the joint objective; preserve the exact tested law and controls.
- UF-CONDITIONAL-MEAN (rejected): Full52 CRPS 0.59699804263973377 (-2.748391% vs M261); full80 CRPS 0.23385584849519025 (+3.146023% vs M261). It fails the joint objective; preserve the exact tested law and controls.
- UF-TOTAL-CONDITIONAL-COVARIANCE (rejected): Full52 CRPS 0.59577166080711608 (-2.948170% vs M261); full80 CRPS 0.23239664339923255 (+2.502416% vs M261). It fails the joint objective; preserve the exact tested law and controls.
- UF-CONVENTIONAL-MEAN-PLUS-COVARIANCE (rejected): Full52 CRPS 0.59699804263973377 (-2.748391% vs M261); full80 CRPS 0.2331864906759 (+2.850791% vs M261). It fails the joint objective; preserve the exact tested law and controls.

Archive receipt: `tmp/catalogue52-20261018/short-horizon-mechanisms/refinement52/unified_frontier/catalogue_support/moment_followup_archive_closure.json`.

## Factorial-closure decision snapshot

The following decision is preserved as at the closure of `factorial-f560a3f6b73353ed`. Its next-action statements describe that historical checkpoint. Later conditional and moment studies have their own records above; this snapshot preserves only the factorial decision checkpoint.

# Completed conventional-volatility / shock-variance factorial

This study preserves rough volatility off, the original fitted point/variance curves, Student corrected-DCC, canonical data/tasks/dates/seeds, 240 complete paths, and exact empirical CRPS cells. It computes the two missing factorial cells; published immutable controls provide the other two. It does not change production.

| Unified law | Full52 improvement vs M261 | Full80 improvement vs M261 |
| --- | ---: | ---: |
| Conventional multiplier on, original nodes (R0) | +1.720014% | -1.673007% |
| Conventional multiplier off/RMS, original nodes | +2.777569% | -2.586853% |
| Conventional multiplier on, variance-only-corrected nodes | +1.898468% | -1.477324% |
| Conventional multiplier off/RMS, variance-only-corrected nodes (selected fast reference) | +2.948170% | -3.461463% |

All 2,652 single-asset origins and 4,080 multiasset origins completed without unresolved failures. The separate independent reducer reconstructed all 6,732 checkpoint pairs, every new score cell/count, all original task identities and saved controls, with bound sources unchanged. The 52 and 80 panels remain separate; no weighted aggregate determines selection.

## Mechanistic conclusion and next action

On52, turning off the conventional future multiplier lowers CRPS by 0.006492008138 with original nodes and 0.006443800247 with corrected nodes. On80 it instead raises CRPS by 0.002071900625 and 0.004498500483. Variance correction improves80 slightly when the multiplier is present, but worsens80 when it is absent. The80 interaction is 0.002426599858; the52 interaction is only 0.000048207890.

The two-panel regression therefore is not explained solely by the rough-removal step or by a universal harmful variance correction. Conventional scale randomness and its interaction with the mapped shocks matter to the multiasset forecasts. This factorial does not separately identify covariance, nonzero-node mean, higher moments, serial volatility or nonlinear compounding as the sole cause. Those require isolated probes.

No factorial law meets the joint objective. Retain the selected fast reference as the best52 control while testing the prepared standalone conditional covariance emission and mean compensation on the same unified base. Both mechanisms have training-only analytic definitions; neither introduces a ticker rule, horizon cutoff or fitted evaluation-period selector. Their combined form is not yet selected or run. Rough conditional-moment/path alternatives remain a documented reserve branch, not an exhausted family.

## Growth, runtime and validation boundaries

`growth-comparison.json` preserves 1,500 new and available retained long-context growth records. Mean path CAGR, median path CAGR and CAGR of mean terminal wealth are distinct. Ratios to strictly positive training CAGR and absolute percentage-point differences are descriptive; they impose no historical floor. Original M26180 and R080 path-growth outputs were not retained, so their values are not inferred from CRPS cells.

The run shares fitted work and original uniforms across variants. Its campaign wall time and ordered stage timings do not establish speedups. New representative single/multi fresh/cached benchmarks remain pending. Canonical panel outcomes have informed development; no independent post-selection or production validation is claimed.

`analysis.json` reduces the final saved cells in a consistent contiguous order. A few raw `results.json` scalar means differ by one to three floating-point ULPs because the original reducer used a different array layout before saving. The underlying cell bytes, denominators and exact CRPS formulas agree. Preserve both scalar representations and the independently checked numerical reduction bounds; do not rewrite valid forecasts or numerical output files to conceal this difference.


## Verified private preservation

Main archive `2f0b153ea2accba072ba` and source/runtime supplement `b928eddbb89b2ed8f477` were uploaded, fresh-downloaded and checked member by member. All 14,334 members and the complete 590-file immutable snapshot passed hashing/readability/restoration checks. Four preselected original forecast path and CRPS byte oracles passed in isolated restored source/data, with original-LAB fallback forbidden. This is restoration validation, separate from seed robustness, independent forecast validation and production validation.

The historical dependency lock differs from the installed campaign environment in 16 peripheral/build distributions. The archive preserves both the old lock and a complete 46-distribution installed inventory with exact restore pins. The replay used separate owned dependency paths sharing the unchanged installed binaries. The exact path/CRPS result does not claim a fresh package-download rebuild.

Raw evidence and provider data remain in the private `Hutch2064/simfolio-engine` research release. This public sidecar contains metadata and hashes only. Existing R52 rows, selected80 evidence, M-model identities and catalogue ranks remain unchanged. No factorial variant wins both panels; the overall research stopping condition remains open.

## Conditional and moment-followup closure snapshots

The two appended study records describe their own closed scored/audited experiments and verified batch preservation. Both full panels remain separate, all controls and rejected variants are retained, and neither probe supplies a joint winner. Later experiments, new-seed validation, independent validation, representative runtime measurement and production validation are separate gates. The four restored path/CRPS oracles cover both studies and both panels; they validate preservation only.

## Interaction and exact conditional-rough reserve closure

Both appended studies completed and independently audited all canonical full52/full80 origins. Separate full52/full80 scores, all controls and rejected laws are retained. Neither law meets the joint objective. Private preservation passed fresh byte/member checks and five restored path/CRPS oracles, including one unchanged original M261 observer; all 240 saved M261 records/80 portfolios were independently reconstructed without extra financial forecasts. This validates restoration only.

Research evidence only. Scores are separate by panel; compute timings are not browser CTV.

## UF-INTERACTION-0c07f7cefb83d7ed

one total covariance kernel plus unchanged existing conventional-only mean compensation; no kappa scale or fitted weight

| Panel | Status | Canonical C-order empirical CRPS by label | Raw-result scalar differences |
| --- | --- | --- | --- |
| Full 52 | complete_scored_and_audited | M261_saved_control: 0.61386958108984568; RMS_corrected_nodes_saved_control: 0.59577166080711608; RMS_original_nodes_saved_control: 0.59681892827927252; conventional_SV_corrected_variance_nodes: 0.60221546105451607; conventional_mean_plus_covariance: 0.59699804263973377; covariance_only: 0.59577166080711608; mean_only: 0.59699804263973377; rough_off_original_nodes_saved_control: 0.60331093641703915; total_covariance_only: 0.59577166080711608; total_kernel_existing_conventional_mean: 0.59699804263973377 | RMS_corrected_nodes_saved_control: raw 0.59577166080711585, 2 ULP; conventional_mean_plus_covariance: raw 0.59699804263973366, 1 ULP; covariance_only: raw 0.59577166080711585, 2 ULP; mean_only: raw 0.59699804263973366, 1 ULP; total_covariance_only: raw 0.59577166080711585, 2 ULP; total_kernel_existing_conventional_mean: raw 0.59699804263973366, 1 ULP |
| Full 80 | complete_scored_and_audited | M261_saved_control: 0.22672308848311776; RMS_corrected_nodes_saved_control: 0.23457102422540146; RMS_original_nodes: 0.23258808209819795; conventional_SV_corrected_variance_nodes: 0.23007252374266968; conventional_mean_plus_covariance: 0.2331864906759; covariance_only: 0.23387900572210277; mean_only: 0.23385584849519025; rough_off_original_nodes_saved_control: 0.23051618147311073; total_covariance_only: 0.23239664339923255; total_kernel_existing_conventional_mean: 0.23171189271702697 | M261_saved_control: raw 0.22672308848311779, 1 ULP; RMS_corrected_nodes_saved_control: raw 0.23457102422540138, 3 ULP; conventional_mean_plus_covariance: raw 0.23318649067589994, 2 ULP; covariance_only: raw 0.23387900572210274, 1 ULP; total_kernel_existing_conventional_mean: raw 0.23171189271702688, 3 ULP |

Runtime: not_measured. Ordered shared-fit stage times remain diagnostic. No representative matched fresh/cached timing or authenticated production browser CTV proof.

Production: not validated.

Canonical panels: exploratory reused evidence. Seed robustness: not_tested. Independent post-selection: not_tested.

- UF-CONTROL-M261 (diagnostic): Unchanged exact saved baseline, kept separately on both panels.
- R52-ROUGH-OFF (diagnostic): Retained factorial reference; improves full52 but deteriorates full80.
- UF-RMS-ORIGINAL-NODES (rejected): Full52 improves 2.777569% versus M261; full80 deteriorates 2.586853%, failing the joint objective.
- UF-CONVENTIONAL-SV-VARIANCE-NODES (rejected): Full52 improves 1.898468%; full80 deteriorates 1.477324%. It fails the joint objective and trails the selected full52 reference.
- R52-RMS-CONTINUOUS-VARIANCE-NODE-CORRECTION (retained): Retain as the fast full52 research reference, not a winner: full52 improves 2.948170%; full80 deteriorates 3.461463%.
- UF-CONDITIONAL-COVARIANCE (rejected): Full52 CRPS 0.59577166080711608 (-2.948170% vs M261); full80 CRPS 0.23387900572210277 (+3.156237% vs M261). It fails the joint objective; preserve the exact tested law and controls.
- UF-CONDITIONAL-MEAN (rejected): Full52 CRPS 0.59699804263973377 (-2.748391% vs M261); full80 CRPS 0.23385584849519025 (+3.146023% vs M261). It fails the joint objective; preserve the exact tested law and controls.
- UF-TOTAL-CONDITIONAL-COVARIANCE (rejected): Full52 CRPS 0.59577166080711608 (-2.948170% vs M261); full80 CRPS 0.23239664339923255 (+2.502416% vs M261). It fails the joint objective; preserve the exact tested law and controls.
- UF-CONVENTIONAL-MEAN-PLUS-COVARIANCE (rejected): Full52 CRPS 0.59699804263973377 (-2.748391% vs M261); full80 CRPS 0.2331864906759 (+2.850791% vs M261). It fails the joint objective; preserve the exact tested law and controls.
- UF-TOTAL-KERNEL-EXISTING-CONVENTIONAL-MEAN (rejected): Full52 CRPS 0.59699804263973377 (-2.748391% vs M261); full80 CRPS 0.23171189271702697 (+2.200395% vs M261). Fails the joint objective; retain the exact law and all controls as exploratory research evidence.

Archive receipt: [normalized archive closure](../../../src/simfolio_forecasting_methodology/resources/results/unified_frontier_interaction_archive_closure.json).

## UF-RESERVE-27c9c8aa0fafac4f

exact retained conditional-normalized rough future states, future conventional off, ORIGINAL nodes/curves/StudentDCC/accounting; no routing

| Panel | Status | Canonical C-order empirical CRPS by label | Raw-result scalar differences |
| --- | --- | --- | --- |
| Full 52 | complete_scored_and_audited | M261_saved_control: 0.61386958108984568; RMS_corrected_nodes_saved_control: 0.59577166080711608; RMS_original_nodes_saved_control: 0.59681892827927252; conditional_rough_no_conventional_original_nodes: 0.605745843721052; conventional_SV_corrected_variance_nodes: 0.60221546105451607; conventional_mean_plus_covariance: 0.59699804263973377; covariance_only: 0.59577166080711608; mean_only: 0.59699804263973377; rough_off_original_nodes_saved_control: 0.60331093641703915; total_covariance_only: 0.59577166080711608; total_kernel_existing_conventional_mean: 0.59699804263973377 | None |
| Full 80 | complete_scored_and_audited | M261_saved_control: 0.22672308848311776; RMS_corrected_nodes_saved_control: 0.23457102422540146; RMS_original_nodes: 0.23258808209819795; conditional_rough_no_conventional_original_nodes: 0.22869542481735233; conventional_SV_corrected_variance_nodes: 0.23007252374266968; conventional_mean_plus_covariance: 0.2331864906759; covariance_only: 0.23387900572210277; mean_only: 0.23385584849519025; rough_off_original_nodes_saved_control: 0.23051618147311073; total_covariance_only: 0.23239664339923255; total_kernel_existing_conventional_mean: 0.23171189271702697 | None |

Runtime: not_measured. Ordered shared-fit stage times remain diagnostic. No representative matched fresh/cached timing or authenticated production browser CTV proof.

Production: not validated.

Canonical panels: exploratory reused evidence. Seed robustness: not_tested. Independent post-selection: not_tested.

- UF-CONTROL-M261 (diagnostic): Unchanged exact saved baseline, kept separately on both panels.
- R52-ROUGH-OFF (diagnostic): Retained factorial reference; improves full52 but deteriorates full80.
- UF-RMS-ORIGINAL-NODES (rejected): Full52 improves 2.777569% versus M261; full80 deteriorates 2.586853%, failing the joint objective.
- UF-CONVENTIONAL-SV-VARIANCE-NODES (rejected): Full52 improves 1.898468%; full80 deteriorates 1.477324%. It fails the joint objective and trails the selected full52 reference.
- R52-RMS-CONTINUOUS-VARIANCE-NODE-CORRECTION (retained): Retain as the fast full52 research reference, not a winner: full52 improves 2.948170%; full80 deteriorates 3.461463%.
- UF-CONDITIONAL-COVARIANCE (rejected): Full52 CRPS 0.59577166080711608 (-2.948170% vs M261); full80 CRPS 0.23387900572210277 (+3.156237% vs M261). It fails the joint objective; preserve the exact tested law and controls.
- UF-CONDITIONAL-MEAN (rejected): Full52 CRPS 0.59699804263973377 (-2.748391% vs M261); full80 CRPS 0.23385584849519025 (+3.146023% vs M261). It fails the joint objective; preserve the exact tested law and controls.
- UF-TOTAL-CONDITIONAL-COVARIANCE (rejected): Full52 CRPS 0.59577166080711608 (-2.948170% vs M261); full80 CRPS 0.23239664339923255 (+2.502416% vs M261). It fails the joint objective; preserve the exact tested law and controls.
- UF-CONVENTIONAL-MEAN-PLUS-COVARIANCE (rejected): Full52 CRPS 0.59699804263973377 (-2.748391% vs M261); full80 CRPS 0.2331864906759 (+2.850791% vs M261). It fails the joint objective; preserve the exact tested law and controls.
- UF-TOTAL-KERNEL-EXISTING-CONVENTIONAL-MEAN (rejected): Full52 CRPS 0.59699804263973377 (-2.748391% vs M261); full80 CRPS 0.23171189271702697 (+2.200395% vs M261). Fails the joint objective; retain the exact law and all controls as exploratory research evidence.
- UF-CONDITIONAL-ROUGH-NO-CONVENTIONAL-ORIGINAL-NODES (rejected): Full52 CRPS 0.605745843721052 (-1.323365% vs M261); full80 CRPS 0.22869542481735233 (+0.869932% vs M261). Fails the joint objective; retain the exact law and all controls as exploratory research evidence.

Archive receipt: [normalized archive closure](../../../src/simfolio_forecasting_methodology/resources/results/unified_frontier_reserve_archive_closure.json).

## Targeted original M261 long control

The separate [M261 diagnostic reference](unified-frontier-m261-long-diagnostic.md) covers 240 original long contexts over 80 portfolios. It is not a new ranked model, a rerun of all 4,080 origins or a new full-panel score. Its restoration includes the actual observer 80:48 and the independent saved 240-record reconstruction. Historical observer/plain parity remains limited to the three original pilots.

## Full total mean and IID — closed study preservation

The two separately scored full52/full80 studies are independently audited and their private preservation is closed. Four method Tasks cover both studies and both panels; two old baseline seed traced/plain pairs bring the batch to eight Task forecasts across six contexts. All 12 old saved seed records and 53,352 CRPS values were rechecked without additional forecasts. This establishes restoration, not full seed robustness, independent post-selection validation, representative runtime or production readiness. Both new laws fail full80 eligibility and neither supplies a joint winner.

Research evidence only. Scores are separate by panel; compute timings are not browser CTV.

## UF-TOTALMEAN-1aff29ce9e6500cb

ONE totalK plus full conditional total-mean law; all52+80 newly forecast/scored; no routing, weights, kappa or profitability screen

| Panel | Status | Canonical C-order empirical CRPS by label | Raw-result scalar differences |
| --- | --- | --- | --- |
| Full 52 | complete_scored_and_audited | M261_saved_control: 0.61386958108984568; RMS_corrected_nodes_saved_control: 0.59577166080711608; RMS_original_nodes_saved_control: 0.59681892827927252; conditional_rough_no_conventional_original_nodes: 0.605745843721052; conventional_SV_corrected_variance_nodes: 0.60221546105451607; conventional_mean_plus_covariance: 0.59699804263973377; covariance_only: 0.59577166080711608; mean_only: 0.59699804263973377; rough_off_original_nodes_saved_control: 0.60331093641703915; total_covariance_only: 0.59577166080711608; total_kernel_existing_conventional_mean: 0.59699804263973377; total_kernel_full_conditional_total_mean: 0.59733582939777519 | None |
| Full 80 | complete_scored_and_audited | M261_saved_control: 0.22672308848311776; RMS_corrected_nodes_saved_control: 0.23457102422540146; RMS_original_nodes: 0.23258808209819795; conditional_rough_no_conventional_original_nodes: 0.22869542481735233; conventional_SV_corrected_variance_nodes: 0.23007252374266968; conventional_mean_plus_covariance: 0.2331864906759; covariance_only: 0.23387900572210277; mean_only: 0.23385584849519025; rough_off_original_nodes_saved_control: 0.23051618147311073; total_covariance_only: 0.23239664339923255; total_kernel_existing_conventional_mean: 0.23171189271702697; total_kernel_full_conditional_total_mean: 0.23155344406726108 | None |

Runtime: not_measured. Concurrent campaign and shared-fit stage times are diagnostic only; no matched fresh/cached timing or deployed-browser CTV proof.

Production: not validated.

Canonical panels: exploratory reused evidence. Seed robustness: not_tested. Independent post-selection: not_tested.

- UF-CONTROL-M261 (diagnostic): Unchanged exact saved baseline, kept separately on both panels.
- R52-ROUGH-OFF (diagnostic): Retained factorial reference; improves full52 but deteriorates full80.
- UF-RMS-ORIGINAL-NODES (rejected): Full52 improves 2.777569% versus M261; full80 deteriorates 2.586853%, failing the joint objective.
- UF-CONVENTIONAL-SV-VARIANCE-NODES (rejected): Full52 improves 1.898468%; full80 deteriorates 1.477324%. It fails the joint objective and trails the selected full52 reference.
- R52-RMS-CONTINUOUS-VARIANCE-NODE-CORRECTION (retained): Retain as the fast full52 research reference, not a winner: full52 improves 2.948170%; full80 deteriorates 3.461463%.
- UF-CONDITIONAL-COVARIANCE (rejected): Full52 CRPS 0.59577166080711608 (-2.948170% vs M261); full80 CRPS 0.23387900572210277 (+3.156237% vs M261). It fails the joint objective; preserve the exact tested law and controls.
- UF-CONDITIONAL-MEAN (rejected): Full52 CRPS 0.59699804263973377 (-2.748391% vs M261); full80 CRPS 0.23385584849519025 (+3.146023% vs M261). It fails the joint objective; preserve the exact tested law and controls.
- UF-TOTAL-CONDITIONAL-COVARIANCE (rejected): Full52 CRPS 0.59577166080711608 (-2.948170% vs M261); full80 CRPS 0.23239664339923255 (+2.502416% vs M261). It fails the joint objective; preserve the exact tested law and controls.
- UF-CONVENTIONAL-MEAN-PLUS-COVARIANCE (rejected): Full52 CRPS 0.59699804263973377 (-2.748391% vs M261); full80 CRPS 0.2331864906759 (+2.850791% vs M261). It fails the joint objective; preserve the exact tested law and controls.
- UF-TOTAL-KERNEL-EXISTING-CONVENTIONAL-MEAN (rejected): Full52 CRPS 0.59699804263973377 (-2.748391% vs M261); full80 CRPS 0.23171189271702697 (+2.200395% vs M261). Fails the joint objective; retain the exact law and all controls as exploratory research evidence.
- UF-CONDITIONAL-ROUGH-NO-CONVENTIONAL-ORIGINAL-NODES (rejected): Full52 CRPS 0.605745843721052 (-1.323365% vs M261); full80 CRPS 0.22869542481735233 (+0.869932% vs M261). Fails the joint objective; retain the exact law and all controls as exploratory research evidence.
- UF-TOTAL-KERNEL-FULL-CONDITIONAL-TOTAL-MEAN (rejected): Full52 CRPS 0.59733582939777519 (-2.693366% vs M261); full80 CRPS 0.23155344406726108 (+2.130509% vs M261). Full80 fails eligibility; no joint winner or production change.

Archive receipt: [normalized private archive closure](../../../src/simfolio_forecasting_methodology/resources/results/unified_frontier_totalmean_archive_closure.json).

## UF-IID-8544ad97dc7caf39

ONE IID conditional scale contrast; BOTH full panels fresh; no totalK, mean compensation, kappa, routing, grid or profitability screen

| Panel | Status | Canonical C-order empirical CRPS by label | Raw-result scalar differences |
| --- | --- | --- | --- |
| Full 52 | complete_scored_and_audited | M261_saved_control: 0.61386958108984568; RMS_corrected_nodes_saved_control: 0.59577166080711608; RMS_original_nodes_saved_control: 0.59681892827927252; conditional_rough_no_conventional_original_nodes: 0.605745843721052; conventional_SV_corrected_variance_nodes: 0.60221546105451607; conventional_mean_plus_covariance: 0.59699804263973377; covariance_only: 0.59577166080711608; iid_conditional_unit_second_moment_scale: 0.59961187910025593; mean_only: 0.59699804263973377; rough_off_original_nodes_saved_control: 0.60331093641703915; total_covariance_only: 0.59577166080711608; total_kernel_existing_conventional_mean: 0.59699804263973377; total_kernel_full_conditional_total_mean: 0.59733582939777519 | None |
| Full 80 | complete_scored_and_audited | M261_saved_control: 0.22672308848311776; RMS_corrected_nodes_saved_control: 0.23457102422540146; RMS_original_nodes: 0.23258808209819795; conditional_rough_no_conventional_original_nodes: 0.22869542481735233; conventional_SV_corrected_variance_nodes: 0.23007252374266968; conventional_mean_plus_covariance: 0.2331864906759; covariance_only: 0.23387900572210277; iid_conditional_unit_second_moment_scale: 0.22784056761333035; mean_only: 0.23385584849519025; rough_off_original_nodes_saved_control: 0.23051618147311073; total_covariance_only: 0.23239664339923255; total_kernel_existing_conventional_mean: 0.23171189271702697; total_kernel_full_conditional_total_mean: 0.23155344406726108 | None |

Runtime: not_measured. Concurrent campaign and shared-fit stage times are diagnostic only; no matched fresh/cached timing or deployed-browser CTV proof.

Production: not validated.

Canonical panels: exploratory reused evidence. Seed robustness: not_tested. Independent post-selection: not_tested.

- UF-CONTROL-M261 (diagnostic): Unchanged exact saved baseline, kept separately on both panels.
- R52-ROUGH-OFF (diagnostic): Retained factorial reference; improves full52 but deteriorates full80.
- UF-RMS-ORIGINAL-NODES (rejected): Full52 improves 2.777569% versus M261; full80 deteriorates 2.586853%, failing the joint objective.
- UF-CONVENTIONAL-SV-VARIANCE-NODES (rejected): Full52 improves 1.898468%; full80 deteriorates 1.477324%. It fails the joint objective and trails the selected full52 reference.
- R52-RMS-CONTINUOUS-VARIANCE-NODE-CORRECTION (retained): Retain as the fast full52 research reference, not a winner: full52 improves 2.948170%; full80 deteriorates 3.461463%.
- UF-CONDITIONAL-COVARIANCE (rejected): Full52 CRPS 0.59577166080711608 (-2.948170% vs M261); full80 CRPS 0.23387900572210277 (+3.156237% vs M261). It fails the joint objective; preserve the exact tested law and controls.
- UF-CONDITIONAL-MEAN (rejected): Full52 CRPS 0.59699804263973377 (-2.748391% vs M261); full80 CRPS 0.23385584849519025 (+3.146023% vs M261). It fails the joint objective; preserve the exact tested law and controls.
- UF-TOTAL-CONDITIONAL-COVARIANCE (rejected): Full52 CRPS 0.59577166080711608 (-2.948170% vs M261); full80 CRPS 0.23239664339923255 (+2.502416% vs M261). It fails the joint objective; preserve the exact tested law and controls.
- UF-CONVENTIONAL-MEAN-PLUS-COVARIANCE (rejected): Full52 CRPS 0.59699804263973377 (-2.748391% vs M261); full80 CRPS 0.2331864906759 (+2.850791% vs M261). It fails the joint objective; preserve the exact tested law and controls.
- UF-TOTAL-KERNEL-EXISTING-CONVENTIONAL-MEAN (rejected): Full52 CRPS 0.59699804263973377 (-2.748391% vs M261); full80 CRPS 0.23171189271702697 (+2.200395% vs M261). Fails the joint objective; retain the exact law and all controls as exploratory research evidence.
- UF-CONDITIONAL-ROUGH-NO-CONVENTIONAL-ORIGINAL-NODES (rejected): Full52 CRPS 0.605745843721052 (-1.323365% vs M261); full80 CRPS 0.22869542481735233 (+0.869932% vs M261). Fails the joint objective; retain the exact law and all controls as exploratory research evidence.
- UF-TOTAL-KERNEL-FULL-CONDITIONAL-TOTAL-MEAN (rejected): Full52 CRPS 0.59733582939777519 (-2.693366% vs M261); full80 CRPS 0.23155344406726108 (+2.130509% vs M261). Full80 fails eligibility; no joint winner or production change.
- UF-IID-CONDITIONAL-UNIT-SECOND-MOMENT-SCALE (rejected): Full52 CRPS 0.59961187910025593 (-2.322595% vs M261); full80 CRPS 0.22784056761333035 (+0.492883% vs M261). Full80 fails eligibility; no joint winner or production change.

Archive receipt: [normalized private archive closure](../../../src/simfolio_forecasting_methodology/resources/results/unified_frontier_IID_archive_closure.json).

The IID seed helper remains BLOCKED_PREPARED_NOT_RUN: its source-only comparison defect is not an established financial seed-law failure. The older 12-trial/24-forecast baseline integration is a separate evidence type. Nine exact historical source epochs remain authoritative. Superseded growth tables are not authority; separately pinned metadata gives 8766-day median/mean row-wise ratios IID0.940970/0.964253, M2610.838983/0.955149 and selected0.983538/0.988876. No matched full52 M261 long-growth baseline is available. These later annotations remain outside the63816 numerical members; growth is descriptive, with no return floor or routing.

# Corrected rough-node studies and conventional bridge — closed study preservation

Private compact preservation is verified; exact source/member/scientific scope follows the normalized closure resources. The private bulk archive has12 closed parts; six fixed restored Task path/CRPS oracles passed. Restoration uses the explicitly retained original46-distribution runtime, so this is not a standalone environment rebuild. Independent saved actual-science review passed. Fresh compact-lineage verification passed.

| Record | 52 CRPS / freshness | 80 CRPS / freshness | Decision scope |
|---|---|---|---|
| Conventional OFF, corrected continuous nodes |0.6048042045569166;2652 fresh origins|0.23045095600899104;4080 fresh origins|Fails joint score objective; exploratory reuse of evaluation panels|
| Conventional ON, corrected continuous nodes |0.6095093947133338;2652 fresh origins|0.22648879635922456;4080 fresh origins|Parked after speed prescreen; no production recommendation|
| Rough covariance on sampled conventional-SV corrected nodes |0.6022154610545161; inherited exact single-history identity,0 fresh forecasts|0.22904705025144675;4080 fresh origins|Full80 does not improve M261; retain tested law and controls|
| Conventional ON, original empirical nodes prototype |0.6105233636157572; historical aggregate,0 fresh forecasts|Not run|Six-case correctness pilot and18 accepted local timing trials; parked after runtime screen|

Detailed model/control scores, portfolio wins/losses, horizon bands and original task/seed/source bindings are preserved in the companion draft packet. Bridge52 inherits a one-history identity; the original-node prototype has no invented full80 score. All failed technical attempts, source corrections, the exact recovered historical Numba index and failed-part prefix witnesses remain in the private provenance chain.

Runtime evidence remains separately typed: original prototype18 accepted trials and bridge18 short+9 long trials are local representative diagnostics. Forecast parity, runtime prescreens, archive reconstruction and production/browser validation are different gates. No full14/site speed, seed robustness, independent post-selection or production validation is claimed. Existing seven unified studies,24 legacy refinement rows and selected80 must remain byte-exact when the final append is prepared.

Private provider data, raw returns, task arrays and native artifacts remain private release assets.

## Closing comparison and recommendation

Keep M261 for the strict joint score-and-FAST-runtime goal. No tested candidate qualifies. FAST retains the stronger52-panel tradeoff; the new original-node IID law's tiny80 mean advantage does not establish a production recommendation. Canonical panels remain exploratory, seed robustness and independent post-selection testing are absent, and production/browser validation is absent. The shared long-horizon CAGR shortfall around0.75 of training-history growth is a diagnostic concern shared with controls; no historical return floor or portfolio exclusion was applied.

Rough-only IID with sampled conventional SV: full6732 fresh Tasks (2652 full52 and4080 full80),240 paths. Full52 CRPS 0.6053046527780706: 1.3952358246142826% better than M261 but 1.600108329765093% worse than selected52. Full80 CRPS 0.22835434759659093: 0.7194940419994467% worse than M261. Rejected by the full80 accuracy gate; no joint winner. The27-trial local controlled runtime prescreen is separate from browser/production performance. Seed robustness and independent post-selection validation remain untested; production unvalidated. Private raw-byte/member preservation is verified; restored-law replay is not claimed for this study. All controls/rejected labels are retained in the preserved compact results; the corrected-node/bridge archive remains unchanged.

Original-node all-volatility IID: research preservation for the canonical/methodology catalogue. Private raw-byte/member preservation in the later batch is verified; this study is separate from the corrected-node/bridge batch.

All 6,732 fresh Tasks completed: 2,652 full52 and 4,080 full80, with 240 paths and all saved controls preserved. Full52 CRPS is 0.600668345431752: 2.1504951645684445% better than M261, but 0.8219062682508493% worse than selected52. Full80 CRPS is 0.22653257917294423: 0.08402730901740929% better than M261, with only 32/80 portfolio wins (48 losses). Both aggregate M261 comparisons pass under this fixed research seed; the small full80 edge on reused development contexts does not establish independent generalization or a production recommendation.

Execution retained byte-equality receipts for 2,378,566,080 original-versus-lean path values across all 6,732 Tasks. Raw path pairs were not retained; the saved auditor rechecked source-bound parity receipts and reconstructed daily losses/counts. This supports the stated all-work financial path comparison, while the full-run elapsed time includes both implementations and recording and is not controlled speed evidence.

The separate lean runtime prescreen has 27 local trials: three contexts, three fresh processes per model/context, and nine cached samples per model/context. Fresh median M261 speedups are 1.497447429428501 for single SPYSIM H8766, 1.1408168490091055 for six-asset annual-policy H126, and 1.209003066051268 for six-asset annual-policy H8766. Selected_fast remains faster in all three. Idle administrative controllers and paused-family rules are preserved in the saved SRIU audit; this is not strict raw quiet, CPU cleanroom, full14, browser/website/production latency or independent post-selection proof.

Growth uses 396 actual fresh candidate checkpoint records at exact temporal H8766, H5844 and H2922, annualized using 252 days per year. Plotted score bands are separate. Matched80 baseline diagnostics are available. Additive saved-only reconciliation now validates156 exact true-H full52 contexts per M261 and selected control, using624 retained raw loss/accounting files. It reconstructs source-bound stored growth and control score vectors; old full paths are unavailable and no new financial path replay is claimed. Full matched tables and limits are preserved in growth-supplement-v3.json. This supersedes the initial draft's pending full52 growth status; the original draft and historical audits remain unchanged. The historical selected/M261 central90-width ratio 1.048236 belongs to H2922; H8766 is 0.986556. Original historical artifacts remain unchanged. Training-return ratios and 75% counts are diagnostics, not forecast floors, exclusions, routing, or causal score decomposition.

Attempt1 completed zero origins because owned TMP directories were missing during initialization. Its retained terminal/excerpt is incomplete; no complete stderr is claimed. Attempt2 used the same scientific source, manifest and admission after the directories were created and completed all Tasks. Source, failed history, pilot/native provenance, accepted and provisional timing records remain required preservation inputs.

The original-versus-corrected all-vol IID contrast retains the same post-RMS finish-clip IID law. Comparison with fused rough-IID × conventional-ON also changes clipping/composition and cannot isolate conventional serial state. Seed robustness, independent post-selection validation and production validation remain untested. All control labels, portfolio scores, counts, horizon bands and audit references are retained in study-draft.json and source-history-refs.json.

Final recommendation: [strict speed and growth assessment](../../../src/simfolio_forecasting_methodology/resources/results/unified_frontier_bridge_publication/closing_evidence/2026-10-19-final-strict-speed-recommendation.md). Ancillary evidence: [typed closed experiments](../../../src/simfolio_forecasting_methodology/resources/results/unified_frontier_bridge_publication/closing_evidence/records-and-scope.json), [native compute audit](../../../src/simfolio_forecasting_methodology/resources/results/unified_frontier_bridge_publication/closing_evidence/selected_fast_cdcc_literal_bound_speed_actual_audit/review.json), [independent stopping review](../../../src/simfolio_forecasting_methodology/resources/results/unified_frontier_bridge_publication/closing_evidence/2026-10-19-independent-strict-fast-stopping-review.md).

The later private batch freshly verifies raw bytes and saved whole-panel evidence. Its six restored financial oracles cover the corrected-node/bridge dependency only, not a replay of these two IID laws. Bounded Markov/VC rejections, RMS/allocation/accounting experiments, native implementation parity and controlled context timings are linked as ancillary evidence, not new ranked models or broad website speed claims. An obsolete unexecuted Markov admission source epoch is unavailable; the receipt and executed corrected epoch are retained, and lossless historical source closure is explicitly not claimed.

Archive receipt: [UF-ROUGH-CORRECTED-OFF-021da156f7455bce](../../../src/simfolio_forecasting_methodology/resources/results/unified_frontier_corrected_OFF_archive_closure.json).

Archive receipt: [UF-ROUGH-CORRECTED-ON-9953676cd480585f](../../../src/simfolio_forecasting_methodology/resources/results/unified_frontier_corrected_ON_archive_closure.json).

Archive receipt: [UF-ROUGH-CONVENTIONAL-BRIDGE-4ce66184d7db04fd](../../../src/simfolio_forecasting_methodology/resources/results/unified_frontier_bridge_archive_closure.json).

Archive receipt: [UF-ROUGH-IID-b67722c7aaebc226](../../../src/simfolio_forecasting_methodology/resources/results/unified_frontier_rough_IID_archive_closure.json).

Archive receipt: [UF-ORIGINAL-NODES-IID-52d4c26b608ac1ac](../../../src/simfolio_forecasting_methodology/resources/results/unified_frontier_original_nodes_IID_archive_closure.json).
