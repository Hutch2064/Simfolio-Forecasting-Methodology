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
