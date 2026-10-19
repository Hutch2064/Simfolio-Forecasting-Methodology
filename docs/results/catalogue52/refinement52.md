# Short-horizon refinement evidence

This is an additive research-results sidecar for the 2026-10-18 short-horizon
mechanism work. It uses the existing M261 specification as its production
reference and does not add canonical model IDs, ranks, or production changes.
The independent `R52-*` research IDs are immutable evidence keys, not M IDs.

The [full JSON sidecar](../../../src/simfolio_forecasting_methodology/resources/results/refinement52.json)
retains full-panel scores, per-portfolio scores where they were produced,
validation state, methodology notes, source paths, and SHA-256 hashes. An
identical copy is in the [master catalogue](../../../../master/catalogs/refinement52_results.json).

The panels stay separate. Canonical52 uses 52 assets, 2,652 origins, 455,832
asset-horizon cells, and 240 paths per origin. It combines 2,496 rolling-origin
tasks with 156 long-split tasks. The historical80 validation panel uses 80
portfolios, 4,080 origins, 701,280 portfolio-horizon cells, and 240 paths per
origin. Both use the original M261 contexts and seeds and exact empirical CRPS
of cumulative terminal log returns. The sidecar does not blend the panels.

| Evidence | Immutable research ID | Canonical52 CRPS | Change vs M261 | Historical80 CRPS | Historical80 status |
| --- | --- | ---: | ---: | ---: | --- |
| M261 production reference | M261 | 0.6138695811 | — | 0.2267230885 | complete control |
| Rough multiplier removed | `R52-ROUGH-OFF` | 0.6033109364 | +1.7200% | 0.2305161815 | complete; 1.6730% worse than M261 |
| Conditional rough normalization | `R52-CONDITIONAL-ROUGH-NORMALIZATION` | 0.6105233636 | +0.5451% | — | not run |
| Conditional rough normalization with EWMA variance targeting | `R52-CONDITIONAL-ROUGH-EWMA` | 0.6172480346 | −0.5504% | — | not run |
| Conditional rough normalization without conventional multiplier | `R52-CONDITIONAL-ROUGH-NO-CONVENTIONAL` | 0.6057458437 | +1.3234% | — | not run |
| Conditional rough latent-state anchor | `R52-STATE-ANCHOR` | 0.6123371739 | +0.2496% | — | not run |
| Permanent training-estimated growth uncertainty | `R52-PERMANENT-GROWTH-UNCERTAINTY` | 0.6461301049 | −5.2553% | — | not run |
| Training-fitted mean-reverting growth uncertainty | `R52-FITTED-GROWTH-UNCERTAINTY` | 0.6045985090 | +1.5103% | — | not run |
| Deterministic RMS multiplier (existing neither-multiplier ablation) | `R52-DETERMINISTIC-RMS-ABLATION` | 0.5968189283 | +2.7776% | — | not run |
| Conditional-mean conventional multiplier | `R52-CONDITIONAL-MEAN-MULTIPLIER` | 0.6035776387 | +1.6766% | — | not run |
| Original-moment return-dispersion curve | `R52-ORIGINAL-MOMENT-VARIANCE-CURVE` | 0.6193050801 | −0.8854% | — | not run |
| Constant raw training log-mean location with fixed dispersion | `R52-RAW-TRAINING-LOG-MEAN-FIXED-DISPERSION` | 0.8295531899 | −35.1351% | — | not run |
| Coherent-growth identity control | `R52-COHERENT-GROWTH-IDENTITY-CONTROL` | — | — | — | stopped after 1,025/2,652 origins; unscored |

The percentages compare each canonical52 CRPS with the M261 canonical52 score;
positive values mean lower CRPS. Rough-off and the selected variance-only candidate have completed historical80 scores in a panel separate from canonical52. No 80-portfolio score is inferred for the other variants.
The permanent-growth run regresses on the full canonical52 panel. Fitted growth
improves on M261 but remains 0.2134% worse than rough-off. The coherent-growth
run was stopped because its upstream fit fixed the mean, making the proposed
state uncertainty identically zero; its checkpoints remain preserved without a
score. The completed variance integration has its own [method and audit](../../../../short-horizon-mechanisms/refinement52/variance/METHOD.md)
and [independent breadth analysis](../../../../short-horizon-mechanisms/refinement52/variance-independent-audit.json).
The deterministic RMS result improves CRPS by 1.0761% versus rough-off and wins
24 of 52 asset-level aggregates; deleting any one asset leaves the aggregate
improvement between 0.621% and 1.130%. This RMS setting is the existing
neither-multiplier ablation inside that integration, not a newly proposed
normalization correction. The conditional-mean multiplier changes the forecast
distribution and is 0.0442% worse than rough-off despite 25 asset wins. The
original-moment curve changes only return dispersion and is 3.7677% worse than
RMS overall (though it wins 30 assets); its source method records that two
numerical fits remain. The raw-location isolation replaces the daily location
with the raw training log mean while retaining M261 dispersion; it wins only
9 assets against rough-off and regresses by 37.5001% overall. These are complete
52-asset results. None has an 80-portfolio score or production registration.

A separate non-scoring [long-growth diagnostic](../../../../short-horizon-mechanisms/refinement52/long-growth-audit.json)
records all three original canonical52 long-split contexts for the completed
variants (audit SHA-256 `1a39308b210a3bf67b579a0e7ee1b5b88dc9a6e476f956e8555ad53d9a8d7a64`).
At the longest 8,766-day horizon, the median across positive-history assets of
the per-asset median-path-CAGR/history-CAGR ratio was 0.7467 for rough-off and
0.7484 for M261; the corresponding mean-path-CAGR ratios were 0.7506 and
0.7546. Of 46 positive-history assets, 24 and 21 rough-off forecasts (median
and mean path CAGR) were below the descriptive 75% guide, versus 23 and 20 for
M261. No positive-history forecast was nonpositive or below 50% of historical
growth. These overlapping contexts are behavior diagnostics, not independent
evidence or a historical-growth floor; the 75% and 50% guides exclude no asset,
and this audit establishes no new 80-portfolio growth evidence.

The long-growth report adds 156 contexts each for RMS, conditional mean,
original-moment dispersion, and raw location. At 8,766 days, their positive-
history median-path/mean-path CAGR ratios were 0.7502/0.7540, 0.7507/0.7541,
0.7475/0.7531, and 0.9925/0.9940. The first three have 22/20, 22/20, and
26/21 of 46 positive-history assets below the descriptive 75% guide; raw
location has none. None of the four has a positive-history forecast below 50%
or nonpositive. The ratios show a moderate long-growth shortfall for the first
three variants, not an extreme collapse; all flags remain descriptive and no
asset or result is excluded.

The [separate local speed benchmark](../../../../short-horizon-mechanisms/speed-benchmark/results.json)
still covers its original fixed compute cases. A later matched [RMSruntime
benchmark](../../../../short-horizon-mechanisms/refinement52/benchmarks/rms-comparison/results.json)
ran 72 fresh-process trials and 288 samples across six predetermined
single-asset contexts. RMSruntime produced byte-identical daily path hashes to
the RMS candidate in every sample. Its median speedup across the six contexts
was 3.2191× versus production and 2.0313× versus rough-off in fresh-fit trials;
the cached-fit medians were 4.4844× and 3.0582×. These are local compute
measurements, not an all52 timing average or website click-to-visible result.
A separate [full-panel rough-off adapter parity audit](../../../../short-horizon-mechanisms/refinement52/runtime/stage-profile.json)
compared all 937,010,880 daily path values and all 455,832 cells byte-for-byte;
its concurrent-worker timings remain diagnostic, and it does not establish
full-panel RMSruntime parity. The selected candidate has a separately audited historical80 score; neither panel changes production registration or deployment.

`short-horizon-mechanisms/refinement52/catalogue_preservation.py` rebuilds and
validates both repository sidecars from the completed receipts. The packaged
loader is
`simfolio_forecasting_methodology.refinement52_results.load_refinement52_results()`;
it returns this separate extension payload without merging it into catalogue
ranks. The loader validates IDs, panel separation, completion state, and source
hash bindings; the builder also checks the underlying bytes and checkpoint
receipts. The exact user objective is copied below for durable context.

<!-- refinement52-additional-results:start -->
## Additional completed refinements

**Current gate status (2026-10-18).** The variance-only node-spread candidate has the lowest completed canonical52 CRPS (0.5957716608), passes exact output parity on all 2,652 contexts, and is faster than production and rough-off in all six prespecified local benchmark contexts for both fresh and cached fits. Fresh-fit median speedups are 2.8646x and 1.7000x; cached-fit medians are 4.1387x and 2.9260x. The timing scope is six fixed local compute contexts, not an all52 timing average or deployed click-to-visible result; earlier open-gate statements in this report are superseded by this status update. No production candidate is registered.

These independently identified candidates were completed after the original sidecar rows. Canonical52 scores remain the primary refinement panel. Rough-off and the selected variance-only candidate have separate completed historical80 scores; other refinements remain `not_run` on that panel. The fixed RMS result is the canonical52 statistical comparison target; speed evidence remains separately scoped below.

| Research ID | Canonical52 CRPS | vs M261 | vs rough-off | vs RMS | Historical80 |
| --- | ---: | ---: | ---: | ---: | --- |
| `R52-ROUGH-OFF-CONTINUOUS-NODE-MOMENTS` | 0.6112011019 | +0.4347% | -1.3078% | -2.4098% | not run |
| `R52-RMS-CONTINUOUS-NODE-MOMENTS` | 0.6060690401 | +1.2707% | -0.4572% | -1.5499% | not run |
| `R52-RMS-CONTINUOUS-VARIANCE-NODE-CORRECTION` | 0.5957716608 | +2.9482% | +1.2497% | +0.1755% | 0.2345710242; -3.4615% vs M261, -1.7590% vs rough-off |
| `R52-CONDITIONAL-GROWTH-STATE-MEAN` | 0.5985909296 | +2.4889% | +0.7824% | -0.2969% | not run |
| `R52-CONDITIONAL-GROWTH-STATE-MEAN-UNCERTAINTY` | 0.6018202228 | +1.9629% | +0.2471% | -0.8380% | not run |
| `R52-ORIGINAL-MOMENT-CONTINUOUS-NODE-MOMENTS` | 0.6272794957 | -2.1845% | -3.9728% | -5.1038% | not run |
| `R52-RMS-STATIONARY-RESIDUAL-ORIGINAL-NODES` | 0.6151235850 | -0.2043% | -1.9580% | -3.0670% | not run |
| `R52-RMS-STATIONARY-RESIDUAL-CONTINUOUS-NODE-MOMENTS` | 0.6233143457 | -1.5386% | -3.3156% | -4.4394% | not run |
| `R52-RMS-STATIONARY-RESIDUAL-SIGNED-Z-ORIGINAL-NODES` | 0.6180617065 | -0.6829% | -2.4450% | -3.5593% | not run |

All rows use the same 52-asset panel: 2,652 origins, 455,832 equally weighted asset/horizon cells, and 240 paths per origin. Positive percentages mean lower CRPS. Score reductions are source-bound to the saved complete result receipts and cells; the selected variance-only candidate alone also has a separately scored historical80 panel; no other candidate's 80-portfolio result is inferred. These experiments do not modify master M definitions, ranks, or production settings.

The two conditional-growth rows add the training-filtered terminal mean-state deviation to the unchanged M261/RMS point curve, with the second also adding coherent fitted AR(1) state uncertainty. Both improve on rough-off but lose to RMS; a separate read-only assessment closes only this growth-state branch and leaves its explicitly unrun RMS zero-centered-uncertainty factorial cell unscored. The original-variance/continuous-node result, both stationary residual-rank results, and all three direct-Student variance variants also lose to RMS. The best canonical52 score is the variance-only node-spread correction at 0.5957716608 versus RMS at 0.5968189283. Continuous-node corrections are pre-clipping and do not assert exact post-clipping moments or conditional-copula uniformity. No historical-growth floor is imposed.

The sidecar adds 2496 saved long-growth contexts across 16 additional refinements, with 156 per refinement at the original 2,922-, 5,844-, and 8,766-day holdouts. Ratios, 75%/50% counts, and nonpositive counts are descriptive selection diagnostics only; they exclude no assets. The RMS saved-metadata audit explicitly records `historical_floor_enforced: false`.

An independent checkpoint-to-cell rebuild verifies the continuous-node studies byte-for-byte; the stationary residual and direct-Student studies retain independent cell rebuilds, checkpoint identities, and source/control hashes. Full daily path arrays were not retained for the residual and direct-Student studies. The selected node-spread runtime benchmark reports fresh-fit medians of 2.8646x versus production, 1.7000x versus rough-off, and 1.0024x versus RMS runtime; cached-fit medians are 4.1387x versus production, 2.9260x versus rough-off, and 1.0180x versus RMS runtime. Full-panel output parity is independently verified over all 2,652 contexts; the speed measurements are local compute evidence from six fixed contexts, not an all52 timing average or deployed click-to-visible measurement. The selected candidate is essentially tied with RMS runtime here, so the benchmark does not attribute the speedup to the score-improving spread correction alone.

Private archival status: `verified_private_online_archive`. A checksum-bound private-release receipt and fresh restored scientific smoke are recorded.

The full hashes, per-asset canonical52 scores, methods, validation state, control cells, runtime scope, factorial diagnostic and archive receipt fields are in the [extension JSON sidecar](../../../src/simfolio_forecasting_methodology/resources/results/refinement52.json). Load it with `simfolio_forecasting_methodology.refinement52_results.load_refinement52_results()`; it does not merge into ranked catalogue IDs.
<!-- refinement52-additional-results:end -->


The selected variance-only candidate has a separate, independently audited historical80 score of **0.2345710242**, -3.4615% versus M261 and -1.7590% versus rough-off. It covers all 80 portfolios, 4,080 origins, 701,280 horizon cells, and 240 paths per origin. The newly verified selected80 archive has its own receipt; the earlier H archive is only an external source/data dependency for this result. This remains research evidence, not a production registration or deployment.

## User-requested objective (verbatim)

/goal. I need you to take our existing overall best candidate baseline (the current production with rough vol removed) and attempt to maximize the score improvement for the 52 asset test specifically. attempt to continue to find out tweaks to improve on its current weaknesses and break it down by patters and don’t overfit based on specific assets, rather find patterns and test logical next methodological tweaks to improve overall score. Do not stop until you don’t think there is any other non-speculative variant to test. 

Prefer fewer arbitrary parameters and data-driven structure over fixed parameters. Check that long-horizon median/mean CAGR does not systematically collapse far below historical CAGR across many assets/portfolios. These guide selection, not exclusion from the sweep.

The goal should be a faster candidate than production and faster than the model that’s production without rough vol as well, and scores better than both, and follow the stopping requirement. 

Save valid completed results, methodology and relevant details in BOTH canonical and master catalogues. Keep 52-asset scores, historical 80-portfolio scores and validation status distinct. Follow existing verified online archival/local-cleanup policy.

For independent bounded work, use Luna 6 xhigh/max subagents; use Sol 6.1 medium/high for complex, detailed implementation.
