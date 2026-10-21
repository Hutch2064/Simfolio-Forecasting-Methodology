# Current selector research status

Goal active. Frozen1,264-series panel and single main scoring protocol unchanged. Maximize full-panel accuracy regardless of cost/complexity; runtime secondary. Both TestFolio-style joint-asset references remain unchanged at1,000paths; with replacement remains the fixed worse reporting reference. No new full-panel winner or catalogue entry is claimed here.

## Complete matched355-case development result

All methods below use exactly the same355cases and1,000final paths. Adaptive procedures also use1,000inner paths.

| Method | MainQ(lower better) | Better than fixed reference |
|---|---:|---:|
| Fixed asset blocks with replacement | 0.800402286283 | Reference |
| Current research working parent | 0.713198729801 | 10.894966% |
| Learned predictive pooling | 0.690170424253 | 13.772057% |
| Hard selection | 0.721700627734 | 9.832763% |

Pooling advances; hard selection loses the working parent and does not advance. This is development evidence, not full-panel, significance or independent-validation evidence. Each request learns whole joint-path mixture weights from its own historical rolling forecasts. The initial166-option bank includes the prior trained model, three raw samplers and162filtered combinations of9volatility methods,3shock laws,3mean treatments and uncertainty on/off. No fixed percentage mixture, arbitrary maximum origin count or automatic prior-winner fallback is introduced. See implementation/DESIGN files for long-horizon support limitations and finite-ensemble allocation.

The initial precision attempt had704successful forecasts and3optimizer-status failures. The numerical correction accepts only a verified convex optimality certificate and continues uncertified iteration-limited solves without a total cap. All3failed cases replayed successfully; a successful multivariate request(case221)matched both prior loss vectors. Complete corrected coverage is710forecasts. Independent audit verifies710selected/710reference checkpoints, every origin/main score, original source aliases and saved convex coefficients. Original failed records remain preserved. The initial harness's nominal355anchor count included the3fits that failed before anchor replay; actual initial executed parent replays were352. This metadata correction does not change financial scores.

## Ongoing efficiency and separate-slice checks

Measured timing in the original development placed about94%of selection time in simulation. The shared implementation preserves mean-specific asset/portfolio states while sharing mean-independent volatility/innovation work. Four integrated tests cover all162filtered options, joint asset states, chunk behavior, full request traces, coefficients and learned weights. One actual8asset historical fold showed bitwise-identical166component terminal ensembles and a1.8758CPU/1.8833wall median ratio in3interleaved warm-code pairs. This is local simulation-phase timing only, not full-request or deployed performance.

A single shared12worker pool now replays all355development requests(710outer loss vectors and all inner component ensemble hashes)against the original references while computing learned-pool forecasts on a separate fixed355case slice. Both use1,000inner/final paths. The new implementation remains subject to complete workload parity before retention; partial results cannot promote it. The separate slice is also adaptively reused research data, not independent validation. Current scientific paths are asset_selector_batched/validation_and_reserved in the local research root. No final full-panel admission; original stopping requirement still applies.

## Preservation

All closed initial/corrected precision results, failed attempts, source, losses, weights, traces, audit and closed paired benchmark are in the linked private archive receipt:1,864files freshly downloaded, disk-restored andSHA256verified. Active writers and their inputs remain local. Large raw research files remain private; public sources, compact summaries and repository-linked recovery records are preserved here. Full-panel catalogue counts remain23unique methods/28executions; these development experiments add none.
