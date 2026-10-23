# Versioned full-panel model catalogue

**24 unique model versions across preserved benchmarks; 31 completed forecast executions; 3 on the current expanded benchmark.**

The single official score is [V11 requested-horizon standardized CRPS](../daily-assets-portfolios-asset-v6-20261020/standardized-crps-v11-20261023/README.md). Historical normalizations and executions remain preserved; rescoring adds no financial model or forecast execution.

| Model | Successful / attempted | Standardized CRPS Q |
|---|---:|---:|
| Asset Joint Blocks — With Replacement | 320017 / 320017 | 0.651056749440 |
| Asset Joint Blocks — History Partition | 320017 / 320017 | 0.652235953173 |
| Request-Specific Endpoint Mixture —167 Recipes | 320017 / 320017 | 0.544202657107 |

On the unchanged mixed-coverage panel, the endpoint mixture’s WR-specific result is **16.4124% lower aggregate standardized CRPS** than fixed WR (Q 0.544202657107 vs 0.651056749440). Under the approved global-worse rule, WOR is the selected reference (Q 0.652235953173), giving the canonical headline **16.5635% lower than WOR**. The forecasts, coverage, and Q values are unchanged; this is a reporting-reference amendment. Zero forecast failures; full source/scale/cache replay matched exactly. These are adaptively reused research results, not untouched validation or guaranteed future improvement.

## Running full-panel execution

A new G2 uniform-lead 2P execution is admitted and running against the unchanged V11 scoring protocol, alongside the fixed WR and WOR controls. Each arm targets 320,017 matched requests at 240 paths. The root-observed start at 2026-10-23 16:40:19 UTC recorded 196 of 64,049 case groups committed and zero failures. No new full score is available yet; the existing headline above and completed catalogue counts remain unchanged. The pending entry in [catalogue.json](catalogue.json) binds the plan, source review, admission, execution manifest, and start receipt by SHA-256.

[catalogue.json](catalogue.json) preserves every historical definition, score, coverage, failure, runtime and recovery reference.
