# Versioned full-panel model catalogue

**24 unique model versions across preserved benchmarks; 31 completed forecast executions; 3 on the current expanded benchmark.**

The single official score is [V11 requested-horizon standardized CRPS](../daily-assets-portfolios-asset-v6-20261020/standardized-crps-v11-20261023/README.md). Historical normalizations and executions remain preserved; rescoring adds no financial model or forecast execution.

| Model | Successful / attempted | Standardized CRPS Q |
|---|---:|---:|
| Asset Joint Blocks — With Replacement | 320017 / 320017 | 0.651056749440 |
| Asset Joint Blocks — History Partition | 320017 / 320017 | 0.652235953173 |
| Request-Specific Endpoint Mixture —167 Recipes | 320017 / 320017 | 0.544202657107 |

The endpoint mixture has **16.41% lower aggregate standardized CRPS** than fixed WR. Zero forecast failures; full source/scale/cache replay matched exactly. These are adaptively reused research results, not untouched validation or guaranteed future improvement.

[catalogue.json](catalogue.json) preserves every historical definition, score, coverage, failure, runtime and recovery reference.
