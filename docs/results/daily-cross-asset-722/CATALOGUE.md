# Versioned full-panel model catalogue

**23 unique model versions across preserved benchmarks; 10 with completed attempts on the current benchmark.**

The current benchmark has one main score. Historical normalizations and all older executions remain in catalogue.json; scores from different benchmarks are not compared.

## Complete successful coverage

This is an evidence inventory, not a mixed-coverage ranking.

| Model | Successful / attempted | Native failures | Main Q | Worker seconds |
|---|---:|---:|---:|---:|
| Equal Mix — Stable Wealth Execution [V5 portfolio 1000] | 64049 / 64049 | 0 | 0.613773001182 | 5982.243 |
| Public — Adaptive Blocks [V5 portfolio 1000] | 64049 / 64049 | 0 | 0.621396240135 | 6080.621 |
| Public — Random Blocks Replacement [V5 portfolio 1000] | 64049 / 64049 | 0 | 0.621041673589 | 4668.629 |
| Public — History Cycles [V5 portfolio 1000] | 64049 / 64049 | 0 | 0.622044083318 | 5659.065 |
| TestFolio-style Asset Blocks — Without Replacement | 64049 / 64049 | 0 | 0.631050886626 | 12956.482 |
| TestFolio-style Asset Blocks — With Replacement | 64049 / 64049 | 0 | 0.632071545704 | 13316.043 |
| Asset Equal Mix / Smooth Gaussian Mean Uncertainty | 64049 / 64049 | 0 | 0.615007113519 | 11721.958 |
| Asset Circular Equal Mix / Smooth Gaussian Mean Uncertainty | 64049 / 64049 | 0 | 0.612497579635 | 11948.529 |
| Asset Fading Volatility / Equal Mix / Smooth Gaussian Mean Uncertainty | 64049 / 64049 | 0 | 0.611810828993 | 10604.135 |

## Incomplete paired results: native forecast failures

This is an evidence inventory, not a mixed-coverage ranking.

| Model | Successful / attempted | Native failures | Main Q | Worker seconds |
|---|---:|---:|---:|---:|
| Equal Mix — Original Wealth Execution [V5 portfolio 1000] | 64048 / 64049 | 1 | 0.613773021864 | 5359.706 |

Worker seconds are recorded compute, not a controlled speed comparison or production latency.

[catalogue.json](catalogue.json) retains every historical model definition, version, benchmark, matched reference, failure, runtime and complete repository-linked recovery evidence. Identical models rerun on another panel add executions, not model counts.
