# Versioned full-panel model catalogue

**13 unique model versions across preserved benchmarks; 5 with completed attempts on the current benchmark.**

The current benchmark has one main score. Historical normalizations and all older executions remain in catalogue.json; scores from different benchmarks are not compared.

## Complete successful coverage

This is an evidence inventory, not a mixed-coverage ranking.

| Model | Successful / attempted | Native failures | Main Q | Worker seconds |
|---|---:|---:|---:|---:|
| Public — Random Blocks (Replacement) | 64049 / 64049 | 0 | 0.622556569111 | 1123.164 |
| Public — Adaptive Blocks | 64049 / 64049 | 0 | 0.623504537655 | 1696.626 |
| Public — Random Blocks (History Cycles) | 64049 / 64049 | 0 | 0.623475021676 | 1353.680 |
| Equal Mix — Shrunk Mean / Gaussian Uncertainty | 64049 / 64049 | 0 | 0.615397619320 | 1328.921 |

## Incomplete paired results: native fitting failures

This is an evidence inventory, not a mixed-coverage ranking.

| Model | Successful / attempted | Native failures | Main Q | Worker seconds |
|---|---:|---:|---:|---:|
| FAST — Exact R52 Accuracy Winner | 64019 / 64049 | 30 | 0.623936560374 | 8793.238 |

Worker seconds are recorded compute, not a controlled speed comparison or production latency.

[catalogue.json](catalogue.json) retains every historical model definition, version, benchmark, matched reference, failure, runtime and complete repository-linked recovery evidence. Identical models rerun on another panel add executions, not model counts.
