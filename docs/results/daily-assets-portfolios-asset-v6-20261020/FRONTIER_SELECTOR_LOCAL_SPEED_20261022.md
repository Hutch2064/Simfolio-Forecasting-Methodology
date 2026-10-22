# Local Frontier vs research selector timing study

This is a local model-compute diagnostic. It does not measure browser click-to-visible latency, production request latency, or model accuracy/parity. The research selector remains undeployed.

## Completed workload and method

The study contains 90 independently reviewed method-runs: 72 base runs across 1- and 3-asset portfolios, horizons of 5, 252, and 12,600 trading days, and 240 or 1,000 paths; plus 18 wide runs across 50, 100, and 200 assets at 252 trading days and 1,000 paths. Each completed workload has three interleaved repetitions per method (45 paired repetition comparisons across 15 workload configurations). The independent review passed for this completed scope. The base receipt is complete; the wide receipt remains intentionally incomplete.

Financial-fit caches were cleared before each trial, while numerical kernels were warmed. The Frontier method includes 15 bps turnover costs; the research selector uses zero costs. Their internal concurrency settings differ, so these are not equal-core comparisons. Results report medians of elapsed wall time and process CPU time separately. Repeated path hashes matched within each method and workload; cross-method path parity was not tested.

## Median timing results

Ratios are Frontier divided by selector, so values above 1 indicate a lower selector median for that metric. Times are seconds. The JSON file contains full-precision medians and maxima.

| Assets | Horizon (trading days) | Paths | Frontier wall | Selector wall | Wall ratio | Frontier CPU | Selector CPU | CPU ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 240 | 0.056 | 0.089 | 0.635 | 0.057 | 0.089 | 0.639 |
| 1 | 5 | 1000 | 0.068 | 0.102 | 0.665 | 0.068 | 0.101 | 0.669 |
| 1 | 252 | 240 | 0.061 | 0.059 | 1.038 | 0.062 | 0.059 | 1.057 |
| 1 | 252 | 1000 | 0.080 | 0.060 | 1.328 | 0.083 | 0.060 | 1.384 |
| 1 | 12600 | 240 | 0.221 | 0.096 | 2.290 | 0.299 | 0.096 | 3.103 |
| 1 | 12600 | 1000 | 0.622 | 0.247 | 2.519 | 0.932 | 0.245 | 3.803 |
| 3 | 5 | 240 | 0.102 | 0.166 | 0.614 | 0.233 | 0.165 | 1.410 |
| 3 | 5 | 1000 | 0.149 | 0.170 | 0.879 | 0.305 | 0.170 | 1.793 |
| 3 | 252 | 240 | 0.120 | 0.110 | 1.098 | 0.268 | 0.109 | 2.448 |
| 3 | 252 | 1000 | 0.204 | 0.119 | 1.705 | 0.386 | 0.119 | 3.258 |
| 3 | 12600 | 240 | 0.504 | 0.179 | 2.811 | 1.010 | 0.179 | 5.638 |
| 3 | 12600 | 1000 | 1.691 | 0.482 | 3.508 | 3.419 | 0.477 | 7.166 |
| 50 | 252 | 1000 | 3.395 | 1.223 | 2.776 | 14.049 | 1.221 | 11.508 |
| 100 | 252 | 1000 | 7.840 | 2.350 | 3.336 | 38.354 | 2.346 | 16.347 |
| 200 | 252 | 1000 | 25.092 | 5.047 | 4.972 | 166.325 | 5.012 | 33.189 |

The selector is slower in several short, small-portfolio cases; the larger-portfolio and longer-horizon cases show lower local medians in this tested sample. This pattern is diagnostic, not an overall model or product win.

## Stopped extension and evidence

The user stopped the final planned wide extension before the 200-asset, 12,600-trading-day case completed. Its six planned method-runs are excluded and no runtime is reported. The wide receipt therefore remains `complete: false`; the independent review applies only to the 90 finished method-runs. The earlier native warmup initialization failure occurred before timing and was resolved before the measured wide runs; both failure and repair records are retained.

The serving source revision was `56a7a4d6643f84b1c72a56ed274c76c55b92c6bf`; the production source files matched that revision. No production code or deployment was changed. The local benchmark does not establish production speed, browser CTV, accuracy, or cross-method numerical parity.

Full private inputs, harnesses, raw measurements, provenance, and failure/cancellation evidence are preserved in the [private evidence release](https://github.com/Hutch2064/simfolio-engine/releases/tag/research-frontier-selector-speed-v1-20261022T053136Z). Private repository access is required. Archive receipt SHA-256: `487ea6bff883fedcba5744f41aa2e73935f22a5def7169a826982b0fab67e330`; inventory SHA-256: `992e8eb5f2f6fdf28e492f9c5e99750b245b8888791b5e0607cab19dcc1a86dc`.

Machine-readable results: [`frontier_selector_local_speed_20261022.json`](frontier_selector_local_speed_20261022.json).
