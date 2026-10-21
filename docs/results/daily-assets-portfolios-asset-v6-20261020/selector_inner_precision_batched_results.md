# Controlled internal-precision study

All 1,728 candidate forecasts and 216 reference forecasts were independently checked from saved losses. There were no native forecast, scoring-integrity or infrastructure errors. All final forecasts use 1,000 paths. The 72 representative requests are adaptively reused development evidence, not the full panel or independent validation.

The table shows the path-scoring criterion. Each score is the fixed canonical main score, averaged descriptively across the three predeclared Monte Carlo seed executions. Lower Q is better; every improvement references the same with-replacement baseline on identical coverage.

| Internal paths | Mean Q | Better than baseline | Seed SD of Q | Median attributed CPU seconds | Maximum CPU seconds |
| --- | --- | --- | --- | --- | --- |
| 10 | 0.765328339 | 20.77511% | 0.005096 | 2.342 | 23.864 |
| 30 | 0.756306561 | 21.70903% | 0.005144 | 4.778 | 32.657 |
| 100 | 0.760880409 | 21.23555% | 0.002822 | 14.576 | 61.455 |
| 300 | 0.761796507 | 21.14072% | 0.003789 | 41.920 | 190.105 |

The preserved 1,000-internal-path path-scoring reference has Q=0.760166313, 21.30947% better than the baseline. It has only one internal seed replication and an older implementation, so this is neither a repeated-seed stability certificate nor an isolated timing comparison.

Thirty internal paths had the lowest mean path-scoring Q among this initial grid. Use 30 for the next matched per-fold training-SD normalization screen, with replication zero fixed in advance rather than selecting the best seed. This is an exploratory research setting, not a production cap, a full-panel promotion, or proof that further precision cannot help.

Runtime attribution charges the shared two-criterion preparation to each standalone output. It includes both objective optimizations, union refits, final simulation, scoring, checkpoint compression/hashing and warmup; it excludes blocked IO wait and record writing after measurement. Some requests exceed ten seconds, so the runtime target has not been met. Full runtime scope is recorded in the verification JSON.

The next contrast changes only historical-fold training-SD normalization. Keep the bank, folds, seeds, final 1,000 paths, outer score, panel and with-replacement reference fixed. Neither diagnostics nor usage frequency add score weight. The goal remains active; no stopping condition is claimed.

Implementations, compact results and verification are published in the research repository. Complete local records and loss checkpoints remain in use by the followup and have not been removed. This compact public publication is not a claim of remote restoration of private raw evidence.
