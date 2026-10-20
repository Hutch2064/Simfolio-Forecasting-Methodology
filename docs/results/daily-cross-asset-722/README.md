# Expanded daily panel: full-panel model catalogue

**13 unique model versions; 12 with a failure-free full-panel result.**

This is an evidence inventory, not a shared-coverage leaderboard. Scores are lower-is-better. Public deltas use the exact matched successful tasks. The FAST comparison is incomplete/preliminary because of native fitting failures.

| Model | Active | Successful / attempted | Native failures | Primary Q | Sensitivity Q | Worker seconds |
|---|---|---:|---:|---:|---:|---:|
| Variance-targeted GARCH FHS + Batch Student Mean | no | 36813 / 36813 | 0 | 0.599161111381 | 0.613940596563 | 891.902 |
| Public — Equal Mix / Gaussian Mean Uncertainty | no | 36813 / 36813 | 0 | 0.592439895851 | 0.606923689656 | 950.610 |
| Equal Bootstrap / GARCH FHS Student Mean Pool | no | 36813 / 36813 | 0 | 0.592588913726 | 0.607091273478 | 932.433 |
| Equal Mix / Stratified Student Mean Uncertainty | no | 36813 / 36813 | 0 | 0.592440484137 | 0.606925326409 | 974.555 |
| Public — Random Blocks (Replacement) | yes | 36813 / 36813 | 0 | 0.597877124904 | 0.612663191657 | 853.391 |
| FAST — Exact 52-panel accuracy winner | yes | 36796 / 36813 | 17 | 0.607842185807 | 0.622990801348 | 6555.203 |
| Public Adaptive + Gaussian HAC Mean Uncertainty | no | 36813 / 36813 | 0 | 0.598919935068 | 0.613462628006 | 931.655 |
| Public Adaptive + Batch Student Mean Uncertainty | no | 36813 / 36813 | 0 | 0.598883643015 | 0.613391792524 | 832.123 |
| Random Blocks + Gaussian HAC Mean Uncertainty | no | 36813 / 36813 | 0 | 0.596956500500 | 0.611365840332 | 1104.201 |
| Random Blocks + Batch Student Mean Uncertainty | no | 36813 / 36813 | 0 | 0.596953113003 | 0.611337581020 | 778.455 |
| Public — Adaptive Blocks | yes | 36813 / 36813 | 0 | 0.599344204327 | 0.614082773465 | 1329.277 |
| Public — Random Blocks (History Cycles) | yes | 36813 / 36813 | 0 | 0.598575405968 | 0.613661472130 | 1016.074 |
| Equal Mix — Shrunk Mean / Gaussian Uncertainty | yes | 36813 / 36813 | 0 | 0.590960694290 | 0.605261229867 | 951.209 |

Worker seconds are summed per-forecast fitting/simulation/scoring measurements, not elapsed campaign time, website latency, or a controlled speed comparison.

See [catalogue.json](catalogue.json) for exact values, model definitions, source versions, both profiles, matched Public scores, failures, runtime, execution fingerprints and archive links. [SCORING.md](SCORING.md) freezes the scoring contract; [RESTORE.md](RESTORE.md) explains recovery.

Current research baseline: **Equal Mix — Shrunk Mean / Gaussian Uncertainty**. Its [full-panel evidence](research/smooth-mean-win-20261020/README.md) includes all class/family/horizon results and limitations. The accuracy-maximization search remains active.
