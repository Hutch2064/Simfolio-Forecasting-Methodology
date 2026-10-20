# Closed development contrast: HARCH multi-timescale volatility

This portfolio-level model replaced only the current winner's120-path GARCH/FHS component by variance-targeted HARCH(1,5,22)/FHS. Exact120 raw random-block paths, smooth mean regularization, batch Gaussian mean uncertainty,240total paths, half whole-path weights, empirical filtered-shock sampling/seeds and native accounting remain. The filter changes the empirical residual values. No family/horizon routes, state count, lag grid or score-trained adjustment. [DESIGN.md](DESIGN.md) records the prospective financial and numerical law.

Six tests passed: conditional variances against the independent arch implementation, analytic gradient/independent constrained optimization, fit equivariance/empirical shock moments/explicit convergence failure, forecast against a separate Python history recursion, exact ARCH1 and constant-variance nested boundaries against the original kernel, and unchanged entire pool-wrapper identity under the original FHS substitution. All245 exact native training prefixes across49families/6classes qualified; both fixed numerical starts converged on every prefix.

Training residual-memory correlations after the original GARCH filter were weak: median.0071,.0035,-.0082 for1,5,22native-observation squared-return averages. These are descriptive overlapping training-prefix checks, not independent significance evidence or a demonstrated forecasting edge. Standard multi-timescale conditional-variance research and a different finite-memory law motivated this single frozen contrast; the field result does not justify a window grid.

| Same245 development origins | Primary better than worst Public comparator | Sensitivity better than worst Public comparator |
|---|---:|---:|
| Public Adaptive Blocks |0%|0%|
| Current full-panel winner |2.462925955589%|2.903057657891%|
| HARCH candidate |2.430364945170%|2.869193996820%|

Public Adaptive Blocks is the worst of the three retained Public comparators on complete fixed-panel coverage under both profiles. The candidate narrowly regresses against the unchanged current winner on both profiles and therefore fails the frozen development gate. No reserved/broad/full escalation, window/weight/objective/seed retuning or claim of universal HARCH inferiority. These are adaptively reused development cases, not independent post-selection or arbitrary mixed-portfolio production validation. See [scores.json](scores.json), [decision.json](decision.json), and every class/family/original-weighted horizon contribution in [diagnostics.json](diagnostics.json).

All735forecasts succeeded with0native failures and0scoring integrity errors. The audit verified735native task identities/loss-vector hashes, including490exact historical Public/current loss-hash replays. Every additive horizon decomposition reconciles the canonical primary/sensitivity aggregate.

The complete757-file private archive was freshly downloaded, every file disk-restored and SHA256verified before735 redundant local raw loss files were retired. [archive-receipt.json](archive-receipt.json) records exact recovery and digests; existing fixed inputs/native source archives remain necessary parents. This was a pilot, not a full attempt, so the permanent full-panel catalogue remains13unique fully attempted versions. The research goal remains active.
