# History-only horizon-matched predictive pooling

This research candidate compares hard selection and convex predictive-distribution
pooling using identical internally generated forecasts. It does not change panel
data, outer outcomes, the fixed baseline, or canonical main scoring.

For each request, fit each eligible option to expanding prefixes of the available
history. Score terminal cumulative portfolio log return at the requested horizon.
When that horizon cannot provide two complete nonoverlapping validation blocks,
use the longest horizon that can; if the history cannot supply two one-day blocks,
disclose the single available block. Include every complete nonoverlapping block,
anchored at the end of history. Two blocks are a minimum to observe variation,
not a statistical stability guarantee or maximum origin count. The earliest prefix
may be only 24 observations for fitted options, a weakness to assess explicitly.

For component empirical distributions F_i with 1,000 members each, retain
A_i=E|X_i-y| and B_ij=E|X_i-X_j|. Averaged across folds, the empirical mixture CRPS
is A'w - (w'Bw)/2 on the probability simplex. Solve this convex problem, checking
the simplex KKT gap and that it is no worse than the best vertex to numerical
precision. Hard selection uses that best vertex. All fitting/selection uses only
the current request's past. Outer panel scores never enter request selection.

Final simulation draws coherent entire joint asset paths from each selected
component, using within-path portfolio accounting. Component counts use largest
remainders in even pairs to satisfy existing antithetic mean construction. Tiny
weights can disappear in the final 1,000-member ensemble; final path counts and
ideal weights are both recorded. The inner empirical-mixture optimum does not
guarantee that its rounded, newly simulated outer ensemble improves forecasts.

The initial bank is 19 options: the unchanged trained leader, both raw block
configurations, and pure constant/EWMA/GARCH/RMS-volatility filters with Gaussian
or empirical joint shocks and sample/shrunk means. This is an initial bank, not
the complete prospective universe. All fitted options currently include joint
mean uncertainty. Unsupported fits are disqualified for the whole request and
reported; raw options still receive actual historical validation. There is no
automatic prior-winner fallback or maximum number of validation origins.

This terminal-horizon, nonoverlapping validation rule is a new hypothesis. It is
not exact parity to the earlier all-origin/all-supported-lead reference. The outer
canonical score still evaluates all leads and aggregation levels unchanged.
Development samples are adaptively reused: no independent-validation,
significance, or superiority guarantee is claimed. Shared selection/refit timing
must not be presented as standalone timing for each mode.

Academic motivation:

- Yao et al., predictive distribution stacking, https://arxiv.org/abs/1704.02030.
  Their general proper-score rationale motivates pooling; chronological rather
  than ordinary exchangeable leave-one-out validation is used here.
- Dombry and Zaoui, CRPS model selection/convex aggregation,
  https://arxiv.org/abs/2411.13974. Their iid theoretical guarantees are not claimed
  for financial time series or these adaptively reused panel samples.
- Krueger et al., simulation-based predictive-distribution evaluation,
  https://arxiv.org/abs/1608.06802. Finite ensemble approximation is explicit.

Admission requires numerical/causal/joint-path tests, then unchanged frozen
72-case/70-family matched outer screening. Official 1,000-path full-panel
evaluation remains subject to the actual goal's stopping and admission rules.
