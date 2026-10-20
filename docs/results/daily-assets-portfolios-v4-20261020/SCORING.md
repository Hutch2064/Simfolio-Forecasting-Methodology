# One main distributional score, version4

Human-authorized2026-10-20 expansion: individual assets plus economically distinct
mixed portfolios, collective portfolio influence at most50percent, one headline
score. Diagnostics guide research and have zero score/promotion influence.
Historical v3 primary/sensitivity results remain immutable, separately archived;
their old names and values are not silently rewritten.

## Main score

Use exact empirical CRPS of cumulative terminal log returns from the native240
paths, divided by training sample SD(ddof1)*sqrt(h), h being native observation
horizon. Apply the existing positive actual-calendar log interval weights,
log((d_h+1)/(d_previous+1)), normalized over eligible horizon dates. Average into
one origin scalar. Fit/scaling use history only. Zero/nonfinite training SD fails
closed; no future scale, score-selected epsilon or diagnostic penalty is used.

This chooses the former sqrt(h) sensitivity calculation as the sole v4 objective.
It compares predictive error in common training-volatility/horizon units without
an extra horizon/prefix discount from the constant-mean IID estimation-reference
term h²/n. The retained positive log-lead weights already balance multiplicative
horizon ranges. Uncertainty in the mean belongs in each model's predictive law;
v4 does not choose a normalization based on a model's claimed uncertainty. The
choice reflects this cross-asset, broad-horizon forecasting goal, not a claim
that sqrt(h) is a uniquely academic or universally optimal normalization.

Positive scaling fixed conditional on training data and positive ex-ante horizon
weights retain the proper marginal CRPS objective. This is not forecast-specific
division by predicted spread. The locally scale-invariant SCRPS is a different
proper scoring rule requiring additional forecast-distribution terms; it is not
equivalent to relabeling a saved scalar CRPS and is not implemented here.
Reference: Gneiting/Raftery(2007), Strictly Proper Scoring Rules, Prediction, and
Estimation, https://doi.org/10.1198/016214506000001437 . The paper supports proper
scores and matched forecast situations, not these exact benchmark weights.

## Portfolio cap and time support

For each calendar quarter retain the existing asset hierarchy: average origin
scalars per series, series per family, families per asset class, available asset
classes equally. In the portfolio branch average origin scalars per portfolio,
portfolios per predefined economic family, available portfolio families equally.
Both branches get half the quarter's weight when portfolio cases are eligible;
asset-only quarters retain full asset weight. Average these quarter scores
equally for ONE main Q. Adding more portfolios inside a family does not increase
that family's weight or the collective portfolio branch's50percent cap.

Freeze evaluation-quarter support from the original individual-asset schedule.
Portfolio origins use the same native rolling/temporal schedule and are admitted
only in those pre-known quarters. Record every calendar-only origin exclusion
before any new model forecast. This prevents portfolio-only quarters from gaining
100percent weight or introducing outcome-dependent support. All input history
is retained; no realized return or model score determines origin admission.
An incomplete matched comparison lacking individual-asset support fails closed,
rather than silently assigning portfolios more weight.

## Coverage, reporting and evidence

Compare models on identical successful task intersections, with partial results
labelled preliminary and native failures separate from score-integrity errors.
Report one Q and its percentage reduction versus the worst of the three Public
comparators on identical matched coverage. Select/report that reference once for
the benchmark's complete main score, not separately by horizon or candidate to
inflate apparent gains. Keep all three Public scores identifiable in the audit.
FAST's failures receive no synthetic penalty and cannot enter a full-coverage
leaderboard beside complete models. Existing models must be run under the new
frozen panel/score before a v4 winner or improvement is claimed.

Calibration/tail/CDF/subgroup diagnostics and runtime remain separate evidence;
they do not create extra scores to choose from. Model promotion requires lower
main Q on the frozen matched gates and complete panel before final promotion.
No significance, independence, arbitrary-portfolio universality or production
latency claim follows from a small benchmark improvement.
