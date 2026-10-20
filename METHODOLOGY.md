# French551 balanced, training-scaled CRPS — score protocol v2

## Purpose and selection

This protocol evaluates marginal cumulative-log-return forecast distributions across every eligible daily lead while reducing the dominance of long lead ranges and high-variability return series. Its formula was selected from methodological considerations before examining rankings under the new formula. It is a declared benchmark objective, not a claim that one universally correct multi-horizon weighting exists. No asset-specific or model-specific weights are fitted.

The original French551 raw CRPS remains an immutable legacy diagnostic. Historical 52-asset and 80-portfolio scores retain their original definitions; they are not relabelled as v2 results. New derived results receive their own protocol identity and bind the original execution identities, input hashes, scale plan, and reporting code.

## Exact definition

For series i and forecast origin o, let n_io be the number of training observations, and s_io the sample standard deviation (ddof=1) of those training daily log returns. For an eligible integer observation horizon h, let L_ioh be the original exact empirical CRPS of the 240 cumulative-log-return forecast draws against the realized cumulative log return. The original self-inclusive M-squared pair denominator is retained.

Primary normalized loss:

    q_ioh = L_ioh / (s_io * sqrt(h + h*h/n_io))

Daily log-horizon weight:

    w_h = log(1 + 1/h)

For each complete series, first average q_ioh over all protocol-eligible origins at each h. Weight those cell means by w_h and divide by log(H_i+1), where H_i is the largest protocol-eligible horizon for the series. Average the resulting series scores equally. All daily leads have strictly positive weight. No missing tasks, failed origins, nonfinite losses, or zero/nonfinite scales may silently reduce a model's final denominator. Partial comparisons use the identical set of complete series for the compared models and disclose that set.

The weights are the exact masses of d(log h) over bins [h,h+1). They telescope to log(H_i+1). Thus intervals with equal ratios of endpoints carry equal weight on the declared observation-count axis. This does not claim uniform importance over calendar years or equal empirical information at different leads.

## Scale rationale and limits

Under a simple IID finite-variance reference with an unknown constant mean estimated from n past returns, the prediction-error variance of a future h-observation sum is sigma-squared times (h + h-squared/n). The training sample standard deviation estimates sigma. This is a common, model-independent reference unit, not an assumption imposed on the forecasting candidates and not a claim to recover the true variance of dependent financial returns.

Both numerator and denominator scale by the same positive factor under a change of return units, so the normalized loss is dimensionless. Using each origin's training history prevents use of future realized volatility or forecast performance in the denominator. Zero variance fails closed; no numerical epsilon or arbitrary cap is introduced.

Predeclared sensitivity: replace the denominator by s_io*sqrt(h), retaining the same log-horizon weights and aggregation. This ignores training-mean estimation uncertainty. It is a sensitivity diagnostic, not an alternative chosen after examining which model wins. Legacy raw CRPS is also retained.

## Scoring-rule interpretation

CRPS is proper for the forecast distribution being scored. Multiplication by a strictly positive constant determined from information available before the outcome, followed by fixed positive arithmetic weighting, preserves conditional propriety for the relevant marginal distributions. Applying this to the empirical 240-member ensemble retains the original finite-ensemble convention; it does not remove Monte Carlo error or turn the self-inclusive estimator into an unbiased estimator of an underlying continuous predictive law.

This training-reference normalization is NOT Bolin and Wallin's locally scale-invariant SCRPS. That distinct proper rule requires separate forecast-observation and forecast-forecast distance components plus a correction term. The original scalar loss checkpoints do not retain those separate components. Plain division by a candidate's forecast standard deviation is not an acceptable substitute.

## Evidence and reporting

Saved original loss vectors can be rescored without refitting. The same versioned sidecar consumes future atomic checkpoints. Validate hashes and execution identity before ingestion; use an idempotent derived index; preserve original loss arrays, status, and errors. Publish no full-panel score until every required origin and complete-series gate passes.

Report model skill as 100*(1-Q_model/Q_public), where the competing public model is the fixed comparison reference and both scores use identical complete series. Positive means better; negative means worse. This ratio is a descriptive comparison of aggregate proper losses, not an additional outcome-wise scoring rule used for fitting.

Report the full horizon coverage and eligible-origin counts. Long-horizon cells often contain very few origins; weighting does not create evidence. Nested horizons, overlapping test windows, and related equity portfolios are dependent. Do not attach IID standard errors or treat all cells as independent. Marginal CRPS does not assess joint path dependence, portfolio dependence, or every investor utility.

## Sources

- Gneiting and Raftery (2007), Strictly Proper Scoring Rules, Prediction, and Estimation: https://doi.org/10.1198/016214506000001437
- Bolin and Wallin (2023), Local Scale Invariance and Robustness of Proper Scoring Rules, Statistical Science: https://doi.org/10.1214/22-STS864 ; author full text https://arxiv.org/abs/1912.05642
- Capistran (2006), On Comparing Multi-Horizon Forecasts: https://doi.org/10.1016/j.econlet.2006.04.010
- Hyndman and Koehler (2006), Another Look at Measures of Forecast Accuracy: https://doi.org/10.1016/j.ijforecast.2006.03.001 (training-scale precedent for point errors, not a proof about probabilistic CRPS)
- Proper Scoring Rules for Multivariate Probabilistic Forecasts Based on Aggregation and Transformation (2025): https://ascmo.copernicus.org/articles/11/23/2025/index.html

The log-horizon measure and the IID training reference above are explicit application-level design choices. The cited papers support scoring-rule principles and scale concerns; they do not prescribe this exact combined Simfolio benchmark.
