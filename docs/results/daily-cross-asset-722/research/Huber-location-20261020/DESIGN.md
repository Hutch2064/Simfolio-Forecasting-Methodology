# Fixed Huber location for forecast drift — isolated contrast

Parent is the verified full722 shrunk-mean / Gaussian-uncertainty pool. Replace
only the sample location used by its smooth drift regularizer with the Huber
location. The original sample mean remains the centering value of the volatility
fit and raw bootstrap, compensated by the mean offset just as in the parent.
Keep batch Gaussian mean dispersion, both volatility/resampling laws,240paths,
weights, native seeds/accounting, panel and both scoring profiles fixed.

## Rationale, source and estimand
Large observations can disproportionately move a finite-history drift estimate.
This is not an assertion that actual crashes are data errors or should be erased.
Raw and empirical filtered return observations remain in the resampling pools;
no input is removed, winsorized, relabelled or replaced. The bounded score affects
only the location estimator. Prior training diagnostics found heavy-tailed return
and filtered-residual distributions; this motivates testing resistance of drift
estimation, not assuming better forecasts.

Primary sources:
Huber(1964), Robust Estimation of a Location Parameter, Annals of Mathematical
Statistics35:73–101, https://doi.org/10.1214/aoms/1177703732
Official HuberT definition/default:
https://www.statsmodels.org/v0.14.4/generated/statsmodels.robust.norms.HuberT.html
Use the fixed conventional c=1.345, not a grid or an outer-score-selected value.
For a Normal location model its known-scale efficiency formula
P(|Z|<=c)^2/E[clip(Z,-c,c)^2] is approximately.95, checked numerically in a test.
That is NOT a claim of95percent forecasting accuracy or financial dominance.

For asymmetric returns, Huber location is generally not the arithmetic mean of
log returns. The contrast therefore changes the predictive drift functional;
it is NOT an unbiased-mean or fully Bayesian posterior correction. We retain
the existing batch mean uncertainty to isolate the location change, not claim
that its Gaussian dispersion is the exact sampling law of a Huber estimator.
Skewness may make this choice worse, which the fixed forecast gates must decide.

## Fixed estimator and native integration
Solve sum(clip((r-m)/s,-c,c))=0 with s=MAD/Normal_quantile(.75), MAD around the
training median. If MAD is zero, use the finite training population SD; a
constant history returns its constant location. These are explicit statistical
boundaries, not discarded histories or a fitted financial floor. Solve the
monotone score by bracketed root finding with floating-point tolerances. The
no-robustification c=infinity boundary returns the exact sample mean for testing
only; the admitted model always uses1.345.

New drift=center_regularizer(Huber_location,unchanged_batch_SE), with the same
existing t²/(1+t²) functional form. Subtract the original sample mean through
the native mean-offset machinery; do not alter shocks, fit or accounting.
Fit every statistic only to the exact native training-prefix simple/log round
trip. No future outcomes, score-selected scale/constant or class/horizon route.

Four tests pass: estimator versus independent statsmodels Huber-objective
minimization; affine/sign/constant/zero-MAD boundaries; standard Normal efficiency
and bounded influence; full native current-winner parity without robustification.
Qualify245training prefixes,49families,6classes before freezing245development,
245reserved,81series/4122broad origins, then full722/36813 before promotion.
Advance unchanged only for lower primary than the current winner on identical
coverage. Disclose sensitivity and class/family/horizon regressions. No retuning
after field outcomes, no independent-validation/significance claim. Catalogue
every full attempt even if losing and preserve all rejected evidence remotely
with verified complete recovery before local raw retirement. Goal remains ongoing.
