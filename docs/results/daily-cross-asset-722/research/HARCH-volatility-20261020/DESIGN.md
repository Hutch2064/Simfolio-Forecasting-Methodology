# HARCH(1,5,22) empirical filtered simulation: prospective frozen contrast

Replace only the current winner's120-path GARCH/FHS half with variance-targeted
HARCH/FHS. Keep the exact120 raw random-block paths, smooth mean regularizer,
batch Gaussian mean uncertainty,240total paths, fixed half whole-path weights,
native seeds/indices and daily accounting. Historical shocks are still centered,
population-variance-normalized empirical filtered residuals; their values change
with the new filter. No family/horizon routing, calibration fitting from panel
losses or return outlier removal.

In normalized training units y=(return-sample mean)/population SD, set
v[t]=1-sum(alpha)+sum_j alpha[j] * mean(y[t-1:t-lag[j]]^2), with lags(1,5,22).
Unavailable historical squared innovations are1. Fixed native-observation
windows represent daily/weekly/monthly timescales, not calendar interpolation.
These are the standard HARCH windows in the author's arch documentation:
https://arch.readthedocs.io/en/latest/univariate/generated/arch.univariate.HARCH.html
Its equation is exactly a weighted average of lagged squared residuals. Primary
research motivation: Müller et al.(1997), Volatilities of different time
resolutions—Analyzing the dynamics of market components, J.Empirical Finance4,
213–239, https://doi.org/10.1016/S0927-5398(97)00007-8 . The publisher abstract
describes differing volatility timescales; full text was unavailable. Do not
attribute the fixed variance target or implementation choices below to that paper.

Fit three nonnegative coefficients with sum<1-1e-8 by Gaussian conditional QMLE,
using fixed starts(.05,.15,.70) and(.10,.10,.30). Softmax coordinates enforce
support; analytic gradients are compared with finite differences. L-BFGS-B
ftol1e-12,gtol1e-7,maxiter500. Require optimizer success, finite loss and
gradient norm≤1e-4. Select the lower eligible training loss. A constant-variance
boundary is an explicit member and may be selected only after a fit converges;
if neither fit converges, reject, with no failed-fit substitution. The training
population variance remains the fixed target. This is three parameters versus
the parent's two, with different conditional memory. It is not a claim of exact
long-range dependence or Bayesian parameter integration.

Future normalized squared innovations recursively replace the training history
in each path's22-slot buffer. Compute each lag mean directly from that buffer,
avoiding cancellation from accumulated running sums. Independent empirical shocks
use the same RNG namespace and draw ordering as the parent FHS. Update volatility
with stochastic innovations only, excluding the independent mean-offset draw,
as in the parent. Standardized history tails remain present; no future variance
floor, score-based multiplier, volatility clipping or window search is added.
Native cumulative wealth/log differences and clipping constants are unchanged.

Qualify against the independent arch implementation and explicit Python
recursion, check the ARCH1/constant-variance nested kernels against the original,
and verify fitting/equivariance/failure rejection and pool wrapper identity.
Run all245 exact native training prefixes, retaining descriptive residual-memory
checks without treating overlapping training prefixes as independent evidence.
Then freeze the existing245development,245reserved,4122broad cases and source
hashes before first financial forecast. Advance only on lower primary canonical
score than the current winner; disclose both profiles and subgroup regressions.
No held-out retuning of lag windows, weights, objectives or seeds. Full722/36813
before promotion; every full attempt enters the permanent catalogue even if it
loses. Current benchmark evidence is adaptively reused and not independent
production validation of arbitrary multi-asset portfolios.
