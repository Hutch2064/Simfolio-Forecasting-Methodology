# Historical drift regularization with unchanged predictive uncertainty

The retained winner uses the entire training sample's mean log return as its
forecast center, plus Gaussian uncertainty whose variance comes from chronological
training batches. Estimation noise can affect the center as well as dispersion.
The two previous contrasts changed volatility fitting and innovation dependence;
neither regularized the winner's expected return. Neither improved development.

The frozen Website Default source at
`resources/french1926/control_sources/smart_iid_forecast.py`, lines877–884,
uses the smooth weight `t_squared/(1+t_squared)`, despite a legacy description
that calls the kernel positive-part shrinkage. This experiment uses that smooth
functional form on both components of the current winner, with the existing
batch-matched Gaussian mean standard error rather than the Default IID standard
error. The Default's other IID dynamics and its historical score are not inherited.

Let mu be the training sample mean and s the retained Gaussian mean-draw standard
deviation. Set forecast center `mu * mu^2/(mu^2+s^2)`. Preserve exactly the existing
Gaussian draw variance s^2. Implement the changed center by adding a constant
`center-mu` to those same draws before the original native accounting kernels.
Keep both volatility fits, raw/filtered shock pools, resampling indices, component
namespaces, 120/120 whole-path weights and240paths unchanged. The zero-noise limit
and imposed no-shrinkage boundary equal the original model. There is no fitted
prior width, OOS score input, class route, horizon threshold or seed search.

General support for addressing expected-return estimation risk comes from the
Bayesian/shrinkage portfolio literature, for example the empirical Bayes study
https://doi.org/10.1017/S0022109000012187. That does not prove this univariate
heuristic optimal: this is not a James–Stein dominance theorem or a calibrated
Bayesian posterior. In particular, deliberately retaining the parent's predictive
variance isolates center regularization; it does not apply a formal posterior
variance contraction. The exact rule is motivated by an existing implementation,
and its forecasting merit must be established on the fixed panel.

Five focused tests cover known values, sign/scale equivariance and noise
monotonicity, invalid inputs, unchanged Gaussian parameter dispersion, and exact
full native loss replay when no shrinkage is imposed. Training-only preflight
quantifies the size of regularization and historical half-mean stability over all
245development prefixes before reading any new field scores.

The frozen stage sequence is245development,245reserved, then81series/4122origins.
Freshly replay Public Adaptive and the unchanged current winner on every identical
case. All490 development reference loss hashes must match the previously retained
summaries. Advance only after complete, finite, failure-free lower primary scores
than the winner; retain both profiles and class/family/horizon regressions.
These reused cases are adaptive research evidence, not an independent holdout.
Full722 evidence and catalogue registration are required before promotion.

One nice15 field worker with single-threaded numerical libraries;1GiB owned pilot
cap under the12GiB research budget and100GiB free reserve. Preserve every unique
source/result losslessly in repository-linked private archives, fresh-download,
disk-restore and SHA256 verify before local retirement. No production changes.
Failure of this exact regularizer does not settle every mean-estimation model or
complete the wider accuracy-maximization goal.
