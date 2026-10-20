# MA(1) training-only qualification — not yet a forecast candidate

The current full722 baseline remains the shrunk-mean / Gaussian-uncertainty
pool. No forecast or ranking change is authorized by these diagnostic results.

Previous training-only short-memory diagnostics found a positive signed lag-one
effect in many equity, bond and FX prefixes, with much weaker higher lags.
For equities the observed median raw lag2 was negative, unlike the positive
lag2 implied by the prior rejected AR1 model; MA1's zero lag2 was closer for
about68percent of equity prefixes. This is descriptive, not identification.
A finite moving-average mechanism is distinct from that AR1 addition and from
stationary blocks of already filtered shocks. It is worth numerical assessment,
not presumed to improve CRPS or proven by sign stability.

Primary references:
- Hyndman and Athanasopoulos, Forecasting: Principles and Practice3,
  Section9.4, moving-average models:
  https://otexts.com/fpp3/arima.html#moving-average-models
- Official statsmodels innovations implementation:
  https://www.statsmodels.org/stable/generated/statsmodels.tsa.innovations.arma_innovations.arma_innovations.html

For centered y_t=epsilon_t+theta*epsilon_(t-1), stationary Gaussian innovations
have v0=1+theta², e0=y0; v_t=1+theta²-theta²/v_(t-1),
e_t=y_t-theta*e_(t-1)/v_(t-1). Concentrate the innovation variance as
mean(e²/v), using the exact log determinant sum(log(v)). The mean remains the
training sample mean, not jointly fitted. Estimate one invertible theta by
bounded scalar optimization, with theta0 as an explicit constant member.
The1e-8 invertibility margin is numerical. The nonnegative latent conditional
variance endpoint uses its theoretical zero lower bound to remove floating
roundoff at the v=1 limit, not an arbitrary financial variance floor.

Four tests pass: exact-filter comparison with statsmodels; profiled likelihood
comparison with an independent dense Cholesky covariance; scale/translation
invariance and recovery of fixed synthetic positive/negative MA1 signals;
exact IID boundary. All245fixed development prefixes and both halves converge.
No evaluation targets or forecast losses were read. Prefixes overlap and are
not independent experiments. Class summaries and every fitted prefix are in
summary.json; the frozen722inputs remain in their existing verified archives.

Unresolved BEFORE any field admission: the Gaussian linear innovations filter
is only a working estimator when coupled with conditional non-Gaussian
volatility. Its conditional latent boundary variance is not automatically an
exact posterior for GARCH empirical shocks. Define a coherent estimation and
simulation law (including parameter and boundary uncertainty), qualify native
accounting and no-dependence boundary parity, and freeze the unchanged matched
stages before testing. Do not simply label a Gaussian MA1 filter plus empirical
GARCH as an exact joint likelihood. No model has been added to the permanent
full-panel count, no coefficient/order/class-route grid is justified here,
and no outer forecast scores may fit a parameter or choose a seed.
