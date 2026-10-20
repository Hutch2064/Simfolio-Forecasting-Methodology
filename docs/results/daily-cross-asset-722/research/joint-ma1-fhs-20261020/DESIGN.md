# Conditional MA1-GARCH innovation dynamics

Current parent is the verified full722 shrunk-mean / Gaussian-uncertainty pool.
Replace only the FHS component with a jointly fitted conditional MA1-GARCH law;
the WR component,120+120paths, mean shrinkage/uncertainty, innovation indices,
namespace seeds, component weights and panel/scoring remain fixed.

Training-only signed-dependence diagnostics and ma1_qualification provide the
motivation. Raw equity lag1 is positive but median lag2 negative, unlike the
positive lag2 implied by AR1; MA1 zero-lag2 is closer for68percent of prefixes.
This is descriptive, not identification or proof of forecast improvement. The
previous AR1 add-on and filtered-residual-block failures do not establish that
this coherent joint finite-memory model cannot help.

Primary technical references:
https://tsmodels.r-universe.dev/tsgarch/doc/arma_garch.html
https://arch.readthedocs.io/en/latest/univariate/forecasting.html
https://www.statsmodels.org/stable/generated/statsmodels.tsa.innovations.arma_innovations.arma_innovations.html
These document ARMA innovation recursions, joint conditional-volatility models
and empirical standardized-shock forecasting. Our fixed variance-targeted
implementation is a research contrast, not a claimed theorem from those sources.

## Explicit conditional law
With training-only sample mean mu fixed, e_t=(r_t-mu)-theta*e_(t-1), initialize
e_(-1)=0. Target innovation variance T=mean(e_t²), V0=T, and
V_t=T*(1-alpha-beta)+alpha*e_(t-1)²+beta*V_(t-1). Minimize conditional Gaussian
quasi-likelihood .5mean(log(V_t)+e_t²/V_t) jointly for theta,alpha,beta. This is
an explicit conditional initialization, NOT exact stationary joint likelihood.
The earlier exact stationary Gaussian MA1 fit supplies only a deterministic
training-derived optimizer start; its latent Gaussian posterior is NOT used.

Compute de_t/dtheta=-e_(t-1)-theta*de_(t-1)/dtheta and dT=mean(2e*de).
The conditional-variance theta derivative includes both dT and lagged-innovation
terms. Analytic gradients are checked independently. Invertibility/stationarity
margins1e-8 are numerical guards, not fitted financial multipliers. Three fixed
starts: the two retained interior GARCH starts at theta0, and the retained
converged GARCH coefficients with the training MA1 seed (or the first interior
start if the retained constant-volatility boundary was selected). No coefficient,
order or prior-width grid. At least one converged eligible joint fit is required;
failed joint fits are not silently replaced by the old model. The successfully
fitted theta0 retained law is an explicit nested boundary; it can be chosen only
after that convergence check and if its training likelihood is no worse.

Center and unit-normalize the empirical conditional innovations. Simulate
future e_t via the same GARCH recursion/empirical shocks, then
r_t=mu+global_mean_offset+e_t+theta*e_(t-1), starting at the actual last
conditional training innovation. That innovation is known under the specified
zero-initialized conditional model; do not add an unrelated Gaussian latent
posterior. MA/GARCH coefficient uncertainty is plug-in, as retained GARCH
coefficients are; the original batch mean uncertainty remains integrated.
No fully Bayesian or exact joint posterior claim.

Use the native simple-return clipping and cumulative-wealth log differences.
At theta0 the exact retained fit and native parameter_paths function are used;
a separate new-kernel test verifies its theta0 accounting bit-for-bit too.
Tests also cover analytic gradients, independent Python conditional likelihood,
fit equivariance/synthetic dependence, and exact finite-memory cumulative
moments from exhaustive discrete shock paths.

## Frozen gate
Qualify all245training prefixes,49families,6classes without reading field
scores, then freeze245development,245reserved and81series/4122broad origins.
Compare fresh Public, current winner and candidate on identical native dates,
origins, eligible horizons and240paths under unchanged v3 scoring. Advance
unchanged only when primary is lower than current winner at every stage;
disclose both profiles and class/family/horizon regressions. No outer-score
fit, retuning, seed selection or class/horizon routes. Evidence is adaptively
reused, not independent validation or significance. Full722/36813 attempt
is required before promotion and every full attempt enters the catalogue,
including a loser. Rejected pilots stay preserved with complete raw evidence.
