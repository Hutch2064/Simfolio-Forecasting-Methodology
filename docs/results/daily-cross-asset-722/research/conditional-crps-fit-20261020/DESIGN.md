# Score-directed conditional volatility fitting

Parent is the verified full722 shrunk-mean / Gaussian uncertainty winner.
Only its FHS component's training estimation objective changes: minimize
conditional Gaussian CRPS rather than negative Gaussian log likelihood. Both
fit the same two variance-targeted stationary GARCH parameters with the same
two fixed starts, numerical stationarity margin, convergence checks and explicit
constant-variance member. No new forecast parameters or score-fitted multipliers.

Primary source: Gneiting and Raftery (2007), Strictly Proper Scoring Rules,
Prediction, and Estimation, JASA102:359–378, Sections1,4 and9.
https://doi.org/10.1198/016214506000001437
https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf
The paper motivates optimum-score estimation in general. This particular
variance-targeted financial recursion is our contrast, not a claim established
by the paper. We minimize positive CRPS; the paper uses a reward orientation.

Motivation: the frozen forecast ranking uses CRPS, whereas the current FHS
parameters optimize a likelihood sensitive to large standardized squared errors.
The earlier Student-likelihood fit failed its development gate; that tests a
different innovation likelihood, not direct CRPS M-estimation. Existing training
diagnostics found heavy-tailed filtered residuals. A different proper fitting
objective may alter volatility persistence without extra complexity. It must
still demonstrate an actual forecast benefit; improved training scores alone
are not a gate, and the Gaussian one-day objective is only a surrogate for the
final multihorizon empirical-innovation forecast distribution.

For standardized centered training return y and normalized conditional variance
v, C= y*erf(y/sqrt(2v)) + sqrt(v)*(2*phi(y/sqrt(v))-1/sqrt(pi)).
dC/dv=(phi(y/sqrt(v))-1/(2sqrt(pi)))/sqrt(v). Propagate the identical GARCH
variance derivatives and transform through the existing softmax parameters.
Tests compare CRPS to an independent CDF integral, its gradient to numerical
differences, the variance recursion bit-for-bit, fit equivariance, and the
current-likelihood boundary to the entire unchanged native forecast.

Training fitting uses only the origin prefix. The sample mean, batch SE,
shrinkage, Gaussian uncertainty draws, empirical filtered innovation pool
normalization, WR component, 120+120 paths, component weights, seed namespaces,
recursive variance law, clipping/accounting and both fixed panel v3 scores are
unchanged. Forecast innovations remain empirical, NOT Gaussian. No outer
evaluation scores enter a fit, seed, route or parameter choice.

Freeze245development,245reserved, then81series/4122representative native origins.
Advance unchanged only for lower primary than the current winner at every stage;
disclose sensitivity and class/family/horizon regressions. Full722/36813 attempts
are required before any promotion, and every full attempt enters the permanent
catalogue even if it loses. These reused research cases are not an independent
holdout or significance test. Do not retune this objective after field results.
