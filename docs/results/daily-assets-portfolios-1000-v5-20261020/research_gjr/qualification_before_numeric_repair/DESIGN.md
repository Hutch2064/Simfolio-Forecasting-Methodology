# Sign-asymmetric FHS volatility, frozen contrast v1

The current best uses a symmetric GARCH(1,1) volatility recurrence in its FHS
half. Positive and negative equal-size shocks therefore have the same next
variance effect. This is a specific untested mechanism, not a new grid of
decay constants. The raw-block half, smooth mean shrinkage, Gaussian batch mean
uncertainty, component seeds,50/50 weights, exact1,000 paths, data, main score,
clipping and native accounting remain unchanged.

Academic motivation: Glosten, Jagannathan and Runkle (1993),
https://doi.org/10.1111/j.1540-6261.1993.tb05128.x, model asymmetric volatility
responses to positive and negative innovations. Hansen and Lunde (2005),
https://doi.org/10.1002/jae.800 (author paper copy:
https://faculty.washington.edu/ezivot/econ589/DoesAnythingBeatGarch11.pdf), find
leverage specifications useful for IBM volatility but not universally superior
for exchange rates. Their variance-forecast results are motivation, not evidence
of improved terminal-distribution CRPS on this panel. Do not inherit their
statistical-significance claims or evaluate this candidate on volatility losses.

Use centered training returns y normalized to population variance one. Fit
positive and negative squared-shock coefficients ap,am and persistence b by
Gaussian conditional QMLE. Training recurrence is
v[t]=omega_train+ap*y[t-1]^2*I(y>=0)+am*y[t-1]^2*I(y<0)+b*v[t-1], v[0]=1.
The observed signed squared-return shares kp,km sum to one, and
omega_train=1-b-ap*kp-am*km. Fixed logistic coordinates ensure nonnegative
coefficients and b+max(ap,am)<1 with the parent's1e-8 numerical margin. This
sufficient stationarity domain is stronger than a Gaussian half-negative
restriction; it supports any subsequent standardized empirical shock pool.
Three fixed starts, exact analytic gradients and the parent's convergence
requirements are declared before field scores. The exact symmetric parent is
an explicit nested fitted member. All asymmetric starts failing is a fitting
failure, not an undocumented fallback.

Recenter and standardize fitted residuals exactly as the parent does. Their
positive/negative second-moment shares may differ from kp,km. Set the forecast
intercept to1-b-ap*E[z^2 I(z>=0)]-am*E[z^2 I(z<0)]. Thus normalized stationary
variance remains one under the actual empirical innovation law, instead of
silently assuming its negative second moment equals0.5. Start with the fitted
next conditional variance. This moment-target correction is training-only and
has no fitted field multiplier. It is an explicit approximate-QMLE plus empirical
moment forecast construction, not a claim that both intercepts estimate the same
Gaussian likelihood. On the symmetric nested boundary the correction and
recurrence equal the exact parent, including full native path arithmetic.

Use the previously frozen70 economic-family representatives and native
development/reserved origins, with deterministic asset support for portfolio
quarters. This roster predates this hypothesis. Those cases are adaptively reused
research evidence, not an independent holdout. Qualify710 historical prefixes
without reading field scores. Any new development forecasts wait for available
shared slots; do not run a second12-worker forecast queue over the current12.
Then evaluate355 matched development tasks with unchanged1,000-path incumbent
and Public references. Only strictly lower complete sole-main Q with no failures
or score-integrity errors advances to355 reserved tasks. Broad coverage and full
panel are required before promotion. Record all family/horizon regressions and
diagnostics without adding penalties or selecting from them. A failed contrast
does not close the wider goal or rule out all asymmetry models; no cutoff/seed/
coefficient grid is admitted by this design.
