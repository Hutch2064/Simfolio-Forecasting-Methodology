# Volatility-coefficient uncertainty: training-only prototype

The parent FHS component currently uses one fitted alpha,beta pair. This new
assessment integrates uncertainty in those coefficients instead. It is a
different mechanism from MA1 dynamics, Student likelihood fitting, CRPS fitting
or mean shrinkage. It is NOT yet a forecast candidate, and no field results or
scores were read. Current mean, batch mean uncertainty and fixed sample target
variance remain proposed controls, not new parameters to tune.

Primary methodological references:
https://journal.r-project.org/articles/RJ-2010-014/
https://journal.r-project.org/articles/RJ-2010-014/RJ-2010-014.pdf
https://tinbergen.nl/discussion-paper/5183/10-046-4-efficient-bayesian-estimation-and-combination-of-garch-type-models
Ardia/Hoogerheide study posterior GARCH inference; their specific Student
innovation, priors and MCMC algorithm are NOT ours. Their2010bayesGARCH paper
explicitly imposes positivity but not stationarity. We impose the stationary
simplex for long forecasts and must state that distinct modeling choice.
No cited result establishes that this model will improve our panel's CRPS.

## Defined working coefficient posterior
Keep the same historical centered returns divided by their sample population
SD, variance target1, initial variance1 and conditional Gaussian likelihood as
the retained FHS. Prior is flat on alpha>=0,beta>=0,alpha+beta<s, where
s=1-1e-8 is the retained numerical stationarity margin. Equivalently the three
simplex fractions have Dirichlet(1,1,1) prior. The sample mean and target
variance are fixed; their uncertainty is NOT integrated here. Gaussian
likelihood plus eventual empirical residual shocks is working/quasi Bayesian
inference, not a fully Bayesian empirical-shock likelihood.

In logistic coordinates alpha=s*exp(x)/(1+exp(x)+exp(y)),
beta=s*exp(y)/(1+exp(x)+exp(y)), logJac=2log(s)+x+y-3logsumexp(0,x,y).
Target density is Gaussian conditional likelihood times this Jacobian; omitting
it would change the prior. Fit the coordinate posterior mode using three fixed
starts and analytic gradient, require convergence, and compute the local
positive-definite Hessian without imposing eigenvalue floors.

Use scrambled Sobol importance quadrature with a bivariate Student proposal,
df5 and standard-deviation multiplier2 relative to the mode information matrix.
These are numerical integration proposal settings, NOT financial Student
innovations or a different posterior target. Normalize exact target/proposal
weights. They cannot be tuned against outer forecast scores. Two independent
scrambles compare integration moments; all numerical errors must be assessed
before field admission. No point-fit fallback may hide inadequate precision.

## Verified prototype scope
Three tests pass: target/Jacobian and mode gradient; prior-only recovery of
analytic uniform-simplex moments; posterior moments against independent
Gauss-Legendre simplex integration on a small fixed synthetic history.
Twelve predeclared short/long training prefixes (one representative family per
asset class, two origins) were used for numerical precision only. Each used
two4096-node scrambles. Minimum importance ESS1712.991; largest difference
between replicated coefficient means2.3883e-5. These are integration checks,
not forecast improvements. All exact prefix IDs/hashes/results are retained
in summary.json. Histories overlap, and no significance claim is made.

## Required next work before any field test
Qualify all245fixed training prefixes and next-variance/covariance precision,
not only coefficient means. Freeze numerical accuracy tolerances before fields;
increase quadrature nodes only to meet those tolerances, with a bounded maximum
and explicit failure if unmet. Potential powers12,14,16 and two scrambles are
numerical precision levels, not alternative predictive models or a score grid.
A meaningful check should compare posterior coefficient means in posterior-SD
units, covariance diagonal accuracy, importance ESS/max weight and the nonlinear
one-step conditional variance. These thresholds remain to be frozen; do NOT
claim the current12-prefix result establishes all-prefix readiness.

For a predictive adapter, retain the exact two-component240path/120each native
pool, current mean law and shock indices. Sample one coefficient pair per FHS
path from the normalized quadrature posterior under an independent fixed
namespace, then recompute that pair's conditional residual pool and final
conditional variance from training history only. Generate the future path using
the unchanged GARCH/native-accounting kernel. Use independent stratified
posterior resampling/permutation to avoid coupling parameter draws to mean
nodes or shock ranks. Record finite-ensemble dependence and posterior numeric
approximation. A collapsed coefficient-posterior boundary must reproduce the
entire current forecast exactly; no numerical-only moment claim substitutes
for native loss parity. Retain zero-mean/unit-variance filtered innovations.
No extra stochastic mean drift, family/horizon routing or score-trained prior.

Only after qualification, freeze245development/245reserved/4122broad origins,
including fresh exact Public/current references, both profiles and all category
regressions. Full722/36813attempt before promotion, catalogue every full attempt
even if losing, preserve complete source/loss evidence before local retirement.
The goal is ongoing; these numerical checks do not satisfy a stopping criterion.
