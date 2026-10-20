# Stationary GARCH coefficient-posterior predictive contrast

Current parent is the verified shrunk-mean / Gaussian uncertainty full722 winner.
Change only its FHS half: integrate volatility coefficient estimation uncertainty
instead of using one fitted alpha,beta. Keep the WR half,120+120paths, mean
shrinkage/batch Gaussian mean uncertainty, native shock indices, component seed
namespaces, weights, accounting, fixed inputs and both v3 profiles unchanged.

## Methodological grounding and limits
Primary references:
https://journal.r-project.org/articles/RJ-2010-014/
https://journal.r-project.org/articles/RJ-2010-014/RJ-2010-014.pdf
https://tinbergen.nl/discussion-paper/5183/10-046-4-efficient-bayesian-estimation-and-combination-of-garch-type-models
Posterior GARCH inference motivates integrating estimation uncertainty. The
published2010bayesGARCH Student innovations, priors and MCMC algorithm are NOT
ours; that paper imposes positivity but does not impose stationarity. No cited
result establishes superiority under our financial CRPS or validates this exact
prior. The previous Student fit, CRPS fit, AR1 and MA1 failures did not test this
uncertainty mechanism. No claim of universal financial or Bayesian optimality.

## Fixed working posterior and predictive mixture
Use the current conditional Gaussian likelihood for historical centered log
returns divided by training population SD, target variance1 and initial
variance1. Prior is uniform in alpha,beta on alpha>=0,beta>=0,alpha+beta<s,
s=1-1e-8, equivalent to Dirichlet(1,1,1) on simplex fractions. The same historical
sample mean and target variance remain fixed, not integrated here. Future
innovations are empirical filtered residuals, not Gaussian. This conditional
working-posterior plus empirical-shock predictive construction is not a fully
Bayesian joint empirical-shock likelihood. Independence of the retained mean
uncertainty draw and coefficient working posterior remains a working assumption.

In softmax coordinates alpha=s exp(x)/(1+exp(x)+exp(y)), beta=s exp(y)/den,
logJac=2log(s)+x+y-3logsumexp(0,x,y). Include that Jacobian; dropping it would
change the prior. Three fixed posterior-mode starts, analytic gradient, positive
definite finite-difference information matrix, no eigenvalue floor or hidden
failed-fit fallback. Scrambled Sobol importance integration uses a bivariate
Student proposal (df5,SDmultiplier2 relative to local information); those are
integration settings, NOT financial Student innovations or new target priors.
Two independent scrambles use the same target at bounded powers12,14,16.

Before field scores, freeze numerical criteria: each scramble ESS>=400,
max normalized weight<=.025, coefficient mean differences<=.05posteriorSD,
covariance diagonal relative differences<=.05 and nonlinear one-step variance
mean relative differences<=.005. Add nodes only to meet these criteria; otherwise
raise an explicit fitting failure. Do not substitute the point estimate or tune
against outer results. Qualification plans/hash receipts record the rules before
all245prefix checks and before any field forecasts. Original12prefix prototype
remains separately preserved. Native-seed and exact native simple/log-roundtrip
checks address deterministic reproducibility, not alternative model selection.

Mix the two normalized importance clouds equally. Independently stratify120
posterior CDF draws and randomly permute them under a new fixed coefficient
namespace. For each selected alpha,beta, recompute its conditional residual pool
and final variance from the training prefix; zero-center/unit-normalize the
pool exactly as in the retained FHS. Simulate that path with the unchanged native
GARCH/accounting kernel and original independent mean-offset draw. The target
sample variance is unchanged across coefficient pairs; stationarity is explicit.
Save the120coefficient pairs and compact precision diagnostics, not redundant
8192+particle pools in every checkpoint. Ensemble members are dependent under
stratified resampling, as the parent mean nodes already are; frozen empirical
CRPS remains unchanged. No class, horizon, seed or score-fitted parameter routes.

## Verification and frozen forecast gates
Nine tests: posterior target/Jacobian/gradient, prior analytic moments,
independent simplex quadrature, exact native pool/variance reconstruction,
one-step variance vs native filter, explicit precision rejection, deterministic
stationary affine-invariant draws, complete collapsed-posterior native forecast
parity, and no point fallback after all node budgets fail. All245training
prefixes span49families/6classes. These are numerical checks, not accuracy wins.

Only after qualified numerical/native training checks, freeze245development,
245reserved,81series/4122broad origins, then full722/36813 before promotion.
Compare fresh Public/current/candidate on identical native dates, origins,
horizons and240paths. Advance unchanged only for lower primary than the current
winner at every stage; report sensitivity and all class/family/horizon losses.
No retuning after field outcomes. These are adaptively reused research cases,
not independent validation/significance. Every full attempt enters the permanent
catalogue even if it loses; all rejected evidence remains recoverable before
local raw files are retired. The wider goal remains unfinished.
