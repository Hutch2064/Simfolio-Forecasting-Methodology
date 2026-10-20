# Integrated local-scale mean uncertainty

Parent: the current full722 shrunk-mean / Gaussian uncertainty winner. This is
one predeclared mechanism, not a prior-width grid or outcome-fitted route.

The prior working-variance contraction nearly tied but lost the development
gate. It plugged the prior variance into observed mean squared and ignored its
uncertainty; at a zero sample mean it collapsed drift uncertainty. This contrast
instead integrates a local-scale prior. It changes BOTH posterior center and
mean dispersion; it is not the same fixed-center contraction contrast.

Primary methodological source: Carvalho, Polson and Scott (2009), Handling
Sparsity via the Horseshoe, PMLR5:73–80, Section2.
https://proceedings.mlr.press/v5/carvalho09a.html
https://proceedings.mlr.press/v5/carvalho09a/carvalho09a.pdf
Their normal-means construction motivates the prior; their sparse-vector
results do not establish superiority for financial return forecasts.

## Working model and derivation
For training-only sample log-return mean m and the unchanged batch-matched
Gaussian SE s, use m|theta ~ Normal(theta,s²), theta|lambda ~
Normal(0,s² lambda²), lambda ~ half-Cauchy(0,1). The global scale is fixed at
one in SE units. Uncertainty in s is NOT integrated: this is a working
normal-summary likelihood with a data-derived likelihood scale, not a fully
Bayesian analysis of the entire daily financial process.

Let z=m/s, kappa=1/(1+lambda²), w=1-kappa. Prior kappa is Beta(1/2,1/2).
Multiplying its prior by the marginal Normal likelihood gives
p(w|z) proportional to w^(-1/2) exp(z² w/2). For u=sqrt(w),
p(u|z) is proportional to exp(z² u²/2) on [0,1]. Given w,
theta|w,m,s ~ Normal(w*m,w*s²). Draw w from that posterior and retain
its full conditional Normal mixture. At m=0, E[w]=1/3, E[w²]=1/5,
and Var(theta)=s²/3, avoiding the zero-mean variance collapse.

The u CDF is exp(a*(u²-1))*D(sqrt(a)*u)/D(sqrt(a)), where a=z²/2
and D is Dawson's integral. The a=0 limit is Uniform[0,1]. Vector bisection
inverts the CDF; independent adaptive quadrature qualifies it, including all
245 training prefixes. No prior floors, arbitrary financial clamps, fitted
multipliers, score-based seeds or posterior truncation are introduced.

The original Gaussian antithetic nodes and permutation remain. Each +/- pair
shares one independently stratified posterior w under a new fixed namespace;
weights are independently permuted across Gaussian magnitude strata. Ensemble
members are dependent as in the parent. Use the unchanged empirical CRPS.
All conditional Normal components have bounded center and variance for finite
m,s, so this drift law has finite exponential moments; this is not a claim about
unbounded recursive volatility trajectories or financial dominance.

## Frozen evidence gate
Keep unchanged volatility fits, raw/filtered innovation indices, component
namespaces, fixed 120+120 path CDF pooling, native wealth accounting and panel v3
scoring. Fit only the training prefix. Reproduce both Public and current winner
exactly on the same245 development origins,49families,6classes, including all
eligible horizons. Advance only for lower primary score than current winner,
then245 reserved origins, then81series/4122origins, with both profiles and
class/family/horizon regressions disclosed. Evidence is adaptively reused;
these are not independent validation or significance claims. No retuning after
field outcomes. Full722/36813 successful tasks required for promotion. Every
full attempt goes in the repository catalogue, including a loser.
