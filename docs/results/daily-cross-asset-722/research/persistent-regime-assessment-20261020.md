# Next assessment: persistent hidden volatility regimes

Status: researched mechanism, not implemented, frozen or admitted to field tests.
The wider accuracy-maximization goal is unfinished. Do not treat the two latest
pilot losses as an evidence-based global stopping review.

## Why a different family remains credible
The current FHS maps volatility to one continuous GARCH recursion. Its coefficient
posterior did not improve development; fixed Huber location damaged drift accuracy.
Neither tests latent, persistent low/high volatility states. Training diagnostics
already show volatility clustering, and classic financial HMM research supplies a
separate mechanism for temporal distribution shape. This is not proof that it
will beat GARCH, nor a reason to retune any rejected coefficient/mean variant.

Primary sources verified in this research turn:
Rydén,Teräsvirta,Åsbrink(1996working paper/1998JAE), Stylized Facts of Daily Return
Series and the Hidden Markov Model:
https://swopec.hhs.se/hastef/abs/hastef0117.htm
The authors model daily-return distributional/temporal features with a zero-mean
Normal mixture governed by a hidden Markov chain; they also report instability
between subseries. These stylized-fact results are not a predictive CRPS theorem.
Hamilton/Susmel(1994), Autoregressive conditional heteroskedasticity and changes
in regime, Journal of Econometrics64:307–333:
https://doi.org/10.1016/0304-4076(94)90067-1
The publisher result identifies changing regimes as a different volatility
mechanism; full article access was403, so do not attribute an unverified formula
or claim to that text.

## Candidate construction to qualify before fields
Start simple: a two-state Gaussian hidden-Markov volatility component pooled
with the exact retained120WRpaths. Keep current training mean, smooth drift
regularizer/Gaussian mean uncertainty, fixed240total paths and half weights.
Fit only each origin prefix. Training emissions are centered at the unchanged
sample mean; states differ in variance, not fitted trend/risk-premium forecasts.
Future states start from the filtered final training probabilities and evolve
under the fitted transition matrix. Gaussian innovations are part of this NEW
component law, not the parent's empirical FHS law; disclose that change rather
than claiming an unchanged shock distribution. WR still preserves raw tails.
Use independent fixed state/shock namespaces. This would be a coherent fitted
Gaussian HMM, not an empirical-emission model with an incompatible Gaussian
posterior silently reused. Unknown state paths are filtered/integrated; first
assess a regularized parameter point fit, then consider additional complexity
only if diagnostics/results support it. No claim all posterior alternatives are
ruled out by the prior GARCH uncertainty contrast.

An unconstrained free-variance Gaussian mixture/HMM can collapse a regime onto
one observation with arbitrarily small variance. Avoid calling such a fit valid.
One transparent MAP proposal (NOT yet frozen): proper independent inverse-Gamma
variance priors with shape2 and scale3*T, T=training population variance. Their
mode is the pooled T and their contribution makes a state variance update
(sum(gamma*r²)+6T)/(sum(gamma)+6), preventing zero-state variance without an
arbitrary return floor. Symmetric row Dirichlet(2,2) transition priors correspond
to one pseudo-transition per cell at the MAP update, keeping probabilities
interior. These are our empirical-prior modeling choices, not prescriptions from
the cited paper or a score-selected grid. Verify the full log-posterior/M-step
and label-order invariance before adopting them. The pooled sample variance used
in the prior is history-only, an empirical-Bayes specification, not fully Bayes
integration over unknown mean/variance. Do not claim a universal optimal prior.

Use two fixed numerical starts, a stable scaled/log-domain forward/backward
algorithm, monotone MAP EM and explicit convergence/empty-state checks. Order
states by variance to remove label ambiguity; initial historical state prior
can be fixed uniform and disclosed. Define stationary/finite transition support
and native-observation time units explicitly. No calendar fill/interpolation,
outer score-fitted state count, cutoff, persistence prior or class routes.

## Required qualification and evidence
Derive/test forward likelihood, smoothed responsibilities and transition counts
against exhaustive state-path enumeration on short histories. Verify the MAP
M-step increases the stated posterior and has no hidden failed-fit fallback.
Qualify fit equivariance, label swaps and all245fixed training prefixes, including
native simple/log conversion. Test equal-state IID Normal boundaries and native
clipping/cumulative wealth accounting independently. A new financial emission
law cannot claim full parity with empirical FHS just because a pool wrapper is
unchanged. Use exact Public/current reference replays in any actual pilot.

After definition and numerical qualification, freeze245development/245reserved,
81series/4122broad origins and full722/36813before promotion. Both profiles,
classes/families/horizons, failures and typed runtime must be reported. No retuning
from outer losses. Every full attempt enters the permanent unique-model catalogue,
including a loser; every rejected pilot retains complete evidence in the linked
repository/private archives before local output retirement. The model remains
unadmitted until the above financial/numerical choices are resolved and frozen.
