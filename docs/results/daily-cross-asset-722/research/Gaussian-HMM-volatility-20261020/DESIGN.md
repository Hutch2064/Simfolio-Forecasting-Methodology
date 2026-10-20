# Frozen contrast to qualify: persistent Gaussian volatility regimes

Replace only the current winner's120-path GARCH/FHS half with a two-state Gaussian
volatility HMM. Keep the exact120 raw random-block paths, current smooth historical
mean regularization, batch Gaussian mean uncertainty, fixed240 total paths and
half whole-path distribution weights. HMM emissions are Gaussian rather than the
parent's empirical filtered shocks. This is a distinct distribution/volatility
law, not a claim that innovation resampling is unchanged. It is a portfolio-series
model, with no family/horizon routing or score-trained combination.

Each native training prefix is centered at its sample mean and normalized by its
population standard deviation. The historical initial state distribution is fixed
uniform. State means are zero; state variances and transition probabilities are
estimated. Future states start from the final filtered state probabilities, then
transition before the first future emission, once per actual native observation.
Independent state-path and Normal-shock seed namespaces are fixed in source.
The retained batch mean uncertainty is an independent working offset, not a full
joint posterior over the HMM's mean, parameters or historical state sequence.

In normalized units use independent IG(shape2,scale3) variance priors and independent
Dirichlet(2,2) transition rows. IG's variance mode is1; the MAP variance M-step is
(weighted squares+6)/(expected occupancy+6). Transition M-step is(count+1)/(row
count+2). These are transparent empirical-prior choices made before field scores,
not a literature-recommended universal prior or an out-of-sample optimized grid.
They prevent zero variance and absorbing states. Sorting by variance only resolves
label symmetry. A truly constant history has a disclosed deterministic-emission
boundary, not a failed-fit fallback.

Two fixed numerical starts: variances(.5,1.5) with transition diagonal.95 and
(.25,3) with diagonal.8. Twenty monotone MAP EM iterations warm each start, followed
by analytic-gradient observed-posterior L-BFGS-B(max1000,ftol1e-14,gtol1e-9,maxls40).
Require optimizer success, finite posterior, per-observation score norm≤1e-6,
MAP EM fixed-point error≤1e-4 and no posterior decrease from the EM warm start.
Choose the highest posterior among valid starts; record every start's status.
Reject the fit if neither converges; no substitution of GARCH or failed fit.
The optimizer's starts and convergence controls are numerical choices, not models
selected by held-out performance. Gaussian-mixture likelihood has local optima;
two starts do not prove the global posterior optimum.

Native daily simple clipping and accumulated-wealth/log-difference accounting use
the canonical implementation's constants and sequence. No new clipping/floor
strategy. Independent golden calculations test this, including equal-variance
Normal and extreme-return boundaries. Exhaustive state enumeration checks the
forward/backward likelihood, smoothed states and expected transition counts;
independent objective optimization, finite-difference gradients and MAP EM ascent
check fitting. The inherited pool's no-substitution boundary replays exactly, but
that is not financial HMM/FHS parity.

After qualification on all245 exact native training prefixes, freeze sources and
the existing245 development/245 reserved/4122 broad cases. Advance only on lower
primary score than the unchanged current winner, disclose both profiles and all
regressions. No state-count/prior/persistence/seed retuning from outer scores.
Fresh full722/36813 required for promotion. Reused benchmark subsets are adaptive
development, not independent validation or significance evidence.

Primary research motivation: Rydén,Teräsvirta,Åsbrink, Stylized Facts of Daily Return
Series and the Hidden Markov Model(1996WP117/1998JAE13:217–244),
https://swopec.hhs.se/hastef/abs/hastef0117.htm . Zero-mean Normal mixtures with
Markov switching can reproduce temporal/distributional return features; reported
parameter instability motivates testing rather than assuming forecast gains.
The priors and fitting rules above are ours, not attributed to that paper.
