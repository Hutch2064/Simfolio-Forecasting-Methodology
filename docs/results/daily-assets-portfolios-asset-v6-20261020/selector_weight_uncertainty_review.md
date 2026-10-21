# Weight uncertainty review — 2026-10-21

Research decision only: no candidate, sampler, fit, score, active source or
admission changed. The current reference computation continues intact.

## Primary-source scope

Yao, Vehtari, Simpson and Gelman, *Using stacking to average Bayesian predictive
distributions* (2018), https://arxiv.org/html/1704.02030 , sections 2.1, 2.4 and
4.1–4.3 were reviewed. The paper develops predictive-distribution stacking with
proper scores. Its Bayesian-bootstrap modification concerns pseudo-BMA weights;
it is not a demonstration that bootstrapping CRPS-optimal stacking weights
improves forecasts. The authors recommend stacking, with pseudo-BMA+ as a cheaper
alternative in their examples. They also discuss weight regularization.

Those results do not establish a performance gain or a posterior uncertainty
interpretation for our dependent, chronological financial validation blocks.
Our forecasts must continue to use only the history available to each request.

## Consequence for the proposed fold-weight idea

This calculation is our own, not a theorem attributed to that paper. Let each
completed historical fold have empirical mixture CRPS

    L_f(w) = A_f' w - 0.5 w' B_f w,

with simplex-constrained model weights w. If random fold probabilities alpha
have mean 1/F, then for any fixed w,

    E_alpha[sum_f alpha_f L_f(w)] = sum_f L_f(w) / F.

Therefore minimizing this expected loss gives the existing equal-fold
objective. Merely integrating uncertainty in the fold probabilities does not
create a new regularizer. Averaging independently optimized weights,

    E_alpha[argmin_w sum_f alpha_f L_f(w)],

is a different forecasting rule. It may help or hurt and would require its own
accuracy test. Calling it Bayesian does not establish an improvement. Dependent
folds also prevent importing an IID bootstrap confidence interpretation.

## Current decision

Do not add a bootstrapped-weight candidate merely because it sounds more robust.
Prioritize the already prepared controlled internal-path study and the qualified
faster numerical implementations. That study changes internal precision while
keeping final forecasts at 1,000 paths, and reports main-score variation across
initial repeated seeds. Three seeds alone do not certify stability.

Exact equality of learned weights is not required for equivalent forecasts.
Likewise, a sparse fit on past validation predictions does not prove an entire
model family can be removed for every future request. The lazy coefficient
prototype checks the full menu's optimization certificate without an arbitrary
component cap, but its weights and final path allocations can differ from the
dense solver. Saved-coefficient qualification does not measure the cost of
generating requested columns or prove whole-request speed or forecast accuracy.

Reconsider fold-weight averaging if a clearly specified mechanism and additional
evidence justify a representative comparison. Retain per-fold coefficients only
in a separately qualified implementation; current aggregate matrices cannot
reconstruct their individual contributions. Do not alter the running reference
to collect them. No stopping or promotion conclusion follows from this review.
