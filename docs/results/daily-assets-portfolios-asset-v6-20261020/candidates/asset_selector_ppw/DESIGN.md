# Corrected PPW component prototypes — no panel admission

The current166-option adaptive menu contains uniform blocks with and without
replacement, IID joint rows, its prior fitted parent and162filtered choices.
It has no standalone data-driven block-length option. An older fixed-half
PPW/FHS hybrid was tested in a different model context. This prototype instead
offers distinct dependence laws to history-only learned pooling, with no fixed
component share or automatic use when unsupported. Neither baseline changes.

Two candidate components share genuine joint asset rows and the existing
within-path holdings kernel. Stationary blocks use geometrically distributed
lengths with a corrected estimated mean; circular blocks use the nearest integer
to the corrected circular estimate. Both sample starts with replacement. This
is not a1to5year fixed range or the uniform1tohistory/horizon baseline.

The estimator is the pinned arch implementation of Politis–White2004 with the
Patton–Politis–White2009 correction. It estimates variance-optimal bootstrap
lengths, not lengths guaranteed to maximize forecast accuracy. Its documented
lag-window and block-length rules are part of that estimator, not new manually
chosen accuracy caps. The prototype uses the largest corrected estimate across
active-asset centered returns and centered squares. That multivariate extension
is an explicit heuristic to retain mean/volatility dependence, not a theorem
about forecast optimality. Internal rolling CRPS must decide whether it helps.

Zero-weight constituents do not set the length. Nonconstant histories too short
for the library's own lag window leave this option unsupported, with an explicit
reason; they do not trigger a replacement forecasting model. For a constant
joint historical return law, any block length generates exactly the same point
law, so length1is its identified representative. A stale prepared fit is rejected
by the full historical matrix and portfolio-weight signatures.

Sources: [official estimator documentation](https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.optimal_block_length.html),
[authors'2009correction](https://public.econ.duke.edu/~ap172/Patton_Politis_White_2009.pdf).
No asymptotic bootstrap theorem is a financial forecasting improvement claim.

The integrated168-option prototype passes eight component/integration tests,
including exact output preservation for all166original components on a joint
fixture, causal prefix/date isolation, full joint-path pooling and explicit
unsupported-history reasons. A matched financial menu-expansion screen remains
required. No financial score, promotion, speedup or
full-panel catalogue increment is claimed.
