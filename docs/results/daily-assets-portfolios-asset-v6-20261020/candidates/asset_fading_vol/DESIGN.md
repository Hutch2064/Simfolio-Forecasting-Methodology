# Fading conditional variance on the raw circular component

This is a targeted joint-asset contrast to the verified circular/FHS hybrid. It addresses the user suggestion that near-term conditioning should fade with forecast lead, without selecting a model on a single horizon or modifying any scoring weight.

The incumbent already has a mean-reverting GARCH(1,1) filtered half. Leave that half, joint residual rows, Gaussian joint mean uncertainty, smooth point-mean shrinkage, raw historical row indices, fixed equal component weights and in-path portfolio accounting unchanged. Change only the raw half: replace each historical deviation from its training mean with a scaled deviation. At native forecast lead t starting at zero, the squared scale is (1-rho^t) + rho^t*v0, where v0 is the conditional variance divided by historical variance. It is a positive convex combination returning to historical variance, rather than collapsing toward zero. The per-asset mean and additive mean-uncertainty draw are not scaled.

Two coherent global specifications are frozen before field testing:

* GARCH fade: reuse the incumbent fit, v0 = its next normalized variance and rho = alpha+beta. No extra optimization.
* EWMA fade: estimate one smoothing coefficient per asset from training history alone by Gaussian quasi-likelihood, initialized at historical variance. Compare to the constant-variance member. Use the final EWMA state for v0 and its smoothing coefficient for rho. This is an explicit mean-reverting overlay, not a claim that a pure EWMA/IGARCH expected variance mean-reverts.

No fixed horizon cutoffs, outcome-fitted decay, per-family routes, held-out calibration or score-selected weights are introduced. Shared historical rows preserve joint innovation sampling; the time-dependent asset scales change conditional covariance coherently. Early shocks remain in cumulative wealth, so fading later increments does not guarantee unchanged long-horizon terminal scores.

Qualification checks positive variance and its historical long-run target, per-asset historical mean preservation, exact incumbent recovery for unit scale, bitwise unchanged filtered half, EWMA recurrence and chunk invariance. Screen both recipes on the same frozen 355 development cases at 100 paths with a shared 12-worker pool. Only the unchanged canonical overall distributional score determines advancement. Any short/medium/long horizon or family breakdown is diagnostic with zero additional score weight. Credible improvements require 1,000-path matched development, reserved and broad checks before full-panel admission; neither a short-horizon win nor this low-path screen establishes a new official best.
