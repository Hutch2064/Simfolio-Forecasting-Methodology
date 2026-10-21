# Request-local sequential CDF-mixture learning (research only)

This experiment keeps `asset_selector_endpoint_flat`'s 167 active recipes, joint
asset fits, completed dated `half_doubling` folds, representation counts, CRPS
endpoint scaling by sqrt(h), deterministic inner seeds, antithetic path allocation,
and final joint-asset/portfolio path kernels. It changes the request-local mixture
learner only. It does not use any outer loss, prior request's weights, catalogue
ranking, or panel OOS observations. It is not production wiring or an OOS win.

`MenuPool` accepts `menu=batch|boa|progressive`, `inner_paths`, and `schedule`.
`prepare` computes all three from the same endpoints, fits selected final components
on the full current prefix, and returns the same dated-request guards as the parent.
`batch` uses the exact parent certified solver, accumulation order and coefficients;
the original `flat` mode maps to `batch`. No solver replacement or new recipe cap.
The end-point generation loop is intentionally copied rather than monkeypatched:
this is isolated from other workers; the real dated contract test checks exact
parent A/B, weights, traces, allocations and final paths.

## Primary literature and the adaptation

Sources read before implementation:

- Berrisch and Ziel, *CRPS Learning*, [equations 33a-f](https://arxiv.org/html/2102.00968#S4.E33).
- Wintenberger, *Optimal learning with Bernstein Online Aggregation*,
  [Figure 2 and equation 2](https://arxiv.org/html/1404.1356#S1).

The CRPS-learning paper combines quantiles with potentially quantile-dependent
weights. Here we combine empirical CDFs with one simplex for the complete request.
The existing exact empirical CRPS is L_t(w)=w'A_t - .5 w'B_t w, whose gradient
is A_t-B_t w. This is a different pooling family: no pointwise quantile guarantee
or claimed optimal rate is transferred to this implementation.

`learning.learn` follows the original Figure 2's explicit **expert-minus-mixture**
linearized excess sign, previous-round rate in the correction, accumulated squared
excess and prior-times-rate exponential weights. This avoids mechanically using
Equation 33's mixture-minus-expert regret with its negative softmax exponent, which
would move away from an analytically better expert under that literal sign reading.
One analytic first-step test verifies the intended sign and exact update.

For a fold with representation count m, the loss is m*L_t, so its linearized excess
is m*(g_k-w'g). Squaring this weighted excess gives m^2, not m, in the variance.
This is ONE weighted representative round. It is not a claim that the sampled
representative has supplied m independent observations or an exact replay of the
unrepresented origins. Chronology remains the chronological represented schedule.

The one intentional Figure 2 range adaptation: initialize a common range unit at
the first nonzero maximum excess, then use E_k=2*unit*2^ceil(log2(max(1,max_s|excess_k|/unit))).
This is a conservative dyadic bound, permutation and positive-scale equivariant,
with finite rates for an expert whose excess is exactly zero. It replaces the
paper's fixed absolute `2^-c` floor (which is undefined in return-scale units
without choosing c). It is data-dependent initial calibration, so the original
proof is not asserted for this adaptation. A wholly zero-excess prefix leaves the
uniform prior unchanged; a singleton/identical-expert universe is mathematically
indifferent. Neither case is a fixed-model winner fallback.

The progressive alternative is the arithmetic mean of w_0,...,w_T, including the
prior and final update, matching the original progressive online-to-batch idea.
These are averages of weights applied to current-prefix component distributions,
not averages of historical forecasts at different dates. No iid risk interpretation
is asserted for dependent returns or information-spaced representatives.

## Compact replay and scope

Each `prepare` retains per-fold float64 A and float64 upper-triangle B (including
diagonal), valid IDs, counts, stops and uint32 seeds; no historical path cubes are
retained. `save_fold_coefficients` optionally writes lossless compressed NPZ with
JSON dated provenance, portfolio weights and bank specification hash.
`load_fold_coefficients` rejects any differing history, training dates, portfolio,
horizon, inner paths, replication, schedule or bank. `unpack_B` reconstructs B
exactly, and `learn` can replay the request's dated coefficients. No hidden cache.

If validity changes over the chronological folds, `prepare` rejects the experiment:
retrospectively deleting experts that fail on a later prefix would contaminate a
sequential-prefix comparison. Stable first-fold invalidity is inherited equally by
the batch and sequential modes. Final-prefix selected fit failures fail closed.

## Numerical limits and next admission

No total iteration, path, origin or candidate cap is introduced. Float64 overflow,
underflow, invalid arithmetic and nonfinite state fail explicitly; weights are not
clipped into an accepted fallback. Exponential underflow also fails rather than
silently eliminating an expert. The current version is not range-normalized, so
extreme coefficient scales (tested at 1e-300 and 1e300) correctly raise; positive
scale equivalence is tested at 1e-8 and 1e8. A broader numerical audit may motivate
algebraic state rescaling before admission to especially extreme inputs.

Recommended next step: a parent-approved matched admission on six already frozen
request contexts spanning short/long horizons and portfolio dependence, inner1000,
all existing recipes and identical dated prefixes/seeds, no selection on results.
Persist one compact coefficient cache per request, run batch/BOA/progressive from
that cache, and independently audit pre-update prefix measurability and CRPS.
Require bitwise parent batch coefficient/weight/path agreement before interpreting
any sequential score. Measure coefficient formation, learning-only, component
refit, path generation, and total time separately. Shared preparation refits the
union of all mode supports, often all recipes for BOA; this is not evidence of an
end-to-end speedup. Outer scores compare candidates only; they never train weights.
No panel sweep was launched by this work.
