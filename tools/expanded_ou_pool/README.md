# Equal complete-path OU and Bayesian EWMA pool

This research candidate is `equal_path_pool_stationary_ou_and_bayesian_ewma_sbb`.
It is a new model with a fixed 0.5/0.5 linear opinion pool. The panel's total
240 simulations comprise 120 complete paths from each unchanged component:

- `asset_map_stationary_ar1_ou_residual_volatility`, selected by its exact ID
  from `tools/endpoint_residual_volatility/models.py`;
- `bayesian_sbb_overlay_ewma_absolute`, built by the existing model registry.

The adapter changes only each component's context model ID and simulation count.
It preserves the task seed, origin, dates, training data and portfolio policy.
Each component performs its own full forecast and portfolio rejoin before the
adapter concatenates the complete paths. It never averages daily returns or
parameters, switches components within a path, or fits mixture weights to test
observations. Even simulation counts are required; the official count is 240.

The covariance rationale is the exact stationary OU residual component's single
state, without the fractional residual covariance fit used by M261. This is a
candidate for measuring the score/runtime tradeoff, with no score or speed claim
until matched evidence exists. Existing component laws and fit budgets are
unchanged. In particular, this is not a parity optimization of the M261 pool.

The probabilistic combination follows the linear opinion pool discussed by
Gneiting and Ranjan (2013), *Combining Predictive Distributions*,
https://arxiv.org/abs/1106.1638 . Component path generation retains the original
dependence, volatility, parameter and sampling assumptions.

Run `tests/test_expanded_ou_pool.py` for the adapter contract. A five-task smoke
and subsequent official full blocks must use immutable source and the existing
expanded-panel evaluator, frozen data, 240 simulations and unchanged scoring.
Smoke subsets cannot establish a full-block score or an OOS winner.
