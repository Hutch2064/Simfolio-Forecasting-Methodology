# M261 mean with GJR filtered history

Candidate: `asset_m261_mean_gjr_garch_student_filtered_history`.

Research adapter only. The full canonical80 plus single52 panel has not been
run for this candidate. No accuracy or speed improvement is established.

## Method

- Preserve the exact M261 deterministic mean: the predecessor's Gaussian
  `student.original_fit` with `existing_sv_rate`, followed by
  `sv_moment_functions.moment_return_curves`.
- Fit each complete asset training prefix using arch's GJR-GARCH(1,1) and
  standardized Student-t likelihood. The historical sample mean is removed;
  fitting uses percent-return units and forecasting uses decimal units.
- Retain signed, chronological standardized residuals. Center each column and
  normalize its empirical second moment to one, without tail clipping.
- Select an expected stationary-bootstrap block length using the actual
  `arch.bootstrap.optimal_block_length` PPW implementation on all aligned
  innovation columns and their squares. Use their maximum selected stationary
  block length, with the mathematical minimum expected length one.
- Sample one common historical row index per simulation/day for every asset;
  continue circularly until a Bernoulli restart. Thus contemporaneous historical
  innovation vectors remain intact, including cross-asset signs.
- Recurse stochastic GJR conditional variance from each fitted terminal state,
  draw the selected joint empirical shock, and add the M261 deterministic mean.
  No conventional Student/Laplace SV or rough volatility fit/forecast runs.
- Retain baseline daily log-return clipping to [-1,1], exact calendar handling,
  portfolio weights, rebalancing and turnover costs through `streamed_paths.rejoin`.

Fit checks require convergence, positive omega, nonnegative positive- and
negative-shock coefficients, nonnegative beta, nu greater than two and strict
`alpha + gamma/2 + beta < 1`. A converged SLSQP fit can cross the coefficient
boundary by floating-point roundoff. Only a correction no larger than its 1e-9
feasibility tolerance is permitted, projected onto that same declared boundary;
raw parameters, used parameters, maximum correction and the resulting refiltered
variance change are recorded. A converged likelihood optimum on the closed
stationarity boundary is approximated under the open GJR constraint by scaling
alpha, gamma and beta together toward `np.nextafter(1.,0.)` persistence. Further
ulp steps are allowed only to make the actual rounded sum strictly less than
one, within the same 1e-9 maximum parameter correction. This is an explicitly
recorded approximate constrained GJR boundary fit, not a separate IGARCH law.
Larger violations, nonconvergence and nonfinite/nonpositive future variances
raise errors. There is no alternative-law fallback.

The Student-likelihood stationarity condition assumes symmetric unit-variance
shocks. The empirical shock distribution can be asymmetric, and its bootstrap
can be serially dependent. Diagnostics therefore also record the empirical
negative-shock second moment and the corresponding *iid* variance persistence;
the Student fit's condition alone is not a proof of unconditional stationarity
for the dependent empirical forecast law.

## Bounded validation

`tests/test_garch_filtered_history.py`: eleven checks passed under CPython 3.12.13,
arch 8, one numerical thread. Includes scalar-recursion oracle, sign asymmetry,
common-row deterministic stationary indices, full-prefix Student fitting, IID
boundary-feasibility projection with variance-difference bound, training-only PPW
inputs, exact mean call-chain and 1/3-asset policy-rejoin simulations. Positive
and negative gamma stationarity-boundary cases, substantive-violation rejection
and retention of the bounded training-fit LRU across path clears are checked.

`failure57_diagnosis.json` records the official single52 failure: GLDSIM,
1987-09-30, 1,960 training rows, converged fit with persistence
1.0000000000000004. The bounded repair/refilter result is recorded separately.

`smoke_results.json` contains actual initialized synthetic forecasts with 504
training rows, 240 paths and 21 future dates. Initialization: 5.91 seconds;
single asset: 0.206 seconds; three assets: 0.114 seconds. These observations have
mixed cache states and concurrent background workloads. They are functional
smokes, not matched performance comparisons. Byte-for-byte M261 mean parity and
the existing-SV timescale receipt passed on every smoke asset.

Run checks from the reference root:

```sh
NUMBA_CACHE_DIR=../cache312 TMPDIR=../native-tmp NUMBA_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
../venv312/bin/python -m pytest -q tests/test_garch_filtered_history.py
```

The standard expanded-panel runner can load `tools/garch_filtered_history/models.py`.
No persistent fitted-model cache is enabled. The numerical kernel has a stable
module import identity so Numba artifacts remain usable across runner/test
adapter module names.

## Source references

- [arch's GJR-GARCH and Student-t modeling documentation](https://arch.readthedocs.io/en/latest/univariate/univariate_volatility_modeling.html)
  specifies the same asymmetric variance recursion and standardized Student law.
- [arch's optimal block-length documentation](https://arch.readthedocs.io/en/latest/bootstrap/generated/arch.bootstrap.optimal_block_length.html)
  describes its PPW algorithm, stationary block lengths and the corrected
  Politis-White / Patton-Politis-White references.
