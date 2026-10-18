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

An inadmissible initial fit triggers deterministic constrained retries of the
same likelihood: first summed likelihood with ftol 1e-12, then average
likelihood (sum divided by training-row count) with ftol 1e-12. Dividing an
objective by a positive constant preserves its exact argmax and constraint
set while conditioning SLSQP's numerical steps. Retry starting values are
feasible interior points; their .99 persistence is initialization only, never
a final fitted parameter restriction. Final admission still uses the same
1e-9 feasibility tolerance, strict stationarity check and convergence gate.
Each attempt records its start, raw/used parameters, likelihood in restored
summed units, objective scale, elapsed time and admission result.

The Student-likelihood stationarity condition assumes symmetric unit-variance
shocks. The empirical shock distribution can be asymmetric, and its bootstrap
can be serially dependent. Diagnostics therefore also record the empirical
negative-shock second moment and the corresponding *iid* variance persistence;
the Student fit's condition alone is not a proof of unconditional stationarity
for the dependent empirical forecast law.

## Bounded validation

`tests/test_garch_filtered_history.py`: twelve checks passed under CPython 3.12.13,
arch 8, one numerical thread. Includes scalar-recursion oracle, sign asymmetry,
common-row deterministic stationary indices, full-prefix Student fitting, IID
boundary-feasibility projection with variance-difference bound, training-only PPW
inputs, exact mean call-chain and 1/3-asset policy-rejoin simulations. Positive
and negative gamma stationarity-boundary cases, substantive-violation rejection
and retention of the bounded training-fit LRU across path clears are checked.
The retry test verifies that an inadmissible raw fit cannot silently become
used parameters and that a feasible starting point and stricter tolerance
are passed to the same likelihood fit.

`failure57_diagnosis.json` records the official single52 failure: GLDSIM,
1987-09-30, 1,960 training rows, converged fit with persistence
1.0000000000000004. The bounded repair/refilter result is recorded separately.

`admission_preflight.py` performs fit-only admission on all 52 assets and 51
canonical training prefixes per asset, without constructing forecast targets,
running paths or calculating OOS scores. `admission_preflight.json` records
all 2,652 passes, every attempt and prefix identity, complete numerical-source
hashes, dataset manifest/fingerprints, package versions and thread settings.
Training digests for official failures 57 and 67 match the failed-run manifests.
The final run used twelve processes: 8.449 seconds execution wall time after
data/task preparation, 69.129 summed fit seconds, zero failures and unchanged
source hashes. There were 2,631 initial-fit passes, three first-retry passes
and eighteen objective-preconditioned second-retry passes. This admits the
historical fits; it does not establish future-path stability or OOS accuracy.

The strict-summed-likelihood-only preflight is preserved as
`admission_preflight_strict_sum_failed.json`: sixteen prefixes still failed
SLSQP convergence/feasibility near the closed stationarity boundary. No
feasibility tolerance was widened. In the final preflight, failure 67 converged
under the preconditioned same likelihood with persistence .9999999999999992
and zero parameter projection. Its recorded summed likelihood agrees exactly
with the unscaled refiltered likelihood. Across all final admissions, the
largest allowed parameter-boundary correction was 7.0361e-10 and the largest
conditional-variance refilter change was 1.8522e-6 in percent-return variance
units; the complete per-prefix effects remain in the artifact.

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
