# Frontier bounded production-method comparison

Three bounded cases compare the standalone Frontier numerical implementation
with production source revision `1485d2088ccbee7fcbc7e81e358d253eef7b6058`.
Read-only hashes from the running production machine match current main
`974b79699d8b7e5a474d21c3f4209b18e4927ccf`. Its Frontier module is byte-identical
to the tested source. An AST comparison found no changes to the engine
forecasting numerical functions or module/class constants; intervening changes
affect benchmark diagnostics and backtest reporting. This verifies the live
numerical source, without claiming a complete deployment-image audit.

| Input | Training observations | Simulations | Daily horizons | Rebalancing |
| --- | ---: | ---: | ---: | --- |
| Synthetic four-asset fixture | 600 | 24 | 40 | Monthly |
| Frozen canonical portfolio 1 | 600 | 24 | 260 | Annual |
| Frozen canonical portfolio 7 | 600 | 24 | 70 | Quarterly |

The parity test keeps three contracts separate:

1. **Native fitted dependence.** `numpy.linalg.eigh` may choose either sign for
   each fitted eigenvector column. If the current fit is `L D`, where `D` is a
   diagonal matrix of `+1` or `-1`, the test compares `L`, the Kalman terminal
   state, and its covariance after the corresponding `D` transform. Mean,
   observations, residual variance, and AR parameters remain strict
   `2e-12` comparisons. A non-sign loading or state change still fails.
2. **Stored conditional simulation and rejoin.** The test rebuilds the
   dependence input from the stored fit arrays and recorded factor orientation,
   then reruns Gaussian uniforms, marginal mapping, and calendar rejoin. The
   stored source-versus-production arrays retain their `5e-14` uniform and
   `2e-12` conditional path tolerances. After the two production float32
   storage boundaries, mapped asset paths and rejoined portfolio paths remain
   exact array comparisons.
3. **Native adapter integration.** The `HistoricalFrontierModel` adapter is
   compared with the manually wired native pipeline under the same current fit,
   seed contract, and rebalance mask. This catches changes to the adapter's
   output or calendar/rebalance behavior.

The factor normals in `simulate_future_gaussian_uniforms` are injected by
factor coordinate. Therefore, a native same-seed simulation can change when
`D` changes, even though the fitted dependence is mathematically equivalent;
the test does not relabel such seeded paths as equal. The strict byte-equality
claim applies to the stored conditional replay with its frozen orientation and
to the persisted local source-versus-production evidence. The public native
fit is rerun for the first and third contracts; the conditional replay is
deliberately fed the stored fit state.

Marginal parameter estimates match exactly. The largest observed difference in
the stored Kalman intermediates is `8.1e-14`; coupled uniforms differ by at
most `7.8e-16`. Both marginal paths and dependence-mapped paths are stored as
float32 by production, before portfolio arithmetic in float64. The canonical
historical reference preserves its original float64 arithmetic.

The current production entrypoint draws a root seed from the caller RNG and
constructs per-asset seeds from it. The historical research adapter derives
streams from model identity, ticker, origin date, horizons, and simulation
count. Their default outputs consequently differ. The comparisons explicitly
align those random streams; changing the canonical seed schedule merely to
match a production default would change the historical experiment identity.

Machine-readable source hashes, input hashes, intermediate arrays, seed
contexts, and results are packaged under
`resources/test_fixtures/frontier/current_production_parity_report.json` and
its three referenced NPZ fixtures. Run the bounded acceptance test with:

```bash
python -m pytest tests/parity/test_frontier_current_production.py -q
```

This establishes bounded numerical agreement with the identified production
source under the three contracts above. The reference-platform fit orientation
is not a cross-platform identity guarantee, and this does not claim a full
white-paper rerun or reproduction of retained CRPS `0.2557255171048505`.
