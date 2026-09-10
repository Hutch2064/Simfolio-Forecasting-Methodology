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

With identical marginal/dependence random streams and the two production
float32 storage steps, all three mapped asset-path arrays and rejoined
portfolio-path arrays are **byte-identical**, with maximum absolute difference
zero. The public computation is rerun in the parity test; these comparisons
are not based only on output dimensions or scalar scores.

Marginal parameter estimates match exactly. The largest observed difference
in Kalman intermediates is `8.1e-14`; coupled uniforms differ by at most
`7.8e-16`. These differences disappear at production's asset-path storage
precision. Both marginal paths and dependence-mapped paths are stored as
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
source. It does not claim a full white-paper rerun or reproduction of retained
CRPS `0.2557255171048505`.
