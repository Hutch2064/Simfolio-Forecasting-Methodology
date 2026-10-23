# V11 scoring recovery

`source_manifest.json` binds exact public source bytes. The `source/` directory keeps the original named campaign and scorer import layout. It contains no price/return input arrays, scale database, forecast paths, credentials or customer records.

With the declared Python 3.12/NumPy research environment, the pure scoring fixtures run from a checkout with:

```sh
python -m pytest -q docs/results/daily-assets-portfolios-asset-v6-20261020/standardized-crps-v11-20261023/source/daily-cross-asset-panel-20261020/research/accuracy-maximization-20261020/requested_horizon_crps_v11/test_score_requested_horizon_crps_v11.py
```

The copied scorer/layout passed all nine fixtures. `public_source_qualification.json` records this limited check. It is not a forecast test or a fresh full-data reconstruction.

A complete private data replay additionally requires the frozen scientific plan, calendar map, panel/catalogue/input arrays, saved request means and matching prefix-scale sidecar. These are not public payloads. Use the private recovery manifests linked by the expanded V9 and development-audit evidence, preserving their original relative paths and every source/input hash. The V11 builder refuses changed sources, missing inputs, altered origin/horizon/date bindings or nonpositive scales. Rebuilding the scale sidecar reads training prefixes only; it performs no forecast fitting.

Historical V9 and V10 outputs keep their original score definitions. To reproduce a V11 result, use the versioned V11 protocol and source/scale/report hashes recorded with that result; do not overwrite old manifests or silently mix score definitions. The full-panel catalogue counts financial methods and executions, so rescoring alone adds neither.

Use [private_recovery_references.json](private_recovery_references.json) for the exact verified core and closed-study release URLs and receipt hashes. The core archive reconstructs the scale database from remote parts and links all3,792 training-input paths/hashes to the previously verified baseline archive.
