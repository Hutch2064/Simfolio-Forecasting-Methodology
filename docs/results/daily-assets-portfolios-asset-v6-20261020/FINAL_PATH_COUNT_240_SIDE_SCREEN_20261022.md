# Final-path-count side screen

A bounded development-side screen compared 240 with 1,000 final paths while holding the source, configuration, and two inner paths fixed. It covered 17 contexts (11 assets and 6 portfolios), 85 requested-horizon cases, five horizons per context, three outer seeds, and two methods: 1,020 forecasts total. The full panel and running sweep were unchanged. This is not a full-panel result or promotion decision.

| Final paths | Endpoint mean Q | Fixed WR mean Q | Endpoint improvement vs fixed WR | Seed-level SD of improvement |
|---:|---:|---:|---:|---:|
| 240 | 0.656440 | 0.719714 | 8.7915% | 1.345 pp |
| 1,000 | 0.654660 | 0.716890 | 8.6805% | 0.416 pp |

The 240-path endpoint mean Q is 0.2718% worse than at 1,000 paths. Its relative advantage over fixed WR is 0.1110 percentage points higher because the fixed reference also shifts; the three-seed variation is larger at 240 paths. This does not support reducing final paths.

An independent verifier replayed all 1,020 raw loss vectors with the canonical Q formula, reproduced all six seed/count aggregate cells, confirmed unchanged weight hashes across counts and seeds, and checked 170 allocations for the expected sum and evenness. No failures or scoring errors occurred.

Concurrent CPU timing reused selection cost across horizon counts, so timing is diagnostic and omitted. The source and internal configuration were unchanged, and this small reused development sample does not establish full-panel equivalence.

Full loss vectors, per-context records, plan, harness, and independent verification are preserved in the [private evidence release](https://github.com/Hutch2064/simfolio-engine/releases/tag/research-final-paths-240-check-v1-20261022T060430Z); private access is required. Archive receipt SHA-256: `2eb8499deef928b3c1b8f08259d13452731b0e4a67ede17f7b61814047930404`. Inventory SHA-256: `ac38daecbe28fe474d23c9ff309d9695f74165fa49a46441a9c8f22f6098898f`. Restore uses the release part manifest and checksum ledger in the receipt.

Machine-readable aggregate and recovery manifest: [`final_path_count_240_side_screen_20261022.json`](final_path_count_240_side_screen_20261022.json).
