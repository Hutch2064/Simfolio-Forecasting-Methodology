# Evidence for parameter-MCMC runtime optimization

See the [report](../../mcmc-runtime-optimization.md) and
[reproduction commands](../../../tools/mcmc_runtime/README.md).
All measurements are local research diagnostics. Production deployment and
browser click-to-visible performance are outside this evidence.

## Final full-panel evidence

- `final-full-panel-receipt.json`: all 80 portfolios, 51 origins per portfolio,
  4,080 tasks, 701,280 cells, exact score and complete-output comparison totals.
- `final-validated-source.zip`: all 73 executed Python sources (21 research
  tool files and 52 unchanged public package files), plus their run manifest.
- `final-manifest.json.gz`: complete task identities, runtime/model settings,
  source hashes, and reference-manifest binding.
- `final-task-parity.json.gz`: all 4,080 complete task comparison receipts.
- `final-unique-chain-checks.json.gz`: complete trace hashes and acceptance
  rates for all 5,098 distinct retained chain traces.
- `final-checkpoint-hashes.json.gz`: hashes of the private persisted loss
  checkpoints. Raw loss arrays can be regenerated with the portable reference
  generator and the archived source; they are not replaced by sampled checks.
- `final-reference-identity-audit.json`: literal task identity equality and
  equality of all 52 public source files to the original paired experiment.

Manifest digests in run receipts are SHA-256 of the sorted JSON serialization
used by the runners. They are distinct from hashes of the formatted JSON files.
Each checkpoint embeds its matching manifest digest. The source archive preserves
the literal executed bytes. `publication-source-audit.json` verifies executable
AST equality for all published sources; one final trailing blank line was removed
after replay, with no executable change. `evidence-index.json`
records file sizes and ordinary file-byte SHA-256 digests of this directory.
Gzip archives use deterministic zero timestamps.

## Original score and earlier validated implementations

`original-panel-results.json`, `original-panel-breadth.json`,
`original-manifest.json.gz`, and `original-checkpoint-hashes.json.gz` preserve
original paired Frontier/MCMC scoring evidence. The incumbent loss vectors
also match the prior Frontier reference exactly, as summarized by
`incumbent-reference-audit.json`.

The serial, parallel four-worker, and calibrated six-worker full-panel
receipts and validated source archives each describe a completed independent
replay. Their execution sources differ from the final source by subsequent
measured execution optimizations. `reference-identity-audit.json` is a
historical audit of the archived serial version, before publication lint
cleanup and GIL release. The final audit has a separate filename.

## Runtime and stopping evidence

`matched-model-cases.json` and `scaling-benchmark-final.json` retain every
interleaved timing sample; `runtime-medians.json` derives their medians.
`optimized-profile-*.txt` forces serial asset/chain fitting for decomposition,
while retaining parallel path computation. Nested profile times overlap.
Whole-process CPU timing includes numerical worker threads. Verification
replays are explicitly paused during isolated timing, as recorded by
`final-nogil-benchmark-isolation.json`; replay elapsed time is not a speed claim.

The finite-input, parallel-path, thread-budget, GIL-release, power/moment-cache,
EWMA/quantile, and random-draw buffer probes retain individual samples and
parity assertions. Some alternatives are intentionally rejected; their
presence is not an instruction to enable them. `test-receipt.json` records
complete installed-wheel tests, lint, the test/reference-generator source
hashes, and the wheel hash.

The [report](../../mcmc-runtime-optimization.md) distinguishes measured elapsed
speedup from CPU reduction, the original MCMC from Frontier, complete
same-platform numerical parity from cross-platform reproducibility, and
research-panel model selection from independent confirmation.
