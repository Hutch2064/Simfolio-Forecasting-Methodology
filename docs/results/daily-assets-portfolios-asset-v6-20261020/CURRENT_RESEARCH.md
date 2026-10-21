# Joint-asset research: active, not complete

The fixed panel contains1264streams (722individual supplied assets/indices and
542constructed portfolios). All forecast methods simulate constituent asset
paths jointly and apply weights and rebalancing inside paths. One unchanged
main score is used. Diagnostics add no selection weight.

Both TestFolio-style baselines completed64049forecasts each at1000paths, with no
native failures or scoring integrity errors. The full independent audit verified
all128098checkpoint identities/hashes and the score reduction. Without replacement
has Q=0.6310508866260718; with replacement has Q=0.6320715457043177. The latter is
the fixed worse reporting reference. Without replacement is0.161478409%better on
that complete identical coverage. Recorded worker compute is12956.48seconds and
13316.04seconds respectively; this is not a controlled speed benchmark.

The prior winner was rebuilt at asset level: an equal whole-path pool of raw
joint blocks and per-asset GARCH(1,1) filtered historical simulation, with smooth
mean regularization and correlated Gaussian batch mean uncertainty. Its two
355-case pilots improved main Q versus the fixed reference by9.1045%and2.7760%.
A frozen7418-case broad test spanning every series and70families improved3.1842%.
All are matched preliminary subsets, not full-panel winner claims or independent
validation. No native or scoring failures occurred. Its complete64049-task
evaluation is now running at1000paths, reusing8128identical qualified checkpoints.
Official promotion waits for complete matched results and independent audit.

Separately, a100-path screen compared six arms across355cases each, sharing one
12-worker pool. These are exploratory results, not official scores or full-panel
catalogue additions. Correlated mean uncertainty improved the raw law in that
screen; Gaussian mean regularization helped further. The lower-path screen cannot
establish improvement against the1000-path hybrid. A further natural endpoint
contrast, filtered simulation without the raw half, is prepared and checked to
reproduce the hybrid's filtered component exactly without generating unused paths.
It has not yet produced forecast accuracy evidence.

All future official scores use1000paths. Preliminary screens may use fewer.
Concurrent candidate tests share12available forecast workers. Official retained
results contain both configurations and the current verified best, while the
append-only catalogue preserves every fully attempted historical implementation,
including losers. Current catalogue:20unique implementation versions,25fully
attempted executions. The running hybrid is not yet a completed catalogue entry.

Complete baseline data/source/loss evidence (134653files) was privately archived,
fresh downloaded, physically restored and SHA256verified. Public catalogue and
recovery metadata were freshly verified at commit3e1f8e524bd3582766a43fecd0d5c12e04c21197.
No scientific originals have been removed while they remain active research inputs.
The catalogue's publication check initially invoked unittest on pytest functions;
the correct runner passed all10tests. This did not alter forecasts or scores.

The goal remains active. A first winner, small gain or a few failed variants does
not satisfy the stopping requirement. Further credible mechanisms remain to test.
