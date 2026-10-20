# Joint asset-level reference configurations, 1,000 simulations

The goal remains active. Improvement research starts from the better new asset configuration.
The worse configuration remains the sole scoring reference. The prior best
portfolio approach must also be tested as an asset-level counterpart. Earlier portfolio-level runs are preserved historical
experiments, not asset-level implementations or current reference scores.

Both configurations use the same frozen 1,264 target streams, 64,049 origins per
model and one main CRPS aggregation. The 722 supplied individual input units
include published French portfolio/index return series; their unavailable
underlying stock constituents are not invented. For the 542 constructed
portfolios, the actual constituent input series are kept as a matrix.

Training uses only observations available at the origin. Assets are aligned at
common observed valuation endpoints; all intervening observations are compounded,
including cryptocurrency weekends. Nothing is forward-filled or interpolated.
Joint historical indices select all constituent columns together. Each scenario
evolves constituent paths and holdings before calculating its portfolio return.
The target weights are restored at each common endpoint, matching the fixed
panel's target construction. Constant-weight equivalence to an aggregated series
is a verification check, not a substitute implementation. The accounting kernel
also tests drifting holdings and periodic rebalancing against direct arithmetic;
these do not add benchmark cases or alter the score.

## Frozen block profile

Block limits are configurable in profile.json. The declared comparison profile
samples integer block lengths uniformly from one observation through
min(training observations, forecast observations). It does not estimate an
optimal length and does not use Politis–White. Both configurations use these
same limits, raw historical returns and 1,000 paths.

* With replacement: draw a length, choose a uniformly distributed valid,
  contiguous, nonwrapping historical starting point, and repeat. Blocks can recur.
* Without replacement: partition the entire chronological history into random
  contiguous blocks, shuffle those blocks, and consume each row once. Repartition
  and reshuffle after every complete history cycle. The last partition block may
  be shorter than the configured minimum. This follows the user's approved
  refill rule for forecasts longer than history.

The public TestFolio documentation describes joint asset block sampling and
within-path portfolio accounting, configurable bounds and replacement choices.
The frozen support-bounded profile is an approximation, not their one-to-five-year
UI default. Their exact random-number stream and start-point distribution are not
publicly verified. Sources:
https://testfol.io/guides/monte-carlo-portfolio-simulator/ and
https://testfol.io/help/ . No claim to reproduce authenticated tool outputs.

## Score and reference

Keep the existing daily_portfolio_panel_crps_v4_train_sd_sqrt_h_logcalendar_two_branch
protocol unchanged: exact empirical CRPS of terminal cumulative log returns,
training SD times square-root observation horizon scaling, actual-calendar
log-lead masses, and the existing quarter/family/class aggregation. Constructed
portfolios have at most half the score. Diagnostics contribute zero.

Only after both configurations complete is the higher, worse main Q selected as
the sole reference. Candidate deltas use 100*(1-Q_candidate/Q_reference) on the
identical successful task intersection, explicitly better/worse and preliminary
if incomplete. This deliberately selected weaker reference is the user's
comparison rule, not evidence of superiority over every available baseline.
There is no reference selection from smoke results, no mixed-coverage leaderboard,
no invented fitting penalty and no significance or independence claim.

## Execution and preservation

The shared scheduler has 12 slots, reassigns freed workers, resumes immutable
source-bound checkpoints and permits source-bound later admissions. Horizon
columns are streamed in batches of 64; all 1,000 simulations enter each CRPS.
No N-by-horizon-by-assets path cube is retained. Actual constituent cumulative
states are computed directly, never derived from portfolio paths.

Qualification covers 70 families, short and long horizons, 154 forecasts and all
542 constructed-portfolio matrices at three causal prefix lengths. The same
154 losses are checked through the real scheduler, then resumed without new
forecasts. Full admission additionally requires verified closed-run preservation,
budget preflight and reproducible source publication. Smoke checkpoints are native
checkpoints from the same frozen full-run identity, not imports from prior models.

Implementations and compact evidence belong in the research repository. Large
source/data/loss archives remain in private repository-linked releases. Verify a
fresh download and physical per-file SHA-256 recovery before retiring local copies.
