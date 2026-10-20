# French551 public stationary-bootstrap refinement pilot

Status: Retired historical pilot. This bounded negative study remains available as evidence, but it is excluded from the active and restarted French551 comparisons.

This completed exploratory pilot compared the exact `TestingNaiveControl` baseline with two conditional-volatility refinements. It used 11 metadata-selected French portfolio series, four predeclared origins per series, 240 paths, and every eligible daily horizon. The pilot is separate from the 551-series campaign.

Neither refinement beat the baseline. The baseline scored 0.8087918052 on the primary v2 score; recursive EWMA residual SBB scored 0.9237505485 (14.21% higher loss), and the mean-reverting alternative scored 0.9601634738 (18.72% higher loss). All 44 tasks completed for each method, but the same pilot was used for the incremental alternative and there is no untouched validation subset. These findings reject these tested implementations only; they do not establish a full-panel ranking or significance result.

The full [methodology](METHODOLOGY.md), predeclared plans, compact per-series summaries, validation record, and source are included here. The per-task losses, fitted parameters, failure evidence, and 44 complete daily path arrays for each of the three variants are preserved in the private research archive. No candidate was added to the public model catalogue, and the production code and the running French551 campaign were not changed.
