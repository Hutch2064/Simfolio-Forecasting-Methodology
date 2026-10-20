# Filtered-shock dependence without a new fitted mean equation

The prior training-only245-prefix diagnostic found remaining signed lag1
correlations after Gaussian GARCH filtering: median0.10677 for equities,
0.08821 for fixed income, and0.06804 for FX. Signs were positive in both halves
of81%,86%,100% of their sampled prefixes, respectively; the FX group has only5
overlapping prefixes and these are not independent significance tests. Higher
lags were much weaker. An earlier explicit AR1 mean addition failed its reserved
predictive stage. That rejects that parameterization; it does not establish that
discarding all residual dependence is optimal.

This mechanism resamples consecutive *filtered innovations*, rather than adding
another AR mean or altering Gaussian likelihood fitting. Fit the exact original
variance-targeted Gaussian GARCH. Estimate a stationary-bootstrap average block
length using the existing Politis-White-style helper on its filtered innovations
and their centered squares, taking the maximum as Public Adaptive already does
on raw returns. Actual block lengths are geometric, historical starts uniform
with replacement, circular continuation. All estimation uses the training prefix.

Keep the original IID index matrix as possible restart locations. Draw independent
restart flags from a separately fixed namespace. At average length1 every flag
restarts, so the index matrix and entire240-path native forecast equal the original
winner byte for byte. The fit, mean Gaussian draws, 120/120 weights, native wealth
accounting, fixed panel and both v3 scores remain unchanged. Do not search seeds,
fixed block lengths, class routes or horizon cutoffs.

Serial dependence among innovations changes volatility-feedback covariances.
Accordingly do not claim the IID GARCH unconditional-variance identity still holds.
The original target remains its training fit scale; actual multi-period forecast
dispersion is evaluated rather than silently renormalized. Finite-path/native
accounting guards remain mandatory, and any boundary event or failed fit disqualifies
that field stage. No invented clipping or variance multiplier is added.

Supporting sources: the original FHS authors describe historical standardized
innovations at https://www.filteredhistoricalsimulation.com/. A direct recent
filtered/block-bootstrap financial application is the2025Risks article
https://www.mdpi.com/2227-9091/13/9/166. These motivate a test; their crypto-tail
results do not prove full-panel CRPS improvement. The independent training
diagnostic here is the main concrete reason for this experiment.

Five tests cover exact IID draw/whole-forecast boundaries, circular continuation,
restarts, stationary uniform marginals and expected continuation frequencies.
Qualify all245 training prefixes before any field scores. Use the frozen245
development cases,245 reserved and81-series/4122-origin representative coverage,
with fresh exact Public/current-winner forecasts on identical cases. All490
development baseline hashes must match the retained historical summaries.
These are adaptive reused research cases, not an independent holdout.

Advance only after complete finite failure-free stages with lower primary score
than the current winner; retain sensitivity/class/family/horizon regressions.
Full722 evaluation and catalogue preservation are mandatory before promotion.
One nice15 field worker, single-threaded libraries,1GiB pilot output budget under
the shared12GiB research cap and100GiB free reserve. Preserve rejected results,
freshly recover/checksum remote archives before local retirement. No production
changes; rejection of this contrast does not close the wider search.
