# Historical training support review — 2026-10-21

This is an unresolved research mechanism, not an accuracy result, panel change
or admission. The running reference remains intact.

The selector's existing schedule uses every nonoverlapping completed block at
the requested horizon where possible. When history is too short, it chooses the
largest shorter horizon that admits two blocks and the numerical minimum-fit
prefix. That minimum is24observations when filtered components are eligible.
It is a numerical eligibility rule, not an established statistical stability
threshold. There is no maximum-origin cap.

The history-only72-request support diagnostic finds67requests with exactly two
folds and first training prefixes of24or25observations. In68requests, the first
validation horizon exceeds its training length. The largest first-fold ratio is
273.625. Historical information uncertainty and Monte Carlo error are separate;
raising simulation counts only addresses the latter.

These facts do not prove that the selector loses accuracy or that past outer
scores are invalid. They motivate examining how well the weight-learning
experiment represents a forecast fitted to the user's complete available
history. The already-qualified prefix-SD variant isolates one origin-scaling
contrast; it does not by itself solve or establish stable historical support.

Another precise contrast is available from the already-preserved historical
scoring implementation: internal scaling by s_train*sqrt(h+h*h/n_train), rather
than s_train*sqrt(h). It would reduce the relative influence of small-prefix,
long-lead historical losses without a manual minimum-history cutoff. This
reference scale comes from the IID mean-estimation prediction-error variance
sigma^2*(h+h*h/n), with sample SD plugged in here. It is not a general
finite-sample guarantee for dependent financial returns or volatility models.
Using it internally would be a new learning-objective candidate, not a change
to the fixed outer main score or a proven improvement. Its simulation, fitting
and final path accounting should be shared with unchanged controls in a
qualified representative comparison if admitted.

More simulation seeds, arbitrary minimum years, fixed exclusions of early
periods or score-selected origin cutoffs are not automatic solutions. Any new
origin rule or internal scale must use only the request's past data, disclose
its historical support and earn improvement on the same canonical main score.
Retain this issue for the evidence-based stopping review; neither legal fits,
two folds nor successful optimization certificates establish it is resolved.
