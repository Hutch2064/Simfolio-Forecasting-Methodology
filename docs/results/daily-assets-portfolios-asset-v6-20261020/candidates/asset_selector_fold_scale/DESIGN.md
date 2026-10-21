# Prefix-normalized historical-fold pooling

One isolated selection-objective contrast, not an admitted panel model. Keep the
166 options, historical origins, calendar/native-lead weights, shared simulation
seeds, component fits and final joint-asset path accounting unchanged. Divide
each completed historical fold's CRPS coefficients by that fold's own training
portfolio log-return sample SD (ddof=1). Its scale is observed before that fold's
validation interval. A nonpositive scale fails explicitly, matching the main
score's rule; no floor, fallback or arbitrary mixing weight is added.

The prior selector omits this per-fold factor. Dividing all terms by the full
request's SD would cancel from the optimizer. Different prefix SDs do not cancel
across folds. The new rule aligns the historical path loss with the same
within-origin normalization used by the fixed main score; it does not change
the panel score, its aggregation or the portfolio branch weight.

The history-only72-request diagnostic found a median largest/smallest prefix-SD
ratio of1.4325703058 and maximum4.9765109448. These are descriptive mechanism
evidence, not forecast gains. All historical periods remain included.

The implementation is a separate source copy of the qualified fast coefficient
candidate with this normalization, a new identity and explicit scale trace and
bundle marker. Active/reference and queued sources are untouched. Aggregate
coefficient checkpoints from the old rule do not retain the individual folds,
so a scaled objective cannot be recovered from those totals alone.

Qualification checks independent canonical main-score reconstruction, prefix
causality, the no-floor rule, unit invariance for a single-asset toy, bundle
identity, streamed coefficient parity and a synthetic joint financial fixture
with1,000final paths. An accuracy test would first compare this rule and its
unchanged control on the same representative requests and internal precision,
within the shared12-worker budget. No real-panel result, admission, pruning,
production-precision choice, runtime gain or promotion is claimed here.
